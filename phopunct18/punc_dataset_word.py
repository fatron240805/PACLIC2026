# -*- coding: utf-8 -*-
"""
punc_dataset_word.py

Chuẩn bị dữ liệu ở mức WORD-LEVEL cho PhoBertLstmCrf (model #17 - PhoPunct).

Khác với punc_dataset.py gốc (dùng cho model #9-16: mBERT/vELECTRA/BERT/XLM-R
[+BiLSTM][+CRF]), file này gather biểu diễn subword đầu tiên của mỗi từ
TRƯỚC khi đưa vào BiLSTM, không dùng sentinel [CLS]/[SEP] trong label space
(7 lớp thuần: O, PERIOD, COMMA, COLON, QMARK, EXCLAM, SEMICOLON).

Fix 2 bug đã phát hiện trong pipeline gốc:
  (1) Truncation: cắt theo SỐ TỪ (word-level) ngay từ đầu, không cắt sau khi
      đã sinh mảng theo subword-index rồi mới slice mảng label/level khác
      -> tránh lệch label-token.
  (2) Padding: vòng lặp pad input_ids/attention_mask và vòng lặp pad
      label_ids/word_mask TÁCH RIÊNG, không dùng chung 1 while-loop để
      word_mask không bị ghi đè thành toàn số 1.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

LABELS: List[str] = ["O", "PERIOD", "COMMA", "COLON", "QMARK", "EXCLAM", "SEMICOLON"]
LABEL2ID: Dict[str, int] = {l: i for i, l in enumerate(LABELS)}
ID2LABEL: Dict[int, str] = {i: l for l, i in LABEL2ID.items()}
EOS_MARKS = ["PERIOD", "QMARK", "EXCLAM"]


# --------------------------------------------------------------------------- #
# 1. Đọc file .txt "token label" -> cắt thành đoạn tối đa ~128 từ tại ranh
#    giới câu hợp lệ (giống readfile() gốc trong punc_dataset.py)
# --------------------------------------------------------------------------- #
def read_examples(filepath: str, max_words_per_chunk: int = 128) -> List[Tuple[List[str], List[str]]]:
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {filepath}")

    tokens, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                logger.warning("Bỏ qua dòng lỗi format tại %s:%d -> %r", filepath, line_no, line)
                continue
            tok, lab = parts
            if lab not in LABEL2ID:
                logger.warning("Nhãn lạ '%s' tại %s:%d, coi như 'O'", lab, filepath, line_no)
                lab = "O"
            tokens.append(tok)
            labels.append(lab)

    df = pd.DataFrame({"token": tokens, "label": labels})
    n = len(df)
    examples: List[Tuple[List[str], List[str]]] = []
    idx = 0
    while 0 <= idx < n:
        sub_df = df.iloc[idx: min(idx + max_words_per_chunk, n)]
        end_idx = sub_df[sub_df.label.isin(EOS_MARKS)].tail(1).index
        if end_idx.empty:
            chunk = df.iloc[idx:]
            next_idx = -1
        else:
            cut = end_idx.item() + 1
            chunk = df.iloc[idx:cut]
            next_idx = cut
        if len(chunk) > 0:
            examples.append((chunk.token.tolist(), chunk.label.tolist()))
        idx = next_idx
    return examples


# --------------------------------------------------------------------------- #
# 2. Convert sang feature tensor mức từ (word-level)
# --------------------------------------------------------------------------- #
@dataclass
class WordFeature:
    input_ids: List[int]        # subword-level, len = max_seq_length
    attention_mask: List[int]   # subword-level
    word_starts: List[int]      # subword-level, 1 tại subword đầu của mỗi từ
    label_ids: List[int]        # word-level, len = max_seq_length (>= số từ thật)
    word_mask: List[int]        # word-level, 1 cho vị trí từ thật


def convert_examples_to_features(
    examples: List[Tuple[List[str], List[str]]],
    tokenizer,
    max_seq_length: int = 256,
) -> List[WordFeature]:
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # cache tokenizer.encode theo unique word-type để tăng tốc trên corpus lớn
    encode_cache: Dict[str, List[int]] = {}

    def encode_word(word: str) -> List[int]:
        if word not in encode_cache:
            encode_cache[word] = tokenizer.encode(word, add_special_tokens=False)
        return encode_cache[word]

    features: List[WordFeature] = []
    n_dropped_empty = 0
    n_truncated = 0

    budget = max_seq_length - 2  # chừa chỗ cho <s> và </s>

    for words, labels in examples:
        subword_ids: List[int] = []
        word_starts: List[int] = []
        kept_label_ids: List[int] = []
        truncated = False

        for w, lab in zip(words, labels):
            piece_ids = encode_word(w)
            if not piece_ids:
                continue
            # FIX (1): cắt tại RANH GIỚI TỪ, kiểm tra ngân sách subword trước
            # khi thêm cả từ vào, không bao giờ cắt giữa 1 từ / lệch mảng.
            if len(subword_ids) + len(piece_ids) > budget:
                truncated = True
                break
            subword_ids.extend(piece_ids)
            word_starts.extend([1] + [0] * (len(piece_ids) - 1))
            kept_label_ids.append(LABEL2ID[lab])

        if not kept_label_ids:
            n_dropped_empty += 1
            continue
        if truncated:
            n_truncated += 1

        input_ids = [bos_id] + subword_ids + [eos_id]
        word_starts_full = [0] + word_starts + [0]
        attention_mask = [1] * len(input_ids)

        # FIX (2): 2 vòng pad TÁCH RIÊNG cho không gian subword và không gian từ
        while len(input_ids) < max_seq_length:
            input_ids.append(pad_id)
            attention_mask.append(0)
            word_starts_full.append(0)

        label_ids = list(kept_label_ids)
        word_mask = [1] * len(kept_label_ids)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)   # 0 = "O", nhưng bị mask nên không ảnh hưởng loss
            word_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(word_starts_full) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(word_mask) == max_seq_length
        assert sum(word_starts_full) == sum(word_mask), "Số word_starts phải khớp số từ thật trong word_mask"

        features.append(WordFeature(
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_starts=word_starts_full,
            label_ids=label_ids,
            word_mask=word_mask,
        ))

    logger.info(
        "convert_examples_to_features: %d examples -> %d features (%d rỗng bị bỏ, %d bị truncate)",
        len(examples), len(features), n_dropped_empty, n_truncated,
    )
    return features


def features_to_dataset(features: List[WordFeature]) -> TensorDataset:
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    word_starts = torch.tensor([f.word_starts for f in features], dtype=torch.long)
    label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    word_mask = torch.tensor([f.word_mask for f in features], dtype=torch.long)
    return TensorDataset(input_ids, attention_mask, word_starts, label_ids, word_mask)


def load_dataset(data_dir: str, split: str, tokenizer, max_seq_length: int, max_examples: int | None = None):
    """split in {'train', 'valid', 'test'}"""
    fname = {"train": "train.txt", "valid": "valid.txt", "test": "test.txt"}[split]
    filepath = os.path.join(data_dir, fname)
    examples = read_examples(filepath)
    if max_examples is not None:
        examples = examples[:max_examples]
    features = convert_examples_to_features(examples, tokenizer, max_seq_length)
    return features_to_dataset(features)
