# -*- coding: utf-8 -*-
"""
baseline_lstm_crf.py

Dùng chung cho model #9-16 trong model_tracking.xlsx:
  #9  mBERT + CRF                  --model_key mbert
  #10 vELECTRA + CRF               --model_key velectra
  #11 BERT + CRF                   --model_key bert
  #12 XLM-R + CRF                  --model_key xlmr
  #13 mBERT + Bi-LSTM + CRF        --model_key mbert   --use_bilstm
  #14 vELECTRA + Bi-LSTM + CRF     --model_key velectra --use_bilstm
  #15 BERT + Bi-LSTM + CRF         --model_key bert     --use_bilstm
  #16 XLM-R + Bi-LSTM + CRF        --model_key xlmr     --use_bilstm

Vì đang retrain lại toàn bộ từ đầu (không checkpoint cũ nào giữ lại), cả 8 model
này dùng CHUNG cách xử lý dữ liệu word-level với punc_dataset_word.py (giống
PhoPunct) thay vì cách subword-level CRF gốc trong journal_hero (bert_crf.py,
bert_lstm_crf.py, ...). Lý do: đảm bảo 17 model so sánh trên cùng 1 pipeline
dữ liệu -> benchmark nội bộ công bằng, không lẫn khác biệt do cách xử lý data.

Checkpoint HuggingFace (đã chốt với Fa, base 12-layer/768-hidden theo đúng mô
tả trong "An efficient transformer-based model for Vietnamese punctuation
prediction"):
  mbert    -> bert-base-multilingual-cased
  velectra -> FPTAI/velectra-base-discriminator-cased
  bert     -> NlpHUST/vibert4news-base-cased
  xlmr     -> xlm-roberta-base

Dữ liệu CHỈ lấy từ data/punctuation/News/ và data/punctuation/Novels/
(train.txt/valid.txt/test.txt) - KHÔNG dùng punctuation/large/ (đã bị cấm).

Chạy CPU smoke test hoặc GPU full train đều dùng chung file này, giống
phobert_lstm_crf.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchcrf import CRF
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer, get_linear_schedule_with_warmup

from punc_dataset_word import LABELS, ID2LABEL, load_dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BACKBONES = {
    "mbert": "bert-base-multilingual-cased",
    "velectra": "FPTAI/velectra-base-discriminator-cased",
    "bert": "NlpHUST/vibert4news-base-cased",
    "xlmr": "xlm-roberta-base",
}

# tên hiển thị khớp đúng cột trong model_tracking.xlsx
MODEL_DISPLAY_NAME = {
    ("mbert", False): "mBERT + CRF",                    # STT 9
    ("velectra", False): "vELECTRA + CRF",               # STT 10
    ("bert", False): "BERT + CRF",                        # STT 11
    ("xlmr", False): "XLM-R + CRF",                        # STT 12
    ("mbert", True): "mBERT + Bi-LSTM + CRF",             # STT 13
    ("velectra", True): "vELECTRA + Bi-LSTM + CRF",       # STT 14
    ("bert", True): "BERT + Bi-LSTM + CRF",                # STT 15
    ("xlmr", True): "XLM-R + Bi-LSTM + CRF",                # STT 16
}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class TransformerWordCrf(nn.Module):
    """Backbone (mBERT/vELECTRA/BERT/XLM-R) [+ BiLSTM] + CRF, gather word-level
    TRƯỚC BiLSTM/classifier, cùng kiến trúc-hoá với PhoBertLstmCrf để so sánh
    công bằng giữa 17 model."""

    def __init__(self, bert_model: str, num_labels: int = 7, use_bilstm: bool = False,
                 lstm_hidden_size: int = 128, dropout: float = 0.2, use_bert_encoder: bool = False):
        super().__init__()
        # FIX: 1 số checkpoint BERT cộng đồng cũ (vd. NlpHUST/vibert4news-base-cased) có
        # config.json thiếu field "model_type", khiến AutoModel không tự nhận diện được
        # kiến trúc dù bản thân model là BERT chuẩn. Ép dùng BertModel trực tiếp cho các
        # trường hợp này (BertModel không cần field "model_type" vì đã biết sẵn là BERT).
        if use_bert_encoder:
            self.bert = BertModel.from_pretrained(bert_model)
        else:
            self.bert = AutoModel.from_pretrained(bert_model)
        hidden = self.bert.config.hidden_size
        self.use_bilstm = use_bilstm
        if use_bilstm:
            self.lstm = nn.LSTM(
                input_size=hidden, hidden_size=lstm_hidden_size,
                num_layers=1, batch_first=True, bidirectional=True,
            )
            classifier_in = lstm_hidden_size * 2
        else:
            self.lstm = None
            classifier_in = hidden
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(classifier_in, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels

    def _gather_word_level(self, sequence_output: torch.Tensor, word_starts: torch.Tensor) -> torch.Tensor:
        batch_size, _, feat_dim = sequence_output.shape
        max_words = word_starts.shape[1]
        word_hidden = torch.zeros(batch_size, max_words, feat_dim,
                                   dtype=sequence_output.dtype, device=sequence_output.device)
        for i in range(batch_size):
            idx = torch.nonzero(word_starts[i], as_tuple=False).squeeze(-1)
            n = idx.numel()
            if n == 0:
                continue
            word_hidden[i, :n] = sequence_output[i, idx]
        return word_hidden

    def forward(self, input_ids, attention_mask, word_starts, label_ids=None, word_mask=None):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        word_hidden = self._gather_word_level(bert_out, word_starts)
        if self.use_bilstm:
            word_hidden, _ = self.lstm(word_hidden)
        word_hidden = self.dropout(word_hidden)
        logits = self.classifier(word_hidden).float()  # cast float32 trước CRF (né NaN fp16)

        if label_ids is not None:
            mask = word_mask.bool()
            log_likelihood = self.crf(logits, label_ids, mask=mask, reduction="mean")
            return -1.0 * log_likelihood
        else:
            mask = word_mask.bool() if word_mask is not None else None
            return self.crf.decode(logits, mask=mask)


# --------------------------------------------------------------------------- #
# Train / eval  (giống hệt logic trong phobert_lstm_crf.py để nhất quán)
# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, device) -> dict:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, word_starts, label_ids, word_mask = (t.to(device) for t in batch)
            pred_seqs = model(input_ids, attention_mask, word_starts, label_ids=None, word_mask=word_mask)
            gold = label_ids.cpu().numpy()
            mask = word_mask.cpu().numpy()
            for i, seq in enumerate(pred_seqs):
                n = int(mask[i].sum())
                y_true.extend(ID2LABEL[t] for t in gold[i][:n])
                y_pred.extend(ID2LABEL[t] for t in seq[:n])

    from sklearn.metrics import classification_report
    punc_marks = ["PERIOD", "COMMA", "COLON", "QMARK", "EXCLAM", "SEMICOLON"]
    report_dict = classification_report(
        y_true, y_pred, labels=punc_marks, digits=4, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        y_true, y_pred, labels=punc_marks, digits=4, zero_division=0
    )
    return {"macro_f1": report_dict["macro avg"]["f1-score"], "report": report_str, "report_dict": report_dict}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", required=True, choices=list(BACKBONES.keys()),
                         help="mbert | velectra | bert | xlmr")
    parser.add_argument("--use_bilstm", action="store_true",
                         help="Bật để train bản +Bi-LSTM+CRF (STT 13-16); tắt = chỉ +CRF (STT 9-12)")
    parser.add_argument("--data_dir", required=True, help="Thư mục chứa train.txt/valid.txt/test.txt (News/ hoặc Novels/)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--lstm_hidden_size", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=float, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_eval_examples", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=10)
    args = parser.parse_args()

    bert_model_name = BACKBONES[args.model_key]
    display_name = MODEL_DISPLAY_NAME[(args.model_key, args.use_bilstm)]

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    use_amp = args.fp16 and device.type == "cuda"

    logger.info("=== %s | backbone=%s ===", display_name, bert_model_name)
    logger.info(json.dumps(vars(args), indent=2, ensure_ascii=False))

    # FIX: mbert/bert dùng WordPiece chuẩn (vocab.txt) - ép dùng BertTokenizer trực tiếp
    # thay vì AutoTokenizer, vì 1 số checkpoint (vd. NlpHUST/vibert4news-base-cased) không
    # khai báo rõ "tokenizer_class" trong tokenizer_config.json, khiến AutoTokenizer đoán
    # nhầm sang nhánh cần sentencepiece dù model không dùng sentencepiece.
    if args.model_key in ("mbert", "bert"):
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name, use_fast=False)

    logger.info("Đang nạp dữ liệu train...")
    train_dataset = load_dataset(args.data_dir, "train", tokenizer, args.max_seq_length,
                                  max_examples=args.max_train_examples)
    logger.info("Đang nạp dữ liệu valid...")
    valid_dataset = load_dataset(args.data_dir, "valid", tokenizer, args.max_seq_length,
                                  max_examples=args.max_eval_examples)
    logger.info("Train: %d features | Valid: %d features", len(train_dataset), len(valid_dataset))

    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.train_batch_size)
    valid_loader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset), batch_size=args.eval_batch_size)

    model = TransformerWordCrf(
        bert_model_name, num_labels=len(LABELS),
        use_bilstm=args.use_bilstm, lstm_hidden_size=args.lstm_hidden_size,
        use_bert_encoder=(args.model_key in ("mbert", "bert")),
    ).to(device)

    total_steps = max(1, int(len(train_loader) * args.num_train_epochs))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio), num_training_steps=total_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_f1 = -1.0
    global_step = 0
    n_epochs = max(1, round(args.num_train_epochs))

    for epoch in range(n_epochs):
        model.train()  # reset train() mỗi epoch (fix bug đã biết: evaluate() để sót eval mode)
        epoch_loss = 0.0
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            input_ids, attention_mask, word_starts, label_ids, word_mask = (t.to(device) for t in batch)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = model(input_ids, attention_mask, word_starts, label_ids=label_ids, word_mask=word_mask)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1
            if global_step % args.logging_steps == 0:
                logger.info("[%s] epoch %d step %d/%d loss=%.4f",
                             display_name, epoch + 1, step + 1, len(train_loader), loss.item())

        logger.info("[%s] Epoch %d xong sau %.1fs, avg_loss=%.4f", display_name, epoch + 1,
                     time.time() - t0, epoch_loss / max(1, len(train_loader)))

        metrics = evaluate(model, valid_loader, device)
        logger.info("[%s] Epoch %d - Valid macro F1 = %.4f\n%s",
                     display_name, epoch + 1, metrics["macro_f1"], metrics["report"])

        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch + 1,
            "best_f1": max(best_f1, metrics["macro_f1"]),
            "model_key": args.model_key,
            "use_bilstm": args.use_bilstm,
            "display_name": display_name,
        }
        torch.save(ckpt, os.path.join(args.output_dir, "last_checkpoint.pt"))

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save(ckpt, os.path.join(args.output_dir, "best_checkpoint.pt"))
            with open(os.path.join(args.output_dir, "eval_results.txt"), "w", encoding="utf-8") as f:
                f.write(f"{display_name}\nBest macro F1: {best_f1:.4f}\n\n")
                f.write(metrics["report"])
            logger.info("-> [%s] best checkpoint mới, macro F1 = %.4f", display_name, best_f1)

    logger.info("[%s] Train xong. Best macro F1 = %.4f. Checkpoint tại %s",
                 display_name, best_f1, args.output_dir)


if __name__ == "__main__":
    main()
