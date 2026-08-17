# -*- coding: utf-8 -*-
"""
baseline_plain.py

Dùng cho model #1-8 trong model_tracking.xlsx (KHÔNG có tầng CRF, khác với
baseline_lstm_crf.py dùng cho #9-16):
  #1 mBERT                  --model_key mbert
  #2 vELECTRA                --model_key velectra
  #3 BERT                    --model_key bert
  #4 XLM-R                    --model_key xlmr
  #5 mBERT + Bi-LSTM         --model_key mbert    --use_bilstm
  #6 vELECTRA + Bi-LSTM      --model_key velectra --use_bilstm
  #7 BERT + Bi-LSTM          --model_key bert     --use_bilstm
  #8 XLM-R + Bi-LSTM         --model_key xlmr     --use_bilstm

Kiến trúc: backbone [+ BiLSTM] -> Linear classifier -> CrossEntropyLoss
(masked theo word_mask, không dùng CRF). Dùng CHUNG punc_dataset_word.py
(word-level gather) như mọi model khác để đảm bảo benchmark công bằng.

Checkpoint HuggingFace (giống hệt #9-16, đã chốt với Fa):
  mbert    -> bert-base-multilingual-cased
  velectra -> FPTAI/velectra-base-discriminator-cased
  bert     -> NlpHUST/vibert4news-base-cased  (cần BertTokenizer/BertModel trực
              tiếp, không dùng AutoTokenizer/AutoModel - xem ghi chú use_bert_encoder)
  xlmr     -> xlm-roberta-base

Dữ liệu CHỈ lấy từ data/punctuation/News/ và data/punctuation/Novels/
(train.txt/valid.txt/test.txt) - KHÔNG dùng punctuation/large/.
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

MODEL_DISPLAY_NAME = {
    ("mbert", False): "mBERT",                    # STT 1
    ("velectra", False): "vELECTRA",               # STT 2
    ("bert", False): "BERT",                        # STT 3
    ("xlmr", False): "XLM-R",                        # STT 4
    ("mbert", True): "mBERT + Bi-LSTM",             # STT 5
    ("velectra", True): "vELECTRA + Bi-LSTM",       # STT 6
    ("bert", True): "BERT + Bi-LSTM",                # STT 7
    ("xlmr", True): "XLM-R + Bi-LSTM",                # STT 8
}


# --------------------------------------------------------------------------- #
# Model - KHÔNG có CRF, chỉ classifier thường + CrossEntropyLoss (masked)
# --------------------------------------------------------------------------- #
class TransformerWordTagger(nn.Module):
    """Backbone (mBERT/vELECTRA/BERT/XLM-R) [+ BiLSTM] -> Linear classifier,
    KHÔNG có CRF. Gather word-level TRƯỚC BiLSTM/classifier, cùng cách xử lý
    dữ liệu như các model có CRF để so sánh công bằng."""

    def __init__(self, bert_model: str, num_labels: int = 7, use_bilstm: bool = False,
                 lstm_hidden_size: int = 128, dropout: float = 0.2, use_bert_encoder: bool = False):
        super().__init__()
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
        logits = self.classifier(word_hidden).float()

        if label_ids is not None:
            # CrossEntropyLoss per-token (reduction='none') rồi mask theo word_mask,
            # vì các vị trí pad được gán nhãn giả "O" (0) - không thể dùng ignore_index
            # đơn giản để phân biệt pad với nhãn "O" thật.
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            per_token_loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
            mask_flat = word_mask.view(-1).float()
            loss = (per_token_loss * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)
            return loss
        else:
            pred_ids = logits.argmax(dim=-1)  # (batch, max_words)
            mask = word_mask.bool()
            # Trả về dạng list-of-list giống CRF.decode() để evaluate() dùng chung logic
            batch_size = pred_ids.shape[0]
            result = []
            for i in range(batch_size):
                n = int(mask[i].sum().item())
                result.append(pred_ids[i, :n].tolist())
            return result


# --------------------------------------------------------------------------- #
# Train / eval (giống hệt logic trong baseline_lstm_crf.py để nhất quán)
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
                         help="Bật để train bản +Bi-LSTM (STT 5-8); tắt = plain (STT 1-4)")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--lstm_hidden_size", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=float, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_eval_examples", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=50)
    args = parser.parse_args()

    bert_model_name = BACKBONES[args.model_key]
    display_name = MODEL_DISPLAY_NAME[(args.model_key, args.use_bilstm)]
    use_bert_encoder = args.model_key in ("mbert", "bert")

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    use_amp = args.fp16 and device.type == "cuda"

    logger.info("=== %s | backbone=%s ===", display_name, bert_model_name)
    logger.info(json.dumps(vars(args), indent=2, ensure_ascii=False))

    # FIX: mbert/bert dùng WordPiece chuẩn - ép BertTokenizer/BertModel trực tiếp,
    # tránh lỗi AutoTokenizer/AutoModel không nhận diện được checkpoint cộng đồng cũ
    # (NlpHUST/vibert4news-base-cased thiếu tokenizer_class/model_type khai báo rõ).
    if use_bert_encoder:
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

    model = TransformerWordTagger(
        bert_model_name, num_labels=len(LABELS),
        use_bilstm=args.use_bilstm, lstm_hidden_size=args.lstm_hidden_size,
        use_bert_encoder=use_bert_encoder,
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
        model.train()
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
