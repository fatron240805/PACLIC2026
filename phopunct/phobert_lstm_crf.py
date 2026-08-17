# -*- coding: utf-8 -*-
"""
phobert_lstm_crf.py

Model #17 trong model_tracking.xlsx: PhoBERT-large + BiLSTM + CRF (PhoPunct).

Kiến trúc (đúng theo ghi chú đã confirm trước đó):
  - PhoBertLstmCrf là nn.Module thuần, KHÔNG kế thừa AutoModelForTokenClassification
  - self.bert = AutoModel.from_pretrained(...)
  - Gather biểu diễn subword đầu tiên của mỗi từ (word_starts) TRƯỚC BiLSTM
  - nn.LSTM(hidden_size=128, num_layers=1, bidirectional=True) trên chuỗi mức từ
  - nn.Linear classifier -> CRF(num_labels, batch_first=True), mask=word_mask
  - Logits cast sang float32 trước khi vào CRF (né NaN khi chạy fp16 trên GPU)

Bug đã fix và giữ nguyên trong file này:
  - model.train() được set lại đầu MỖI epoch (không chỉ set 1 lần đầu train)
  - CRF mask: vị trí đầu tiên (index 0) luôn là từ thật -> hợp lệ với torchcrf

Dùng chung cho cả 2 chế độ:
  - CPU smoke test: --device cpu --model_name_or_path vinai/phobert-base
    (hoặc phobert-large nếu máy đủ RAM) --max_train_examples 50 --num_train_epochs 1
  - GPU full train: --device cuda --model_name_or_path vinai/phobert-large --fp16
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
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from punc_dataset_word import LABELS, ID2LABEL, load_dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class PhoBertLstmCrf(nn.Module):
    def __init__(self, bert_model: str, num_labels: int = 7, lstm_hidden_size: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        hidden = self.bert.config.hidden_size
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels

    def _gather_word_level(self, sequence_output: torch.Tensor, word_starts: torch.Tensor) -> torch.Tensor:
        """Gather biểu diễn subword đầu tiên của mỗi từ -> chuỗi mức từ (padded)."""
        batch_size, _, feat_dim = sequence_output.shape
        max_words = word_starts.shape[1]  # dùng chung độ dài max_seq_length làm upper-bound
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
        lstm_out, _ = self.lstm(word_hidden)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out).float()  # cast float32 trước CRF (fix NaN fp16)

        if label_ids is not None:
            mask = word_mask.bool()
            log_likelihood = self.crf(logits, label_ids, mask=mask, reduction="mean")
            return -1.0 * log_likelihood
        else:
            mask = word_mask.bool() if word_mask is not None else None
            return self.crf.decode(logits, mask=mask)


# --------------------------------------------------------------------------- #
# Train / eval
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
    macro_f1 = report_dict["macro avg"]["f1-score"]
    return {"macro_f1": macro_f1, "report": report_str, "report_dict": report_dict}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Thư mục chứa train.txt/valid.txt/test.txt")
    parser.add_argument("--model_name_or_path", default="vinai/phobert-large")
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
    parser.add_argument("--fp16", action="store_true", help="Chỉ có tác dụng khi --device cuda")
    parser.add_argument("--max_train_examples", type=int, default=None,
                         help="Giới hạn số đoạn train (dùng cho CPU smoke test)")
    parser.add_argument("--max_eval_examples", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    use_amp = args.fp16 and device.type == "cuda"

    logger.info("=== Cấu hình chạy ===")
    logger.info(json.dumps(vars(args), indent=2, ensure_ascii=False))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    logger.info("Đang nạp dữ liệu train...")
    train_dataset = load_dataset(args.data_dir, "train", tokenizer, args.max_seq_length,
                                  max_examples=args.max_train_examples)
    logger.info("Đang nạp dữ liệu valid...")
    valid_dataset = load_dataset(args.data_dir, "valid", tokenizer, args.max_seq_length,
                                  max_examples=args.max_eval_examples)
    logger.info("Train: %d features | Valid: %d features", len(train_dataset), len(valid_dataset))

    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.train_batch_size)
    valid_loader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset), batch_size=args.eval_batch_size)

    model = PhoBertLstmCrf(args.model_name_or_path, num_labels=len(LABELS),
                            lstm_hidden_size=args.lstm_hidden_size).to(device)

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
        model.train()  # FIX: set lại train() mỗi epoch, không để sót sau evaluate()
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
                logger.info("epoch %d step %d/%d loss=%.4f", epoch + 1, step + 1, len(train_loader), loss.item())

        logger.info("Epoch %d xong sau %.1fs, avg_loss=%.4f", epoch + 1, time.time() - t0,
                     epoch_loss / max(1, len(train_loader)))

        metrics = evaluate(model, valid_loader, device)
        logger.info("Epoch %d - Valid macro F1 = %.4f\n%s", epoch + 1, metrics["macro_f1"], metrics["report"])

        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch + 1,
            "best_f1": max(best_f1, metrics["macro_f1"]),
        }
        torch.save(ckpt, os.path.join(args.output_dir, "last_checkpoint.pt"))

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save(ckpt, os.path.join(args.output_dir, "best_checkpoint.pt"))
            with open(os.path.join(args.output_dir, "eval_results.txt"), "w", encoding="utf-8") as f:
                f.write(f"Best macro F1: {best_f1:.4f}\n\n")
                f.write(metrics["report"])
            logger.info("-> best checkpoint mới, macro F1 = %.4f", best_f1)

    logger.info("Train xong. Best macro F1 = %.4f. Checkpoint tại %s", best_f1, args.output_dir)


if __name__ == "__main__":
    main()
