#!/usr/bin/env bash
# run_phopunct_full.sh
#
# Chạy trên vast.ai (RTX 3090). Thứ tự: train Novels -> train News -> tự tắt instance.
# Đặt file này cùng cấp với phobert_lstm_crf.py, punc_dataset_word.py, data/ (đúng như tree Fa gửi).
#
# Cách dùng:
#   chmod +x run_phopunct_full.sh
#   ./run_phopunct_full.sh 2>&1 | tee run_phopunct_full.log
#
# An toàn khi lỗi: nếu 1 trong 2 lượt train fail, script SẼ KHÔNG tắt máy ngay -
# nó in lỗi, ngủ SLEEP_ON_ERROR_MIN phút (mặc định 30) để Fa kịp SSH vào xem log,
# rồi mới tắt (tránh vừa mất công debug vừa bị tính tiền vô thời hạn nếu Fa
# không online). Muốn tắt ngay khi lỗi (không chờ) thì set SLEEP_ON_ERROR_MIN=0.

set -uo pipefail

# --------------------------------------------------------------------------- #
# Cấu hình - sửa trực tiếp ở đây hoặc export biến môi trường trước khi chạy
# --------------------------------------------------------------------------- #
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$WORKDIR/data/punctuation}"
OUT_ROOT="${OUT_ROOT:-$WORKDIR/outputs}"
MODEL_NAME="${MODEL_NAME:-vinai/phobert-large}"

MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-256}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
LOGGING_STEPS="${LOGGING_STEPS:-50}"

SLEEP_ON_ERROR_MIN="${SLEEP_ON_ERROR_MIN:-30}"
SHUTDOWN_CMD="${SHUTDOWN_CMD:-sudo shutdown -h now}"

LOG_DIR="$WORKDIR/logs"
mkdir -p "$LOG_DIR" "$OUT_ROOT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

fail_and_wait() {
    local step_name="$1"
    log "!!! LỖI ở bước: $step_name — xem log tại $LOG_DIR"
    log "!!! Máy sẽ KHÔNG tắt ngay. Chờ ${SLEEP_ON_ERROR_MIN} phút để Fa kịp SSH vào kiểm tra."
    log "!!! Nếu Fa đã vào xem xong, có thể Ctrl+C để hủy đếm giờ, hoặc chờ hết giờ máy tự tắt."
    if [ "$SLEEP_ON_ERROR_MIN" -gt 0 ]; then
        sleep "$((SLEEP_ON_ERROR_MIN * 60))"
    fi
    log "Tắt máy sau lỗi (đã hết thời gian chờ)."
    eval "$SHUTDOWN_CMD"
    exit 1
}

# --------------------------------------------------------------------------- #
# 0. Kiểm tra môi trường trước khi train (fail fast, đỡ mất công chờ download
#    xong PhoBERT-large rồi mới báo thiếu thư viện)
# --------------------------------------------------------------------------- #
log "=== Kiểm tra môi trường ==="
nvidia-smi || { log "!!! Không thấy GPU (nvidia-smi lỗi)."; fail_and_wait "kiểm tra GPU"; }

python3 -c "import torch, transformers, torchcrf, pandas, sklearn; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())" \
    || { log "!!! Thiếu thư viện. Chạy: pip install torch transformers pytorch-crf pandas scikit-learn"; fail_and_wait "kiểm tra thư viện"; }

for f in phobert_lstm_crf.py punc_dataset_word.py; do
    [ -f "$WORKDIR/$f" ] || { log "!!! Không tìm thấy $f trong $WORKDIR"; fail_and_wait "kiểm tra file script"; }
done
for d in Novels News; do
    for split in train valid test; do
        [ -f "$DATA_ROOT/$d/$split.txt" ] || { log "!!! Thiếu $DATA_ROOT/$d/$split.txt"; fail_and_wait "kiểm tra dữ liệu"; }
    done
done
log "Môi trường OK. DATA_ROOT=$DATA_ROOT | MODEL_NAME=$MODEL_NAME"

cd "$WORKDIR"

# --------------------------------------------------------------------------- #
# 1. Train Novels
# --------------------------------------------------------------------------- #
log "=== [1/2] Bắt đầu train PhoPunct trên Novels ==="
python3 phobert_lstm_crf.py \
    --data_dir "$DATA_ROOT/Novels" \
    --model_name_or_path "$MODEL_NAME" \
    --output_dir "$OUT_ROOT/phopunct_novels" \
    --device cuda --fp16 \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --logging_steps "$LOGGING_STEPS" \
    2>&1 | tee "$LOG_DIR/phopunct_novels.log"

if [ ! -f "$OUT_ROOT/phopunct_novels/best_checkpoint.pt" ]; then
    fail_and_wait "train Novels (không thấy best_checkpoint.pt)"
fi
log "=== [1/2] Xong Novels. Best F1: $(grep 'Best macro F1' "$OUT_ROOT/phopunct_novels/eval_results.txt" || echo 'không đọc được eval_results.txt') ==="

# --------------------------------------------------------------------------- #
# 2. Train News
# --------------------------------------------------------------------------- #
log "=== [2/2] Bắt đầu train PhoPunct trên News ==="
python3 phobert_lstm_crf.py \
    --data_dir "$DATA_ROOT/News" \
    --model_name_or_path "$MODEL_NAME" \
    --output_dir "$OUT_ROOT/phopunct_news" \
    --device cuda --fp16 \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --logging_steps "$LOGGING_STEPS" \
    2>&1 | tee "$LOG_DIR/phopunct_news.log"

if [ ! -f "$OUT_ROOT/phopunct_news/best_checkpoint.pt" ]; then
    fail_and_wait "train News (không thấy best_checkpoint.pt)"
fi
log "=== [2/2] Xong News. Best F1: $(grep 'Best macro F1' "$OUT_ROOT/phopunct_news/eval_results.txt" || echo 'không đọc được eval_results.txt') ==="

# --------------------------------------------------------------------------- #
# 3. Xong cả 2 -> tắt máy
#    LƯU Ý: nhớ tải kết quả về (scp) TRƯỚC KHI script này chạy, hoặc chạy nó
#    trong 1 lần scp riêng sau bước 2 nếu Fa muốn chắc chắn giữ được checkpoint.
#    Script này CHƯA tự scp vì cần host/port/key riêng của từng instance vast.ai.
# --------------------------------------------------------------------------- #
log "=== Train xong cả Novels + News. Checkpoint tại $OUT_ROOT/phopunct_novels và $OUT_ROOT/phopunct_news ==="
log "=== Chuẩn bị tắt máy sau 60 giây (Ctrl+C để hủy nếu cần giữ máy tải file trước) ==="
sleep 60
eval "$SHUTDOWN_CMD"
