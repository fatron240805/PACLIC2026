#!/usr/bin/env bash
# run_baseline_full.sh
#
# Dùng chung cho model #9-16 (mBERT/vELECTRA/BERT/XLM-R x [CRF]/[BiLSTM+CRF]).
# Thứ tự: train Novels -> train News -> tự tắt instance. Giống hệt logic
# run_phopunct_full.sh, chỉ đổi sang gọi baseline_lstm_crf.py.
#
# Cách dùng:
#   chmod +x run_baseline_full.sh
#   ./run_baseline_full.sh --model_key xlmr --use_bilstm     # model #16
#   ./run_baseline_full.sh --model_key mbert                 # model #9 (không BiLSTM)
#
# An toàn khi lỗi: giống run_phopunct_full.sh - lỗi thì KHÔNG tắt máy ngay,
# chờ SLEEP_ON_ERROR_MIN phút (mặc định 30) rồi mới tắt.

set -uo pipefail

# --------------------------------------------------------------------------- #
# Parse --model_key / --use_bilstm (còn lại forward hết cho baseline_lstm_crf.py
# qua các biến môi trường override bên dưới nếu cần)
# --------------------------------------------------------------------------- #
MODEL_KEY=""
USE_BILSTM_FLAG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --model_key) MODEL_KEY="$2"; shift 2 ;;
        --use_bilstm) USE_BILSTM_FLAG="--use_bilstm"; shift 1 ;;
        *) echo "Tham số không nhận diện: $1"; exit 1 ;;
    esac
done
if [ -z "$MODEL_KEY" ]; then
    echo "Thiếu --model_key (mbert|velectra|bert|xlmr). Ví dụ: ./run_baseline_full.sh --model_key xlmr --use_bilstm"
    exit 1
fi

# --------------------------------------------------------------------------- #
# Cấu hình - sửa trực tiếp hoặc export biến môi trường trước khi chạy
# --------------------------------------------------------------------------- #
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$WORKDIR/data/punctuation}"
OUT_ROOT="${OUT_ROOT:-$WORKDIR/outputs}"

MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-256}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
LOGGING_STEPS="${LOGGING_STEPS:-50}"

SLEEP_ON_ERROR_MIN="${SLEEP_ON_ERROR_MIN:-30}"
SHUTDOWN_CMD="${SHUTDOWN_CMD:-sudo shutdown -h now}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"   # set AUTO_SHUTDOWN=0 nếu muốn giữ máy sống sau khi xong

RUN_TAG="${MODEL_KEY}$( [ -n "$USE_BILSTM_FLAG" ] && echo "_bilstm" )"
LOG_DIR="$WORKDIR/logs"
mkdir -p "$LOG_DIR" "$OUT_ROOT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$RUN_TAG] $*"; }

fail_and_wait() {
    local step_name="$1"
    log "!!! LỖI ở bước: $step_name — xem log tại $LOG_DIR"
    log "!!! Chờ ${SLEEP_ON_ERROR_MIN} phút để Fa kịp SSH vào kiểm tra trước khi tắt máy."
    if [ "$SLEEP_ON_ERROR_MIN" -gt 0 ]; then
        sleep "$((SLEEP_ON_ERROR_MIN * 60))"
    fi
    if [ "$AUTO_SHUTDOWN" = "1" ]; then
        log "Tắt máy sau lỗi (đã hết thời gian chờ)."
        eval "$SHUTDOWN_CMD"
    fi
    exit 1
}

# --------------------------------------------------------------------------- #
# 0. Kiểm tra môi trường
# --------------------------------------------------------------------------- #
log "=== Kiểm tra môi trường ==="
nvidia-smi || { log "!!! Không thấy GPU."; fail_and_wait "kiểm tra GPU"; }

python3 -c "import torch, transformers, torchcrf, pandas, sklearn; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())" \
    || { log "!!! Thiếu thư viện."; fail_and_wait "kiểm tra thư viện"; }

for f in baseline_lstm_crf.py punc_dataset_word.py; do
    [ -f "$WORKDIR/$f" ] || { log "!!! Không tìm thấy $f trong $WORKDIR"; fail_and_wait "kiểm tra file script"; }
done
for d in Novels News; do
    for split in train valid test; do
        [ -f "$DATA_ROOT/$d/$split.txt" ] || { log "!!! Thiếu $DATA_ROOT/$d/$split.txt"; fail_and_wait "kiểm tra dữ liệu"; }
    done
done
log "Môi trường OK. model_key=$MODEL_KEY | use_bilstm=$([ -n "$USE_BILSTM_FLAG" ] && echo true || echo false)"

cd "$WORKDIR"

# --------------------------------------------------------------------------- #
# 1. Train Novels
# --------------------------------------------------------------------------- #
log "=== [1/2] Bắt đầu train trên Novels ==="
python3 baseline_lstm_crf.py \
    --model_key "$MODEL_KEY" $USE_BILSTM_FLAG \
    --data_dir "$DATA_ROOT/Novels" \
    --output_dir "$OUT_ROOT/${RUN_TAG}_novels" \
    --device cuda --fp16 \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --logging_steps "$LOGGING_STEPS" \
    2>&1 | tee "$LOG_DIR/${RUN_TAG}_novels.log"

if [ ! -f "$OUT_ROOT/${RUN_TAG}_novels/best_checkpoint.pt" ]; then
    fail_and_wait "train Novels (không thấy best_checkpoint.pt)"
fi
log "=== [1/2] Xong Novels. Best F1: $(grep 'Best macro F1' "$OUT_ROOT/${RUN_TAG}_novels/eval_results.txt" || echo 'không đọc được') ==="

# --------------------------------------------------------------------------- #
# 2. Train News
# --------------------------------------------------------------------------- #
log "=== [2/2] Bắt đầu train trên News ==="
python3 baseline_lstm_crf.py \
    --model_key "$MODEL_KEY" $USE_BILSTM_FLAG \
    --data_dir "$DATA_ROOT/News" \
    --output_dir "$OUT_ROOT/${RUN_TAG}_news" \
    --device cuda --fp16 \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --logging_steps "$LOGGING_STEPS" \
    2>&1 | tee "$LOG_DIR/${RUN_TAG}_news.log"

if [ ! -f "$OUT_ROOT/${RUN_TAG}_news/best_checkpoint.pt" ]; then
    fail_and_wait "train News (không thấy best_checkpoint.pt)"
fi
log "=== [2/2] Xong News. Best F1: $(grep 'Best macro F1' "$OUT_ROOT/${RUN_TAG}_news/eval_results.txt" || echo 'không đọc được') ==="

# --------------------------------------------------------------------------- #
# 3. Xong cả 2 -> tắt máy (trừ khi AUTO_SHUTDOWN=0)
# --------------------------------------------------------------------------- #
log "=== Train xong cả Novels + News cho $RUN_TAG. Checkpoint tại $OUT_ROOT/${RUN_TAG}_novels và $OUT_ROOT/${RUN_TAG}_news ==="
if [ "$AUTO_SHUTDOWN" = "1" ]; then
    log "=== Chuẩn bị tắt máy sau 60 giây (Ctrl+C để hủy nếu cần giữ máy tải file trước) ==="
    sleep 60
    eval "$SHUTDOWN_CMD"
else
    log "=== AUTO_SHUTDOWN=0, máy sẽ KHÔNG tự tắt. Fa tự quản lý instance. ==="
fi
