#!/usr/bin/env bash
# run_plain_1_to_8.sh
#
# Tự động train HẾT model #1-8 trong model_tracking.xlsx (4 backbone x
# [Plain]/[Bi-LSTM], KHÔNG có CRF) trên 1 instance vast.ai mới.
# Thứ tự: #1 Novels -> #1 News -> #2 Novels -> #2 News -> ... -> #8 News -> tắt máy.
# Nếu 1 lượt lỗi, dừng lại NGAY (không chạy tiếp các model sau), chờ Fa vào xem.
#
# Cách dùng:
#   chmod +x run_plain_1_to_8.sh
#   ./run_plain_1_to_8.sh
#
# Đặt cùng cấp với baseline_plain.py, punc_dataset_word.py, data/ (giống cấu
# trúc run_baseline_full.sh/run_phopunct_full.sh đã dùng ở instance trước).

set -uo pipefail

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
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"   # set AUTO_SHUTDOWN=0 nếu muốn giữ máy sống sau khi xong hết

LOG_DIR="$WORKDIR/logs"
mkdir -p "$LOG_DIR" "$OUT_ROOT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [run_plain_1_to_8] $*"; }

fail_and_wait() {
    local step_name="$1"
    log "!!! LỖI ở bước: $step_name — xem log tại $LOG_DIR"
    log "!!! Dừng lại, KHÔNG train tiếp các model còn lại. Chờ ${SLEEP_ON_ERROR_MIN} phút để Fa kịp SSH vào kiểm tra."
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
# 0. Kiểm tra môi trường trước khi chạy bất cứ gì
# --------------------------------------------------------------------------- #
log "=== Kiểm tra môi trường ==="
nvidia-smi || { log "!!! Không thấy GPU."; fail_and_wait "kiểm tra GPU"; }

python3 -c "import torch, transformers, torchcrf, pandas, sklearn; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())" \
    || { log "!!! Thiếu thư viện. Chạy: pip install torch transformers pytorch-crf pandas scikit-learn sentencepiece"; fail_and_wait "kiểm tra thư viện"; }

for f in baseline_plain.py punc_dataset_word.py; do
    [ -f "$WORKDIR/$f" ] || { log "!!! Không tìm thấy $f trong $WORKDIR"; fail_and_wait "kiểm tra file script"; }
done
for d in Novels News; do
    for split in train valid test; do
        [ -f "$DATA_ROOT/$d/$split.txt" ] || { log "!!! Thiếu $DATA_ROOT/$d/$split.txt"; fail_and_wait "kiểm tra dữ liệu"; }
    done
done
log "Môi trường OK. Bắt đầu train tuần tự model #1 -> #8 (16 lượt Novels+News)."

cd "$WORKDIR"

# --------------------------------------------------------------------------- #
# Danh sách 8 model theo đúng thứ tự STT 1-8 trong model_tracking.xlsx
# Mỗi phần tử: "model_key|use_bilstm_flag|run_tag|stt"
# --------------------------------------------------------------------------- #
MODELS=(
    "mbert||mbert_plain|1"
    "velectra||velectra_plain|2"
    "bert||bert_plain|3"
    "xlmr||xlmr_plain|4"
    "mbert|--use_bilstm|mbert_bilstm_plain|5"
    "velectra|--use_bilstm|velectra_bilstm_plain|6"
    "bert|--use_bilstm|bert_bilstm_plain|7"
    "xlmr|--use_bilstm|xlmr_bilstm_plain|8"
)

for entry in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_KEY USE_BILSTM_FLAG RUN_TAG STT <<< "$entry"

    log "########## Model #$STT ($RUN_TAG) ##########"

    for DOMAIN in Novels News; do
        DOMAIN_LOWER=$(echo "$DOMAIN" | tr '[:upper:]' '[:lower:]')
        OUT_DIR="$OUT_ROOT/${RUN_TAG}_${DOMAIN_LOWER}"
        LOG_FILE="$LOG_DIR/${RUN_TAG}_${DOMAIN_LOWER}.log"

        log "=== Model #$STT ($RUN_TAG) - Bắt đầu train trên $DOMAIN ==="
        python3 baseline_plain.py \
            --model_key "$MODEL_KEY" $USE_BILSTM_FLAG \
            --data_dir "$DATA_ROOT/$DOMAIN" \
            --output_dir "$OUT_DIR" \
            --device cuda --fp16 \
            --max_seq_length "$MAX_SEQ_LENGTH" \
            --train_batch_size "$TRAIN_BATCH_SIZE" \
            --eval_batch_size "$EVAL_BATCH_SIZE" \
            --num_train_epochs "$NUM_TRAIN_EPOCHS" \
            --learning_rate "$LEARNING_RATE" \
            --logging_steps "$LOGGING_STEPS" \
            2>&1 | tee "$LOG_FILE"

        if [ ! -f "$OUT_DIR/best_checkpoint.pt" ]; then
            fail_and_wait "Model #$STT ($RUN_TAG) - train $DOMAIN (không thấy best_checkpoint.pt)"
        fi
        log "=== Model #$STT ($RUN_TAG) - Xong $DOMAIN. Best F1: $(grep 'Best macro F1' "$OUT_DIR/eval_results.txt" || echo 'không đọc được') ==="
    done

    log "########## Model #$STT ($RUN_TAG) HOÀN TẤT cả Novels + News ##########"
done

# --------------------------------------------------------------------------- #
# Xong hết 8 model (16 lượt) -> tắt máy (trừ khi AUTO_SHUTDOWN=0)
# --------------------------------------------------------------------------- #
log "=== TRAIN XONG TOÀN BỘ MODEL #1-8 (16 lượt Novels+News). Checkpoint tại $OUT_ROOT ==="
if [ "$AUTO_SHUTDOWN" = "1" ]; then
    log "=== Chuẩn bị tắt máy sau 60 giây (Ctrl+C để hủy nếu cần giữ máy tải file trước) ==="
    sleep 60
    eval "$SHUTDOWN_CMD"
else
    log "=== AUTO_SHUTDOWN=0, máy sẽ KHÔNG tự tắt. Fa tự quản lý instance. ==="
fi
