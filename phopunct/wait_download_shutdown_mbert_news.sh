#!/usr/bin/env bash
# wait_download_shutdown_mbert_news.sh
#
# Chạy TỪ MÁY LOCAL (WSL) - không phải trong SSH. Tự động:
#   1. Poll SSH mỗi 30s xem tiến trình mBERT+CRF News còn chạy không
#   2. Khi kết thúc, kiểm tra log có dòng "Train xong" chưa (tránh false positive
#      do best_checkpoint.pt còn sót từ epoch trước nếu bị crash giữa chừng)
#   3. Nếu thật sự xong -> rsync tải checkpoint về -> shutdown instance
#   4. Nếu không thấy "Train xong" -> DỪNG LẠI, không tải, không tắt máy
#
# Cách dùng:
#   chmod +x wait_download_shutdown_mbert_news.sh
#   ./wait_download_shutdown_mbert_news.sh
# (có thể chạy nền: nohup ./wait_download_shutdown_mbert_news.sh > wait.log 2>&1 &)

set -uo pipefail

PORT=61836
IP=171.250.6.198
REMOTE_LOG="/root/phopunct/logs/mbert_news.log"
REMOTE_OUT="/root/phopunct/outputs/mbert_news/"
LOCAL_OUT="/mnt/d/COLING2027/2026_08_09/phopunct/outputs_from_gpu/mbert_news/"

SSH_OPTS="-p $PORT -o ServerAliveInterval=15 -o ServerAliveCountMax=3"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Bắt đầu chờ mBERT+CRF News train xong (kiểm tra mỗi 30s)..."

while true; do
    STILL_RUNNING=$(ssh $SSH_OPTS "root@$IP" "pgrep -f baseline_lstm_crf.py" 2>/dev/null)
    if [ -z "$STILL_RUNNING" ]; then
        break
    fi
    log "Vẫn đang chạy, chờ 30 giây..."
    sleep 30
done

log "Tiến trình đã kết thúc (không còn process). Đang kiểm tra log..."

MATCH_COUNT=$(ssh $SSH_OPTS "root@$IP" "grep -c 'Train xong' $REMOTE_LOG" 2>/dev/null || echo "0")

if [ "$MATCH_COUNT" -ge "1" ] 2>/dev/null; then
    log "XONG THẬT SỰ (có dòng 'Train xong' trong log). Bắt đầu tải checkpoint..."

    mkdir -p "$LOCAL_OUT"
    until rsync -avP -e "ssh $SSH_OPTS" "root@$IP:$REMOTE_OUT" "$LOCAL_OUT"; do
        log "Rớt kết nối lúc tải, thử lại sau 5 giây..."
        sleep 5
    done
    log "TẢI mbert_news XONG."

    log "Đang tắt instance..."
    ssh $SSH_OPTS "root@$IP" "sudo shutdown -h now"
    log "Đã gửi lệnh tắt máy."
else
    log "!!! LỖI: Không thấy dòng 'Train xong' trong log (MATCH_COUNT=$MATCH_COUNT)."
    log "!!! KHÔNG tải, KHÔNG tắt máy. Fa cần tự SSH vào kiểm tra $REMOTE_LOG."
fi
