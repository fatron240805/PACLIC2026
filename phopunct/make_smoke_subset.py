# -*- coding: utf-8 -*-
"""
make_smoke_subset.py

Trích 1 subset nhỏ từ News/ (hoặc Novels/) để chạy smoke test CPU nhanh
trước khi đẩy full data lên GPU. Cắt CHỈ tại các dòng có nhãn EOS
(PERIOD/QMARK/EXCLAM) để không cắt giữa câu.

Cách dùng (PowerShell, dùng forward-slash cho path để an toàn):
  python make_smoke_subset.py --src_dir "D:/COLING2027/vipunct_prototype/vipunct_proto/data/punctuation/News" --dst_dir "./smoke_data/News" --n_train_lines 4000 --n_valid_lines 800
"""
import argparse
import os

EOS_LABELS = {"PERIOD", "QMARK", "EXCLAM"}


def cut_at_boundary(lines, target_n_lines):
    """Giữ tối đa target_n_lines dòng đầu, nhưng lùi lại tới dòng EOS gần nhất
    để không cắt giữa câu."""
    if target_n_lines >= len(lines):
        return lines
    cut = target_n_lines
    while cut > 0:
        parts = lines[cut - 1].strip().split()
        if len(parts) == 2 and parts[1] in EOS_LABELS:
            break
        cut -= 1
    if cut == 0:
        cut = target_n_lines  # không tìm thấy ranh giới -> giữ nguyên cắt cứng (hiếm)
    return lines[:cut]


def process_file(src_path, dst_path, n_lines):
    with open(src_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    subset = cut_at_boundary(lines, n_lines)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        f.writelines(subset)
    print(f"{src_path} -> {dst_path}: {len(subset)}/{len(lines)} dòng")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--dst_dir", required=True)
    ap.add_argument("--n_train_lines", type=int, default=4000)
    ap.add_argument("--n_valid_lines", type=int, default=800)
    ap.add_argument("--n_test_lines", type=int, default=800)
    args = ap.parse_args()

    process_file(os.path.join(args.src_dir, "train.txt"),
                 os.path.join(args.dst_dir, "train.txt"), args.n_train_lines)
    process_file(os.path.join(args.src_dir, "valid.txt"),
                 os.path.join(args.dst_dir, "valid.txt"), args.n_valid_lines)
    process_file(os.path.join(args.src_dir, "test.txt"),
                 os.path.join(args.dst_dir, "test.txt"), args.n_test_lines)


if __name__ == "__main__":
    main()
