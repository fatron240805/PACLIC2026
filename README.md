# PhoPunct — Vietnamese Punctuation Restoration Benchmark

Controlled benchmark for Vietnamese punctuation restoration, comparing transformer
backbones (mBERT, vELECTRA, ViBERT, XLM-R, PhoBERT-large) with and without a
Bi-LSTM layer on top, all under a shared CRF decoding layer, a shared training
budget, and a shared word-level data pipeline — evaluated on **News** and
**Novels** domains.

This repository contains the training code used for the model comparison
reported in `model_tracking.xlsx` (rows #9–#17): four baseline backbones
(`+CRF` / `+Bi-LSTM+CRF`) plus the proposed full system, **PhoPunct**
(PhoBERT-large + Bi-LSTM + CRF).

> **Note on data:** the `data/punctuation/News/` and `data/punctuation/Novels/`
> directories are intentionally **not** included in this repository (large raw
> text files). See [Data format](#data-format) below to reconstruct them from
> your own corpus.

---

## Repository structure

```
phopunct/
├── data/
│   └── punctuation/
│       ├── News/            # train.txt / valid.txt / test.txt (not tracked in git)
│       └── Novels/          # train.txt / valid.txt / test.txt (not tracked in git)
├── logs_from_gpu/           # raw training logs from full GPU runs (reference)
├── outputs_from_gpu/        # trained checkpoints downloaded from GPU instances
├── punc_dataset_word.py     # shared word-level data pipeline (used by all models)
├── baseline_lstm_crf.py     # models #9–#16: mBERT / vELECTRA / ViBERT / XLM-R × [+Bi-LSTM]+CRF
├── phobert_lstm_crf.py      # model #17: PhoPunct (PhoBERT-large + Bi-LSTM + CRF)
├── make_smoke_subset.py     # cuts a small CPU-sized subset for smoke testing
├── run_baseline_full.sh     # vast.ai full-training driver for baseline_lstm_crf.py
├── run_phopunct_full.sh     # vast.ai full-training driver for phobert_lstm_crf.py
└── wait_download_shutdown_mbert_news.sh   # local watchdog: poll → rsync → shutdown
```

---

## Task and label set

Punctuation restoration is framed as word-level sequence tagging over 7 classes:

```
O, PERIOD, COMMA, COLON, QMARK, EXCLAM, SEMICOLON
```

Input files use a simple two-column, one-token-per-line format (blank lines are
skipped, one file = one continuous stream of tokens):

```
tto O
trung O
ương O
...
sinh PERIOD
```

## Data format

Each domain directory must contain three files with this exact layout:

```
data/punctuation/<Domain>/
├── train.txt
├── valid.txt
└── test.txt
```

- `<Domain>` is `News` or `Novels`.
- Each line is `<token> <label>` separated by whitespace; the label must be one
  of the 7 classes above (unknown labels are coerced to `O` with a warning).
- `punc_dataset_word.py` groups consecutive lines into training chunks of up
  to 128 words, cutting only at sentence-ending labels (`PERIOD` / `QMARK` /
  `EXCLAM`) so no chunk splits mid-sentence.

Only these two directories are used for training. There is no `large/` or
`generated/` data source in this pipeline — every model in this benchmark is
trained on the exact same `News/` and `Novels/` files to keep the comparison
fair.

---

## Architecture

All 9 model configurations (8 baselines + PhoPunct) share the same shape:

```
tokens → subword tokenizer → transformer backbone
       → gather first-subword-per-word ("word_starts")   [BEFORE any BiLSTM]
       → (optional) Bi-LSTM(hidden=128, 1 layer, bidirectional)
       → dropout → Linear classifier → cast to float32
       → CRF(num_labels=7, batch_first=True), mask = word_mask
```

Gathering word-level representations *before* the Bi-LSTM (rather than after)
is deliberate: it keeps the sequence length the recurrent layer sees at the
word level, not the (longer, backbone-dependent) subword level, so the
Bi-LSTM/CRF stage is architecturally identical across all backbones.

| Script | Covers | Backbone(s) |
|---|---|---|
| `baseline_lstm_crf.py` | models #9–#16 | `mbert`, `velectra`, `bert` (ViBERT), `xlmr` |
| `phobert_lstm_crf.py` | model #17 (PhoPunct) | `vinai/phobert-large` |

`--model_key` → HuggingFace checkpoint mapping used by `baseline_lstm_crf.py`:

| `--model_key` | Checkpoint | Display name |
|---|---|---|
| `mbert` | `bert-base-multilingual-cased` | mBERT |
| `velectra` | `FPTAI/velectra-base-discriminator-cased` | vELECTRA |
| `bert` | `NlpHUST/vibert4news-base-cased` | **ViBERT** |
| `xlmr` | `xlm-roberta-base` | XLM-R |

> `NlpHUST/vibert4news-base-cased` ships without `model_type` /
> `tokenizer_class` metadata, so `AutoModel`/`AutoTokenizer` cannot auto-detect
> it. `baseline_lstm_crf.py` instantiates `BertModel`/`BertTokenizer` directly
> for `model_key in {"mbert", "bert"}` to work around this.

Add `--use_bilstm` to switch a run from the plain `+CRF` variant to the
`+Bi-LSTM+CRF` variant.

---

## Installation

```bash
pip install torch transformers pytorch-crf pandas scikit-learn
```

Tested with CUDA (fp16 mixed precision) on RTX 3090 / RTX 3090 Ti instances,
and on CPU for smoke testing.

## Quickstart

### 1. Smoke test on CPU (recommended before any full GPU run)

```bash
python make_smoke_subset.py \
    --src_dir  "data/punctuation/News" \
    --dst_dir  "./smoke_data/News" \
    --n_train_lines 4000 --n_valid_lines 800 --n_test_lines 800

# Baseline model, CPU smoke test
python baseline_lstm_crf.py \
    --model_key bert \
    --data_dir ./smoke_data/News \
    --output_dir ./smoke_out/bert_news \
    --device cpu --max_train_examples 50 --num_train_epochs 1

# PhoPunct, CPU smoke test (use phobert-base to keep it light on CPU)
python phobert_lstm_crf.py \
    --data_dir ./smoke_data/News \
    --model_name_or_path vinai/phobert-base \
    --output_dir ./smoke_out/phopunct_news \
    --device cpu --max_train_examples 50 --num_train_epochs 1
```

### 2. Full GPU training — single model

```bash
# Baseline: XLM-R + Bi-LSTM + CRF on News
python baseline_lstm_crf.py \
    --model_key xlmr --use_bilstm \
    --data_dir data/punctuation/News \
    --output_dir outputs/xlmr_bilstm_news \
    --device cuda --fp16 \
    --max_seq_length 256 --train_batch_size 16 --eval_batch_size 32 \
    --num_train_epochs 8 --learning_rate 2e-5 --logging_steps 50

# PhoPunct on Novels
python phobert_lstm_crf.py \
    --data_dir data/punctuation/Novels \
    --model_name_or_path vinai/phobert-large \
    --output_dir outputs/phopunct_novels \
    --device cuda --fp16 \
    --max_seq_length 256 --train_batch_size 16 --eval_batch_size 32 \
    --num_train_epochs 8 --learning_rate 2e-5 --logging_steps 50
```

### 3. Full GPU training — driver scripts (used on vast.ai)

Both drivers train **Novels then News** back-to-back, then shut the instance
down. On failure they sleep `SLEEP_ON_ERROR_MIN` minutes (default 30) before
shutting down, so you have time to SSH in and inspect the logs.

```bash
chmod +x run_baseline_full.sh run_phopunct_full.sh

# One baseline configuration (Novels -> News -> shutdown)
./run_baseline_full.sh --model_key xlmr --use_bilstm

# PhoPunct (Novels -> News -> shutdown)
./run_phopunct_full.sh
```

Key environment variable overrides (all optional, same for both drivers):

| Variable | Default | Meaning |
|---|---|---|
| `DATA_ROOT` | `<workdir>/data/punctuation` | root containing `News/` and `Novels/` |
| `OUT_ROOT` | `<workdir>/outputs` | where checkpoints are written |
| `MAX_SEQ_LENGTH` | `256` | hard ceiling — see note below |
| `TRAIN_BATCH_SIZE` / `EVAL_BATCH_SIZE` | `16` / `32` | |
| `NUM_TRAIN_EPOCHS` | `8` | |
| `LEARNING_RATE` | `2e-5` | applied uniformly to backbone + Bi-LSTM + CRF |
| `SLEEP_ON_ERROR_MIN` | `30` | grace period before auto-shutdown on failure |
| `AUTO_SHUTDOWN` (`run_baseline_full.sh` only) | `1` | set to `0` to keep the instance alive after success |

> **`--max_seq_length` ceiling:** PhoBERT-large's `max_position_embeddings`
> is 258 (2 slots reserved for `<s>`/`</s>`), so **256 is a hard ceiling**
> for `phobert_lstm_crf.py`, not just a default.

### 4. Local watchdog (optional)

`wait_download_shutdown_mbert_news.sh` runs from your **local machine** (not
over SSH), polls a remote vast.ai instance every 30s, and only downloads +
shuts the instance down once it confirms a `"Train xong"` line in the remote
log (to avoid a false positive from a stale checkpoint left over from a crash).
Edit `PORT`, `IP`, `REMOTE_LOG`, `REMOTE_OUT`, `LOCAL_OUT` at the top of the
script before use.

---

## Checkpoints and evaluation

Each run writes to `<output_dir>/`:

- `last_checkpoint.pt` — overwritten every epoch
- `best_checkpoint.pt` — overwritten whenever validation macro F1 improves
- `eval_results.txt` — best macro F1 + full `sklearn.classification_report`

Checkpoint dict keys: `model_state_dict`, `optimizer_state_dict`,
`scheduler_state_dict`, `epoch`, `best_f1` (`baseline_lstm_crf.py` also stores
`model_key`, `use_bilstm`, `display_name`).

Evaluation reports macro/micro/weighted F1 over the 6 punctuation classes
(`PERIOD`, `COMMA`, `COLON`, `QMARK`, `EXCLAM`, `SEMICOLON`); the `O` class is
excluded from the report.

## Benchmark results (macro F1, best checkpoint)

From the full GPU runs in `logs_from_gpu/`:

| Model | News | Novels |
|---|---|---|
| mBERT + CRF | 0.5376 | 0.4339 |
| mBERT + Bi-LSTM + CRF | 0.5354 | 0.4237 |
| vELECTRA + CRF | 0.5384 | 0.4064 |
| vELECTRA + Bi-LSTM + CRF | 0.5345 | 0.4022 |
| ViBERT + CRF | 0.5797 | 0.4978 |
| ViBERT + Bi-LSTM + CRF | 0.5766 | 0.4591 |
| XLM-R + CRF | 0.5862 | 0.5018 |
| XLM-R + Bi-LSTM + CRF | 0.5789 | 0.4690 |
| **PhoPunct** (PhoBERT-large + Bi-LSTM + CRF) | 0.6135 | 0.4897 |

A recurring finding across this benchmark: under a **shared** training
configuration (same learning rate, epochs, batch size), the plain `+CRF`
variant outperforms the `+Bi-LSTM+CRF` variant of the same backbone in most
backbone/domain combinations — adding a randomly-initialized Bi-LSTM on top of
an already-finetuned bidirectional transformer does not reliably help under
this budget. `COLON`/`SEMICOLON` scores on Novels are near zero for several
models, which tracks the very low absolute number of training instances for
those classes in the Novels split, not a modeling bug.

---

## Known issues and fixes already applied in this code

- **fp16 + CRF NaNs:** `CRF`'s internal `logsumexp` can produce NaN gradients
  under fp16 autocast, silently causing `GradScaler` to skip optimizer steps.
  Fixed by casting logits to `float32` immediately before the CRF layer in
  both `baseline_lstm_crf.py` and `phobert_lstm_crf.py`.
- **`model.train()` reset per epoch:** explicitly called at the top of every
  epoch loop so a prior call to `evaluate()` never leaves the model stuck in
  `eval()` mode for the next epoch.
- **Truncation/label alignment (`punc_dataset_word.py`):** truncation happens
  at word boundaries by checking the subword budget *before* appending a
  word's pieces, avoiding any subword/word-index mismatch.
- **Padding (`punc_dataset_word.py`):** subword-space padding
  (`input_ids`/`attention_mask`/`word_starts`) and word-space padding
  (`label_ids`/`word_mask`) are two separate loops, so `word_mask` is never
  over-filled with 1s (which previously caused `log_likelihood = -inf` from
  the CRF on step one).
- **`vibert4news-base-cased` loading:** direct `BertModel`/`BertTokenizer`
  instantiation instead of `AutoModel`/`AutoTokenizer` (see architecture table
  above).

---

## Citation

If this benchmark or code is useful to your work, please check back for the
associated paper's citation details once published. This README does not
include a BibTeX entry yet.
