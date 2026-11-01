# Improvement log

Each entry records one measured change to the intent classifier: what was measured, the
prediction written before the run, the numbers, and the decision taken because of them.
Entries are in the order the work happened.

## Summary

| # | Date | Change | Measure | Before | After |
|---|---|---|---|---|---|
| 1 | 2026-10-25 | Our validation split instead of the official one (TF-IDF baseline) | Validation macro-F1 | official 0.919 | ours 0.887 (−3.2 points, all leak) |
| 2 | 2026-11-01 | Fine-tuned AraBERT instead of TF-IDF (champion v3 → v4) | Validation macro-F1 | 0.887 | 0.918, mean of 3 seeds (+3.2 points, above the noise for every seed) |

## How the numbers are produced

- **Runs:** MLflow experiments `leak-check` and `intent-classifier`; each run records its git commit,
  data version and settings. Reproduce with `scripts/train_baseline.py` (TF-IDF, on the laptop) and
  `scripts/kaggle_push.py` + `scripts/log_kaggle_runs.py` (AraBERT, trained on a Kaggle T4 GPU; every
  metric is computed on the laptop from the saved predictions).
- **Macro-F1** averages F1 over the intents present in each set (the Tunisian test set has 27 of 77).
- **Uncertainty:** 95% bootstrap intervals, 1,000 resamples. On our validation split whole questions
  are resampled, because versions of one question are not independent; the official files have no
  question IDs, so single texts are resampled.

## Entry 1 — Baseline, and what the official validation split's leak costs (2026-10-25)

**Question.** 94% of the official validation versions have a version of the same question in
train (`notebooks/01_explore_data.ipynb`, part 2). How much does that inflate a score?

**Measure.** The TF-IDF + logistic regression baseline, macro-F1:

- **A:** trained on the official train file, scored on the official validation file;
- **B:** trained on our train split, scored on our validation split (grouped by question);
- both also scored on the official MSA and Palestinian test sets, which neither has seen.

**Prediction (2026-10-24, before any run).** The leak will not change the TF-IDF baseline's
score noticeably: A − B will be a random difference, not a real one. Tested as: the gap is no
larger than the noise floor measured in step 2.

**Revised prediction (2026-10-25, still before any run).** The baseline uses character n-grams
(`configs/baseline.toml`), not whole words, so it can partly match a leaked paraphrase: بطاقتي and
البطاقة share no word but share "بطاق". So TF-IDF will gain *something* from the leak, but less
than AraBERT, because it is still a keyword model and cannot match a paraphrase that shares no
characters. Tested as:

1. **TF-IDF (step 2):** A − B is above the noise (the 95% interval of the difference excludes 0).
2. **AraBERT (step 3):** AraBERT's gap between the official and our validation sets is larger than
   TF-IDF's. Number added 2026-10-31, before any AraBERT run: **about 4 points**, not more, because
   TF-IDF with character n-grams already behaves similarly.

**Result.** Runs `tfidf-logreg-official` and `tfidf-logreg-grouped` in `leak-check`, commit
`c6e2687`, data version `c38e4d6b`.

| | Validation macro-F1 (95% interval) | MSA test | Palestinian test |
|---|---|---|---|
| A: official split | 0.919 (0.906–0.929) | 0.884 | 0.855 |
| B: our split | 0.887 (0.863–0.900) | 0.881 | 0.852 |

- **A − B = +3.2 points (95% interval +1.3 to +5.8).** The interval excludes 0: the leak inflates
  the baseline's validation score by about 3 points.
- **On the test sets the two models are within 0.3 points.** They learned from almost the same
  texts and are equally good; only the measurement differs.
- **A's validation score overstates its MSA test score by 3.4 points; B's by 0.5.** Our validation
  split predicts test performance; the official one does not.
- TF-IDF training is deterministic: the same run in `intent-classifier` reproduced B exactly, so
  the baseline has no training noise. Its noise comes only from which questions are in validation
  (B's interval is about ±2 points).

**Prediction vs result.** The original prediction (no noticeable change) was wrong. The revised
one, part 1, was right: character n-grams let TF-IDF exploit the leaked paraphrases, by about 3
points.

**Result, part 2 (2026-11-01).** Runs `arabert-official-seed42` in `leak-check` and
`arabert-grouped-seed42` in `intent-classifier`, commit `f53dd00`, data version `c38e4d6b`.

| | Validation macro-F1 | MSA test | Palestinian test |
|---|---|---|---|
| A: official split | 0.955 | 0.921 | 0.901 |
| B: our split | 0.919 | 0.914 | 0.889 |

- **A − B = +3.6 points (95% interval +2.0 to +5.6)**, against TF-IDF's +3.2 (+1.3 to +5.8).
- A's validation score overstates its MSA test score by 3.4 points, as it did for TF-IDF; B's by
  0.6. The test differences between A and B are within AraBERT's seed spread (entry 2).
- A's MSA test score, 0.921, is the paper's number: the fine-tuning setup reproduces the published
  result.

**Prediction vs result, part 2.** "About 4 points, not more": right (3.6). "Larger than TF-IDF's":
not shown. The gap is 0.4 points larger, but the two intervals overlap almost entirely, so the data
cannot tell them apart. With character n-grams, TF-IDF exploits the leak about as much as AraBERT.

**Decision.** Every model decision uses our validation split. A gain smaller than about 3 points
measured on the official validation split could be the leak alone.

**Note.** The `val_predictions.parquet` artifact of these runs lacks the predictions (a bug, since
fixed). The interval of A − B was computed from predictions re-made with the logged models, which
reproduce the logged scores exactly.

## Entry 2 — Fine-tuned AraBERT against the TF-IDF champion (2026-11-01)

**Question.** Does fine-tuned AraBERT (`aubmindlab/bert-base-arabertv02`, `configs/arabert.toml`)
beat the baseline by more than the noise, and how much do its scores vary between seeds?

**Measure.** Three runs on our split (seeds 42, 43, 44) in `intent-classifier`, trained on a Kaggle
GPU; the gate in the plan (paired 95% bootstrap interval of AraBERT − champion validation macro-F1
above 0 for every seed). Also: the paper's MSA test macro-F1 of 0.921 as a reproduction check, and
latency per message on a laptop CPU.

**Predictions (2026-10-31, before any run).**

1. AraBERT on our split will score higher than the baseline on the test sets, but maybe lower on
   validation; and its test score will be closer to its validation score, because the bias was
   removed from the validation data.
2. The spread between seeds will be about **0.5 points**: random initialisation matters little,
   because a large pretrained model converges to almost the same weights.
3. Latency: about **60 ms per message** on average on the laptop CPU (a rough guess).

**Result.** Runs `arabert-grouped-seed42`, `-seed43` and `-seed44` in `intent-classifier`, commit
`f53dd00`, data version `c38e4d6b`; each trained in 5.5–8 minutes on a Kaggle T4 (best epochs 6, 8
and 5). The champion is the re-logged baseline, v3 (the same model as v1, returning probabilities).

| Macro-F1 | Validation | MSA | Palestinian | Saudi | Moroccan | Tunisian |
|---|---|---|---|---|---|---|
| TF-IDF champion (v3) | 0.887 | 0.881 | 0.852 | 0.719 | 0.612 | 0.499 |
| AraBERT, mean of 3 seeds | 0.918 | 0.917 | 0.895 | 0.828 | 0.769 | 0.592 |
| Gain (points) | +3.2 | +3.5 | +4.3 | +10.9 | +15.7 | +9.3 |
| Seed spread, max − min (points) | 0.3 | 0.5 | 1.0 | 2.6 | 1.7 | 4.0 |

- **Gate: passed by every seed.** The paired 95% intervals of the gain over v3 are +1.75 to +5.14
  (seed 42), +1.73 to +4.98 (43) and +1.40 to +4.76 (44). The median seed, 43 (validation
  0.919), is registered as **v4 @champion**.
- **The largest gains are on the dialects absent from training** (Saudi, Moroccan, Tunisian: +9 to
  +16 points). AraBERT's pretraining text taught it words that TF-IDF can only match by spelling.
- **Our validation split predicts the test score:** mean validation 0.918 against mean MSA test
  0.917.
- **The laptop copy of each model reproduced Kaggle's validation predictions exactly** (100% label
  agreement), so the fp16 save and the fp32 CPU load serve the model that was trained.
- **Latency on the laptop CPU** (a 2014 AMD A10 with 4 cores, no AVX2, and 4 GB of memory under WSL):
  p50 435–495 ms and p95 728–955 ms per message, one at a time; about 200 ms per message in
  batches. TF-IDF: p50 12 ms. The model takes 272 MB on disk.
- Validation loss is lowest at epochs 4–5 and then rises, while macro-F1 still improves slightly:
  the model grows more confident, including on its mistakes. Early stopping uses macro-F1, the
  measure that is reported.

**Prediction vs result.**

1. Higher than the baseline on the test sets: **right**, on all five. Lower on validation:
   **wrong**: 3.2 points higher, beyond the noise. Test closer to validation: **right**. On our
   split they are 0.1 points apart; the official split overstates AraBERT by 3.4 points
   (entry 1).
2. Seed spread of about 0.5 points: **right on validation and MSA** (0.3 and 0.5), **wrong on the
   unseen dialects** (1.7 to 4.0). The seed matters most where the model extrapolates, and where
   the test set is small (Tunisian: 999 texts, 27 intents).
3. About 60 ms per message: **wrong, by about 8×** (p50 about 450 ms). This laptop's CPU is the
   cause, not the model's design: BERT-base needs about 5 billion floating-point operations for a
   30-token message, and this CPU manages about 10 billion a second.

**Decision.** AraBERT (v4) is the champion; v3 stays in the registry. Latency was reported, not
gated, as agreed, and half a second per message on this laptop is the first problem for serving
(step 4). Comparing two AraBERT models on one dialect needs several seeds: a single seed can move
the Tunisian score by 4 points.
