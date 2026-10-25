# Improvement log

Each entry records one measured change to the intent classifier: what was measured, the
prediction written before the run, the numbers, and the decision taken because of them.
Entries are in the order the work happened.

## Summary

| # | Date | Change | Measure | Before | After |
|---|---|---|---|---|---|
| 1 | 2026-10-25 | Our validation split instead of the official one (TF-IDF baseline) | Validation macro-F1 | official 0.919 | ours 0.887 (−3.2 points, all leak) |

## How the numbers are produced

- **Runs:** MLflow experiments `leak-check` and `intent-classifier`; each run records its git commit,
  data version and settings. Reproduce with `scripts/train_baseline.py`.
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
   TF-IDF's.

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
points. Part 2 (AraBERT gains more) is tested in step 3.

**Decision.** Every model decision uses our validation split. A gain smaller than about 3 points
measured on the official validation split could be the leak alone.

**Note.** The `val_predictions.parquet` artifact of these runs lacks the predictions (a bug, since
fixed). The interval of A − B was computed from predictions re-made with the logged models, which
reproduce the logged scores exactly.
