# Project 1: Arabic banking intent classifier

Agreed on 2026-10-23; replaces the earlier draft (`job-search/plans/01-arabic-banking-intents.md`).
Each step gets a short design (goal, approach, risks) before any code is written.

## Goal

Fine-tune an Arabic encoder (full fine-tuning) to classify banking customer messages into the 77
ArBanking77 intents, and build a production-standard system around it: versioned data, tracked runs,
a model registry with a promotion gate, an API, CI and a model card.

## Data

- **ArBanking77, full corpus** from [SinaLab/ArBanking77](https://github.com/SinaLab/ArBanking77) (`data/`),
  CC BY-SA 4.0. Human-localized from the English BANKING77. The Hugging Face copy is only a sample; do not use it.
- Files (`label`, `text`): MSA + Palestinian train 21,559 rows (63–475 per intent, median 272);
  validation 2,464; tests: MSA 3,574, Palestinian 3,807, Saudi 3,580, Moroccan 3,574, Tunisian 999.
- **Known issues** (see `docs/log.md`):
  - The official validation split leaks: 2,200 source questions have one version in train and another in
    validation, so 94% of the validation versions have a version of the same question in train. We build
    our own validation split from train, **grouped by source question**
    (`Banking77_full_corpus.csv` links each Arabic version to its English source).
  - 49 rows of the full corpus have unescaped quotes inside a text: 8 break into the wrong number of
    columns, and 41 load correctly but keep a stray `"` in the text. The loader must handle both.
- Domain caveat: the messages are localized from a European fintech app's queries, not real Arab bank chats.

## Approach

1. **Baseline:** TF-IDF + logistic regression.
2. **Main model:** full fine-tuning of AraBERT (`aubmindlab/bert-base-arabertv02`).
3. **Quality check:** AraBERT should come close to the paper's F1 of 0.92 on MSA before any other number is trusted.

## Evaluation

- Accuracy and macro-F1 on each official test set (MSA, Palestinian, Saudi, Moroccan, Tunisian).
- The most-confused intent pairs.
- p50 / p95 response time on CPU.
- **Promotion gate**, with numbers agreed before training: macro-F1 above the baseline by an agreed margin,
  and p95 CPU latency under an agreed limit.

## The system

- DVC for data; MLflow for runs and the registry (aliases, not stages).
- The tokenizer and label map travel with the model as one versioned unit.
- FastAPI `/predict` returning the top 3 intents with confidence scores; Docker; pytest + GitHub Actions CI.
- Model card on the Hugging Face Hub; README results table; `eval/IMPROVEMENT_LOG.md` with each
  prediction, written before the run, next to its result; `docs/log.md`.

## Working conventions

- Build in steps: a file or folder is added only in the step that first uses it.
- GPU training runs on Kaggle or Colab (T4).
- ArBanking77 is CC BY-SA 4.0: credit SinaLab, and release anything derived from the data under the same licence.

## Steps (about 2 hours each)

| # | Step | Adds |
|---|---|---|
| 0 | Repo setup | this plan, `docs/log.md`, `README.md` |
| 1 | Data: load the full corpus, grouped validation split, data card; the evaluation harness | `pyproject.toml`, DVC, `src/`, `tests/` |
| 2 | Baseline; agree the promotion gate; MLflow tracking | MLflow, `eval/IMPROVEMENT_LOG.md` |
| 3 | AraBERT full fine-tuning (Kaggle); reproduction check; registry | training code, notebook for Kaggle |
| 4 | Serving: `/predict`, Docker, CI | API, `Dockerfile`, `.github/workflows/` |
| 5 | Model card on the Hub; README results; wrap-up | model card |

## Parked ideas (not in scope; revisit later)

- An Egyptian Arabic + Franco test set, written by hand.
- Franco (Arabizi) transliteration before the model.
- Comparing pretrained encoders (MARBERTv2, an Egyptian BERT).
- A confidence threshold that hands uncertain messages to a human.

## Related

- Project 2 (`job-search/plans/02-masry-support-ticket.md`) is planned separately, later.
- A similar public study exists: [AmrKhalil9/arabic-banking-intent-dialects](https://github.com/AmrKhalil9/arabic-banking-intent-dialects). Cite it in the README.
