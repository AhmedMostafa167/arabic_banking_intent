# Data card: ArBanking77, as used in this project

**Source.** [ArBanking77](https://github.com/SinaLab/ArBanking77), SinaLab, Birzeit University: the full
corpus from the authors' GitHub repository. The Hugging Face copy (`SinaLab/ArBanking77`) is only a
sample of about 1,000 training rows, so it is not used. Licence: CC BY-SA 4.0. Anything derived from
the data, including the fine-tuned model, carries the same licence.

> Jarrar, Birim, Khalilia, Erden and Ghanem. *ArBanking77: Intent Detection Neural Model and a New
> Dataset in Modern and Dialectical Arabic.* ArabicNLP 2023, pages 276–287.
> [aclanthology.org/2023.arabicnlp-1.22](https://aclanthology.org/2023.arabicnlp-1.22)

**What it is.** Arabic versions of the English [BANKING77](https://arxiv.org/abs/2003.04807) customer
queries: one query per row, labelled with one of 77 intents ("card arrival", "exchange rate",
"transfer fee charged", ...). People localised each English query into Modern Standard Arabic and
the Palestinian dialect. The test sets add Saudi, Moroccan and Tunisian versions.

## The splits used here

`data/processed/`, tracked with DVC (version `c38e4d6b`), built by `scripts/prepare_data.py`:

| Split | Rows | Source questions | Dialects | Rows per intent |
|---|---|---|---|---|
| Train | 21,571 | 8,993 | MSA 10,729, Palestinian 10,842 | 63–474 (median 272) |
| Validation (ours) | 2,397 | 999 | MSA 1,191, Palestinian 1,206 | 8–52 (median 31) |
| Test, MSA | 3,574 | | MSA | all 77 intents |
| Test, Palestinian | 3,807 | | Palestinian | all 77 intents |
| Test, Saudi | 3,580 | | Saudi | all 77 intents |
| Test, Moroccan | 3,574 | | Moroccan | all 77 intents |
| Test, Tunisian | 999 | | Tunisian | **27 of the 77 intents** |

- **Our validation split replaces the official one, which leaks.** 2,198 source questions have one
  Arabic version in the official train file and another in its validation file. So 94% of the
  official validation rows have a paraphrase of the same question in training, and that inflates
  validation macro-F1 by about 3 points (3.2 for TF-IDF, 3.6 for AraBERT; `eval/IMPROVEMENT_LOG.md`,
  entry 1).
- **How it is built:** the corpus's train and validation versions are pooled, then split by source
  question (`StratifiedGroupKFold`, 10% of questions, seed 42 in `configs/data.toml`). Every version of
  a question lands on one side, and no source question appears in both train and validation (checked).
- **The test files are used as published.**
- **Macro-F1 averages over the intents present in each set.** For Tunisian that is 27 intents;
  counting all 77 would add 50 zero scores.

## Cleaning

- 8 rows of `Banking77_full_corpus.csv` break into the wrong number of columns because of unescaped
  quotation marks; they are dropped.
- 41 more rows load but keep a stray `"` in the text; they are dropped from the training pool.
- Each load checks that rows loaded plus rows dropped equals the records in the file.

## Limitations

- **The domain is not real Arab banking chat.** The queries are localised from a European fintech
  app's customer messages, so intents such as "top up by bank transfer" reflect that app.
- **Training covers two varieties only:** MSA and Palestinian. Saudi, Moroccan and Tunisian are
  test-only, and the model scores much lower on them (Tunisian macro-F1 about 0.6).
- **Not covered at all:** Egyptian, Gulf dialects beyond Saudi, other Levantine dialects, and
  Arabizi (Arabic written in Latin letters).
- The Tunisian test set is small (999 rows, 27 intents), so its scores vary more.

## Rebuilding it

```bash
bash scripts/download_data.sh             # the raw files from the authors' repository
uv run python scripts/prepare_data.py     # data/processed/
dvc status                                # unchanged means the same data as every logged run
```
