"""Build the train, validation and test sets from the raw ArBanking77 files.

Train and validation come from the full corpus, split by source question, so no question has
versions on both sides (the official validation split leaks). The official test files are kept as
published. The logic is in arabic_banking_intents.training.data; the split settings in configs/data.toml.

    uv run python scripts/prepare_data.py
    dvc add data/processed
"""

import tomllib

from arabic_banking_intents.helpers.config import get_settings
from arabic_banking_intents.training.data import (
    COLUMNS,
    build_pool,
    load_corpus,
    load_labels,
    load_official,
    split_by_question,
    to_versions,
)
from arabic_banking_intents.models.enums.DataEnum import TestSetEnum


def main() -> None:
    settings = get_settings()
    raw, out = settings.DATA_RAW_DIR, settings.DATA_PROCESSED_DIR
    params = tomllib.loads((settings.CONFIGS_DIR / "data.toml").read_text())

    labels = load_labels(raw / "Banking77_intents.csv")
    versions = to_versions(load_corpus(raw / "Banking77_full_corpus.csv"), labels)
    pool = build_pool(versions)
    train, val = split_by_question(pool, params["val_fraction"], params["seed"])

    out.mkdir(parents=True, exist_ok=True)
    train[COLUMNS].to_parquet(out / "train.parquet", index=False)
    val[COLUMNS].to_parquet(out / "val.parquet", index=False)
    for test_set in TestSetEnum:
        test = load_official(raw / f"Banking77_Arabized_{test_set}_test.csv", labels, dialect=test_set.value)
        test.to_parquet(out / f"test_{test_set.lower()}.parquet", index=False)
        print(f"test_{test_set.lower()}: {len(test):,} texts, {test['label'].nunique()} intents")
    labels.to_json(out / "labels.json", orient="records", force_ascii=False, indent=2)

    n_pool = versions["split"].isin(["train", "val"]).sum()
    print(f"dropped {n_pool - len(pool)} versions with a stray quote")
    print(f"train: {len(train):,} versions from {train['qid'].nunique():,} questions")
    print(f"val:   {len(val):,} versions from {val['qid'].nunique():,} questions")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
