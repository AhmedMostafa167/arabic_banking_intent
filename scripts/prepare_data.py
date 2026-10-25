"""Build the train, validation and test sets from the raw ArBanking77 files.

Train and validation come from the full corpus, split by source question, so no question has
versions on both sides (the official validation split leaks). The official test files are kept as
published. The exploration behind each step is in notebooks/01_explore_data.ipynb.

    uv run python scripts/prepare_data.py
    dvc add data/processed
"""

import csv
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

SEED = 42
VAL_FRACTION = 0.10
EXPECTED_MALFORMED_ROWS = 8  # unescaped quotes split these rows into 14-15 fields
SLOTS = ["MSA1", "MSA2", "PAL1", "PAL2"]
ID_PATTERN = r"(Tr|D|Te)\d+"
SPLIT_NAMES = {"Tr": "train", "D": "val", "Te": "test"}
TEST_SETS = ["MSA", "PAL", "Saudi", "Moroccan", "Tunisian"]
COLUMNS = ["qid", "version_id", "dialect", "text", "label", "intent"]


def load_labels(path: Path) -> pd.DataFrame:
    """The 77 intents. `label` (0-76) follows the file's order, which is the corpus Intent_ID - 1."""
    labels = pd.read_csv(path).rename(columns={"label_en": "intent", "label_ar": "intent_ar"})
    labels.insert(0, "label", range(len(labels)))
    assert len(labels) == 77 and labels["intent"].is_unique and labels["intent_ar"].is_unique
    return labels


def to_label(names: pd.Series, lookup: dict[str, int]) -> pd.Series:
    labels = names.map(lookup)
    unknown = set(names[labels.isna()])
    assert not unknown, f"unknown intent names: {unknown}"
    return labels.astype(int)


def load_corpus(path: Path) -> pd.DataFrame:
    """One row per source question, without the rows that don't split into the header's columns."""
    with open(path, encoding="utf-8", newline="") as f:
        records = list(csv.reader(f))
    bad_idx = [i for i, row in enumerate(records) if len(row) != len(records[0])]
    corpus = pd.read_csv(path, skiprows=bad_idx)

    assert len(bad_idx) == EXPECTED_MALFORMED_ROWS, f"expected {EXPECTED_MALFORMED_ROWS} malformed rows, found {len(bad_idx)}"
    assert len(corpus) + len(bad_idx) == len(records) - 1, "csv and pandas disagree on the rows"
    return corpus


def to_versions(corpus: pd.DataFrame) -> pd.DataFrame:
    """One row per Arabic version, with its official split taken from the ID prefix."""
    parts = []
    for slot in SLOTS:
        part = corpus[["QID", "Intent_en", f"QuestionID_{slot}", f"Question_{slot}"]].rename(columns={
            "QID": "qid", "Intent_en": "intent",
            f"QuestionID_{slot}": "version_id", f"Question_{slot}": "text",
        })
        part["slot"] = slot
        part["dialect"] = slot[:3]
        parts.append(part)

    versions = (
        pd.concat(parts, ignore_index=True)
        .dropna(subset=["text"])  # an empty slot means no version was written
        .sort_values(["qid", "slot"])
        .reset_index(drop=True)
    )

    # 5 PAL1 IDs read "Train"; their texts are in the official train file
    prefix = (
        versions["version_id"].str.extract(f"^{ID_PATTERN}$")[0]
        .mask(versions["version_id"] == "Train", "Tr")
    )
    versions["split"] = prefix.map(SPLIT_NAMES)
    assert versions["split"].notna().all(), "a version has no split"
    return versions


def split_by_question(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified by intent and grouped by question: all versions of a question land on one side."""
    sgkf = StratifiedGroupKFold(n_splits=round(1 / VAL_FRACTION), shuffle=True, random_state=SEED)
    train_idx, val_idx = next(sgkf.split(pool, y=pool["label"], groups=pool["qid"]))
    train, val = pool.iloc[train_idx], pool.iloc[val_idx]

    assert not set(train["qid"]) & set(val["qid"]), "a question is in both train and val"
    assert train["label"].nunique() == val["label"].nunique() == 77, "an intent is missing"
    assert len(train) + len(val) == len(pool), "rows lost in the split"
    return train, val


def load_official_test(path: Path, dialect: str, lookup: dict[str, int], names: pd.Series) -> pd.DataFrame:
    test = pd.read_csv(path)
    test["label"] = to_label(test["label"], lookup)
    test["intent"] = test["label"].map(names)
    test["dialect"] = dialect
    return test[["dialect", "text", "label", "intent"]]


def main() -> None:
    labels = load_labels(RAW / "Banking77_intents.csv")
    by_en = dict(zip(labels["intent"], labels["label"]))
    by_ar = dict(zip(labels["intent_ar"], labels["label"]))

    versions = to_versions(load_corpus(RAW / "Banking77_full_corpus.csv"))
    versions["label"] = to_label(versions["intent"], by_en)

    in_pool = versions["split"].isin(["train", "val"])
    has_quote = versions["text"].str.contains('"', regex=False)  # left over from the broken CSV quoting
    pool = versions[in_pool & ~has_quote].reset_index(drop=True)
    train, val = split_by_question(pool)

    OUT.mkdir(parents=True, exist_ok=True)
    train[COLUMNS].to_parquet(OUT / "train.parquet", index=False)
    val[COLUMNS].to_parquet(OUT / "val.parquet", index=False)
    for name in TEST_SETS:
        test = load_official_test(RAW / f"Banking77_Arabized_{name}_test.csv", name, by_ar, labels["intent"])
        test.to_parquet(OUT / f"test_{name.lower()}.parquet", index=False)
        print(f"test_{name.lower()}: {len(test):,} texts, {test['label'].nunique()} intents")
    labels.to_json(OUT / "labels.json", orient="records", force_ascii=False, indent=2)

    print(f"dropped {(in_pool & has_quote).sum()} versions with a stray quote")
    print(f"train: {len(train):,} versions from {train['qid'].nunique():,} questions")
    print(f"val:   {len(val):,} versions from {val['qid'].nunique():,} questions")
    print(f"wrote {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
