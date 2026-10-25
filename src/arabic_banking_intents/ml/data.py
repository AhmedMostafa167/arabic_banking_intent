"""ArBanking77: the raw SinaLab files, our train/validation split, and the label map.

The exploration behind each rule here is in notebooks/01_explore_data.ipynb.
"""

import csv
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from arabic_banking_intents.models.enums.DataEnum import DataVariantEnum, TestSetEnum

EXPECTED_MALFORMED_ROWS = 8  # unescaped quotes split these rows into 14-15 fields
SLOTS = ["MSA1", "MSA2", "PAL1", "PAL2"]
ID_PATTERN = r"(Tr|D|Te)\d+"
SPLIT_NAMES = {"Tr": "train", "D": "val", "Te": "test"}
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


def to_versions(corpus: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
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
    versions["label"] = to_label(versions["intent"], dict(zip(labels["intent"], labels["label"])))
    return versions


def build_pool(versions: pd.DataFrame) -> pd.DataFrame:
    """The corpus's train and validation versions, without the texts left with a stray quote."""
    in_pool = versions["split"].isin(["train", "val"])
    has_quote = versions["text"].str.contains('"', regex=False)  # left over from the broken CSV quoting
    return versions[in_pool & ~has_quote].reset_index(drop=True)


def split_by_question(pool: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified by intent and grouped by question: all versions of a question land on one side."""
    sgkf = StratifiedGroupKFold(n_splits=round(1 / val_fraction), shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(pool, y=pool["label"], groups=pool["qid"]))
    train, val = pool.iloc[train_idx], pool.iloc[val_idx]

    assert not set(train["qid"]) & set(val["qid"]), "a question is in both train and val"
    assert train["label"].nunique() == val["label"].nunique() == 77, "an intent is missing"
    assert len(train) + len(val) == len(pool), "rows lost in the split"
    return train, val


def load_official(path: Path, labels: pd.DataFrame, dialect: str | None = None) -> pd.DataFrame:
    """An official SinaLab file (Arabic intent name + text), with our label IDs."""
    frame = pd.read_csv(path)
    frame["label"] = to_label(frame["label"], dict(zip(labels["intent_ar"], labels["label"])))
    frame["intent"] = frame["label"].map(labels["intent"])
    frame["dialect"] = dialect
    return frame[["dialect", "text", "label", "intent"]]


def load_label_names(processed_dir: Path) -> list[str]:
    """Intent names indexed by label, from labels.json."""
    labels = json.loads((processed_dir / "labels.json").read_text(encoding="utf-8"))
    return [entry["intent"] for entry in sorted(labels, key=lambda entry: entry["label"])]


def load_train_val(variant: DataVariantEnum, raw_dir: Path, processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Our grouped split, or SinaLab's official train and validation files (whose validation leaks)."""
    if variant == DataVariantEnum.GROUPED:
        return pd.read_parquet(processed_dir / "train.parquet"), pd.read_parquet(processed_dir / "val.parquet")
    labels = load_labels(raw_dir / "Banking77_intents.csv")
    return (
        load_official(raw_dir / "Banking77_Arabized_MSA_PAL_train.csv", labels),
        load_official(raw_dir / "Banking77_Arabized_MSA_PAL_val.csv", labels),
    )


def load_test_sets(processed_dir: Path) -> dict[TestSetEnum, pd.DataFrame]:
    return {test_set: pd.read_parquet(processed_dir / f"test_{test_set.lower()}.parquet") for test_set in TestSetEnum}
