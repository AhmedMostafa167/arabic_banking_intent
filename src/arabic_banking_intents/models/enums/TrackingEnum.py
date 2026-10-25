from enum import StrEnum


class ExperimentEnum(StrEnum):
    LEAK_CHECK = "leak-check"  # what the official validation split's leak does to a score; never registered
    INTENT_CLASSIFIER = "intent-classifier"  # real candidates, all trained on our grouped split


class AliasEnum(StrEnum):
    CHAMPION = "champion"  # the model the API serves


class ModelFamilyEnum(StrEnum):
    TFIDF_LOGREG = "tfidf-logreg"


class ComputeEnum(StrEnum):
    LAPTOP = "laptop"
    KAGGLE = "kaggle"


class TagEnum(StrEnum):
    GIT_COMMIT = "git_commit"
    GIT_DIRTY = "git_dirty"  # "true" if the run used uncommitted code; such runs are never registered
    DATA_VERSION = "data_version"  # md5 of data/processed in data/processed.dvc
    RAW_DATA_VERSION = "raw_data_version"  # md5 of data/raw in data/raw.dvc
    DATA_VARIANT = "data_variant"
    MODEL_FAMILY = "model_family"
    TRAINED_ON = "trained_on"
