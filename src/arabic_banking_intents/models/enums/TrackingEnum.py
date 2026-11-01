from enum import StrEnum


class ExperimentEnum(StrEnum):
    LEAK_CHECK = "leak-check"  # what the official validation split's leak does to a score; never registered
    INTENT_CLASSIFIER = "intent-classifier"  # real candidates, all trained on our grouped split
    SMOKE = "smoke"  # quick end-to-end checks of the Kaggle loop; safe to delete
    SERVING_BENCHMARKS = "serving-benchmarks"  # load tests of the running API, one run per load level
    OPTIMISATION = "optimisation"  # the ONNX variants of a version, one run per rung; never registered


class AliasEnum(StrEnum):
    CHAMPION = "champion"  # the model the API serves


class ModelMetadataEnum(StrEnum):
    # The 77 names, in the order of the model's probability columns. Write it with .value: the
    # metadata goes into the MLmodel YAML file, and the YAML writer rejects enum members
    INTENT_NAMES = "intent_names"


class ModelFamilyEnum(StrEnum):
    TFIDF_LOGREG = "tfidf-logreg"
    ARABERT = "arabert"


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
    GPU = "gpu"
    RELOGGED_FROM = "relogged_from"  # on a registered copy of an earlier version, logged again unchanged
    RELOG_COMMIT = "relog_commit"  # the commit of the code that logged that copy
    MODEL_VERSION = "model_version"  # on a benchmark run: the registered version the API was serving
    RUNG = "rung"  # the step of the optimisation ladder a run measures
    DERIVED_FROM = "derived_from"  # on an optimised variant: the registered version it was built from
