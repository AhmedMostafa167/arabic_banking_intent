"""Train the TF-IDF + logistic regression baseline and log the run to MLflow.

    uv run python scripts/train_baseline.py --data official --experiment leak-check
    uv run python scripts/train_baseline.py --data grouped --experiment leak-check
    uv run python scripts/train_baseline.py --data grouped --experiment intent-classifier

`official` trains on SinaLab's train file and scores on their (leaky) validation file; `grouped`
uses our split. Both are also scored on the five official test sets. Start the MLflow server
first (see README), and commit first: a run from uncommitted code is tagged git_dirty=true.
"""

import argparse
import pickle
import time
import tomllib
from pathlib import Path

import mlflow
import pandas as pd

from arabic_banking_intents.helpers.config import PROJECT_ROOT, get_settings
from arabic_banking_intents.training.baseline import TEXT_COLUMN, build_pipeline
from arabic_banking_intents.training.data import load_label_names, load_test_sets, load_train_val
from arabic_banking_intents.training.evaluate import latency_ms
from arabic_banking_intents.training.tracking import connect, flatten, lineage_tags, log_evaluation, log_val_predictions
from arabic_banking_intents.models.enums.DataEnum import DataVariantEnum
from arabic_banking_intents.models.enums.TrackingEnum import (ComputeEnum, ExperimentEnum, ModelFamilyEnum,
                                                              ModelMetadataEnum, TagEnum)

LATENCY_SAMPLE = 200  # validation messages timed one at a time


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train and log the TF-IDF baseline.")
    parser.add_argument("--data", type=DataVariantEnum, choices=list(DataVariantEnum), required=True)
    parser.add_argument("--experiment", type=ExperimentEnum, choices=list(ExperimentEnum), required=True)
    parser.add_argument("--config", type=Path, default=settings.CONFIGS_DIR / "baseline.toml")
    args = parser.parse_args()
    if args.experiment == ExperimentEnum.INTENT_CLASSIFIER and args.data != DataVariantEnum.GROUPED:
        parser.error("intent-classifier candidates are trained on our grouped split only")

    connect(settings.MLFLOW_TRACKING_URI)
    config = tomllib.loads(args.config.read_text())
    intent_names = load_label_names(settings.DATA_PROCESSED_DIR)
    train, val = load_train_val(args.data, settings.DATA_RAW_DIR, settings.DATA_PROCESSED_DIR)
    test_sets = load_test_sets(settings.DATA_PROCESSED_DIR)
    pipeline = build_pipeline(config)

    params = flatten(config) | {"data_variant": args.data, "train_rows": len(train), "val_rows": len(val)}
    if args.data == DataVariantEnum.GROUPED:
        params |= flatten(tomllib.loads((settings.CONFIGS_DIR / "data.toml").read_text()), "split.")

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"{ModelFamilyEnum.TFIDF_LOGREG}-{args.data}"):
        mlflow.set_tags(lineage_tags(PROJECT_ROOT, settings.DATA_RAW_DIR, settings.DATA_PROCESSED_DIR) | {
            TagEnum.DATA_VARIANT: args.data,
            TagEnum.MODEL_FAMILY: ModelFamilyEnum.TFIDF_LOGREG,
            TagEnum.TRAINED_ON: ComputeEnum.LAPTOP,
        })
        mlflow.log_params(params)

        start = time.perf_counter()
        pipeline.fit(train[[TEXT_COLUMN]], train["label"])
        mlflow.log_metric("training_seconds", time.perf_counter() - start)

        # The official validation file has no question IDs, so its interval resamples single texts
        val_pred = pipeline.predict(val[[TEXT_COLUMN]])
        groups = val["qid"] if args.data == DataVariantEnum.GROUPED else None
        log_evaluation("val", val["label"], val_pred, intent_names, groups)
        for test_set, test in test_sets.items():
            log_evaluation(f"test_{test_set.lower()}", test["label"], pipeline.predict(test[[TEXT_COLUMN]]), intent_names)

        p50, p95 = latency_ms(
            lambda text: pipeline.predict_proba(pd.DataFrame({TEXT_COLUMN: [text]})),
            val[TEXT_COLUMN].head(LATENCY_SAMPLE).tolist(),
        )
        mlflow.log_metrics({"latency_p50_ms": p50, "latency_p95_ms": p95, "model_size_mb": len(pickle.dumps(pipeline)) / 1e6})

        log_val_predictions(val, val_pred)
        mlflow.log_artifact(str(settings.DATA_PROCESSED_DIR / "labels.json"))
        # Probabilities, like every model here: the API can serve whichever model is champion the same way
        mlflow.sklearn.log_model(pipeline, name="model", input_example=val[[TEXT_COLUMN]].head(3),
                                 pyfunc_predict_fn="predict_proba", metadata={ModelMetadataEnum.INTENT_NAMES.value: intent_names})


if __name__ == "__main__":
    main()
