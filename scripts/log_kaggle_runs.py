"""Wait for the Kaggle training notebook, download its output, and log every run in it to MLflow.

    uv run python scripts/log_kaggle_runs.py                 # the notebook kaggle_push.py started
    uv run python scripts/log_kaggle_runs.py --dir <folder>  # an output that is already downloaded

Each run's metrics are computed here from its saved predictions, with the same code as the
baseline. A saved model is checked before anything is logged: the copy on this machine must
reproduce Kaggle's validation predictions, so the model that is served is the model that was scored.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import mlflow
import numpy as np

from arabic_banking_intents.helpers.config import get_settings
from arabic_banking_intents.training import arabert_model
from arabic_banking_intents.training.arabert_model import AraBertClassifier
from arabic_banking_intents.training.baseline import TEXT_COLUMN
from arabic_banking_intents.training.data import load_label_names, load_test_sets, load_train_val
from arabic_banking_intents.training.evaluate import latency_ms, macro_f1
from arabic_banking_intents.training.tracking import connect, flatten, log_evaluation, log_val_predictions
from arabic_banking_intents.models.enums.DataEnum import DataVariantEnum
from arabic_banking_intents.models.enums.TrackingEnum import (ComputeEnum, ExperimentEnum, ModelFamilyEnum,
                                                              ModelMetadataEnum, TagEnum)

MIN_LABEL_AGREEMENT = 0.995  # fp16 weights and another GPU/CPU may flip a few near-ties, not more
MAX_MACRO_F1_GAP = 0.002
LATENCY_SAMPLE = 200
CURVES = ["loss", "grad_norm", "learning_rate", "eval_loss", "eval_macro_f1"]
POLL_SECONDS = 60


def kaggle(*args: str) -> str:
    result = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{result.stdout}{result.stderr}")
    return result.stdout


def wait_and_download(pending: dict, runs_dir: Path) -> Path:
    while True:
        status = kaggle("kernels", "status", pending["kernel"]).strip()
        print(time.strftime("%H:%M"), status, flush=True)
        if "COMPLETE" in status.upper():
            break
        if any(word in status.upper() for word in ("ERROR", "CANCEL")):
            raise SystemExit(f"the notebook did not finish: {status}; its log is on kaggle.com")
        time.sleep(POLL_SECONDS)
    out = runs_dir / f"output-{pending['git_commit'][:7]}-{time.strftime('%Y%m%d-%H%M%S')}"
    kaggle("kernels", "output", pending["kernel"], "-p", str(out))
    return out


def check_served_copy(model_dir: Path, val, kaggle_predicted: np.ndarray) -> tuple[AraBertClassifier, dict]:
    """Load the model as serving will, re-predict validation on this CPU, compare with Kaggle."""
    classifier = AraBertClassifier(model_dir)
    local_predicted = classifier.predict_proba(val[TEXT_COLUMN].tolist()).argmax(axis=1)
    agreement = float(np.mean(local_predicted == kaggle_predicted))
    gap = abs(macro_f1(val["label"], local_predicted) - macro_f1(val["label"], kaggle_predicted))
    if agreement < MIN_LABEL_AGREEMENT or gap > MAX_MACRO_F1_GAP:
        raise SystemExit(f"{model_dir}: the local copy does not reproduce Kaggle's predictions "
                         f"(agreement {agreement:.4f}, macro-F1 gap {gap:.4f}); nothing logged")
    return classifier, {"consistency_label_agreement": agreement, "consistency_macro_f1_gap": gap}


def log_run(run_dir: Path, settings, intent_names: list[str]) -> None:
    info = json.loads((run_dir / "run_info.json").read_text())
    history = json.loads((run_dir / "history.json").read_text())
    variant = DataVariantEnum(info["variant"])
    _, val = load_train_val(variant, settings.DATA_RAW_DIR, settings.DATA_PROCESSED_DIR)
    test_sets = load_test_sets(settings.DATA_PROCESSED_DIR)

    # Kaggle's predictions must line up row by row with our files
    predictions = {path.stem: np.load(path) for path in (run_dir / "predictions").glob("*.npz")}
    assert np.array_equal(predictions["val"]["labels"], val["label"].to_numpy()), "validation rows differ from ours"
    for test_set, test in test_sets.items():
        assert np.array_equal(predictions[f"test_{test_set.lower()}"]["labels"], test["label"].to_numpy()), f"{test_set} rows differ"
    val_predicted = predictions["val"]["probabilities"].argmax(axis=1)

    model_dir = run_dir / "model"
    served = check_served_copy(model_dir, val, val_predicted) if model_dir.exists() else None

    if info["smoke"]:
        experiment = ExperimentEnum.SMOKE
    elif variant == DataVariantEnum.GROUPED:
        experiment = ExperimentEnum.INTENT_CLASSIFIER
    else:
        experiment = ExperimentEnum.LEAK_CHECK
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"{ModelFamilyEnum.ARABERT}-{info['name']}") as run:
        lineage = info["lineage"]
        mlflow.set_tags({
            TagEnum.GIT_COMMIT: lineage[TagEnum.GIT_COMMIT],
            TagEnum.GIT_DIRTY: lineage[TagEnum.GIT_DIRTY],
            TagEnum.DATA_VERSION: lineage[TagEnum.DATA_VERSION],
            TagEnum.RAW_DATA_VERSION: lineage[TagEnum.RAW_DATA_VERSION],
            TagEnum.DATA_VARIANT: variant,
            TagEnum.MODEL_FAMILY: ModelFamilyEnum.ARABERT,
            TagEnum.TRAINED_ON: ComputeEnum.KAGGLE,
            TagEnum.GPU: info["gpu"],
        })
        mlflow.log_params(flatten(info["config"]) | flatten(info["packages"], "package.") | {
            "seed": info["seed"], "data_variant": variant, "train_rows": info["train_rows"], "val_rows": info["val_rows"],
        })
        mlflow.log_metrics({key: info[key] for key in ("training_seconds", "best_epoch", "epochs_run", "truncated_share")})
        for entry in history:  # the training curves, at the steps they were recorded
            for key in CURVES:
                if key in entry:
                    mlflow.log_metric(f"curve/{key}", entry[key], step=entry["step"])

        groups = val["qid"] if variant == DataVariantEnum.GROUPED else None
        log_evaluation("val", val["label"], val_predicted, intent_names, groups)
        for test_set, test in test_sets.items():
            key = f"test_{test_set.lower()}"
            log_evaluation(key, test["label"], predictions[key]["probabilities"].argmax(axis=1), intent_names)
        log_val_predictions(val, val_predicted)

        if served:
            classifier, consistency = served
            p50, p95 = latency_ms(lambda text: classifier.predict_proba([text]), val[TEXT_COLUMN].head(LATENCY_SAMPLE).tolist())
            size_mb = sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file()) / 1e6
            mlflow.log_metrics(consistency | {"latency_p50_ms": p50, "latency_p95_ms": p95, "model_size_mb": size_mb})
            mlflow.log_artifact(str(settings.DATA_PROCESSED_DIR / "labels.json"))
            # Logged as code: the wrapper file travels inside the model, so moving it here breaks nothing
            mlflow.pyfunc.log_model(name="model", python_model=arabert_model.__file__,
                                    artifacts={"model_dir": str(model_dir)}, input_example=val[[TEXT_COLUMN]].head(3),
                                    metadata={ModelMetadataEnum.INTENT_NAMES: intent_names})
    print(f"logged {info['name']} to {experiment} as run {run.info.run_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Kaggle output and log its runs to MLflow.")
    parser.add_argument("--dir", type=Path, help="an output folder that is already downloaded")
    args = parser.parse_args()

    settings = get_settings()
    connect(settings.MLFLOW_TRACKING_URI)
    pending = None
    if args.dir:
        out = args.dir
    else:
        pending = json.loads((settings.RUNS_DIR / "pending.json").read_text())
        out = wait_and_download(pending, settings.RUNS_DIR)

    run_dirs = sorted(path.parent for path in out.rglob("run_info.json"))
    if not run_dirs:
        raise SystemExit(f"no runs found in {out}")
    for run_dir in run_dirs:  # the notebook trained the inputs that were pushed, not an older version
        commit = json.loads((run_dir / "run_info.json").read_text())["lineage"][TagEnum.GIT_COMMIT]
        if pending and commit != pending["git_commit"]:
            raise SystemExit(f"{run_dir.name} was trained from commit {commit[:7]}, not the pushed {pending['git_commit'][:7]}")

    intent_names = load_label_names(settings.DATA_PROCESSED_DIR)
    for run_dir in run_dirs:
        log_run(run_dir, settings, intent_names)


if __name__ == "__main__":
    main()
