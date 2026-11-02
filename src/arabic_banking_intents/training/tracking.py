"""MLflow helpers: connect loudly, tag every run with the code and data behind it, log evaluations."""

import subprocess
import tempfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from numpy.typing import ArrayLike

from arabic_banking_intents.models.enums.TrackingEnum import TagEnum
from arabic_banking_intents.training.evaluate import bootstrap, evaluate, interval


def connect(tracking_uri: str) -> None:
    """Point MLflow at the tracking server, and stop now if it isn't running.

    Otherwise the first log call retries for minutes before failing, after the model has trained.
    """
    try:
        urllib.request.urlopen(f"{tracking_uri}/health", timeout=5)
    except OSError as error:
        raise RuntimeError(f"no MLflow server at {tracking_uri}; start it first (see README)") from error
    mlflow.set_tracking_uri(tracking_uri)


def _dvc_md5(data_dir: Path) -> str:
    """The hash DVC recorded for a tracked folder, read from its .dvc file."""
    return yaml.safe_load(data_dir.with_suffix(".dvc").read_text())["outs"][0]["md5"]


def lineage_tags(project_root: Path, raw_dir: Path, processed_dir: Path) -> dict[str, str]:
    """The git commit (and whether the code had uncommitted changes) and both data versions."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=project_root, capture_output=True, text=True, check=True).stdout.strip()

    return {
        TagEnum.GIT_COMMIT: git("rev-parse", "HEAD"),
        TagEnum.GIT_DIRTY: str(bool(git("status", "--porcelain"))).lower(),
        TagEnum.DATA_VERSION: _dvc_md5(processed_dir),
        TagEnum.RAW_DATA_VERSION: _dvc_md5(raw_dir),
    }


def log_evaluation(
    prefix: str,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    intent_names: Sequence[str],
    groups: ArrayLike | None = None,
) -> dict:
    """Accuracy, macro-F1 and its 95% bootstrap interval as metrics; per-intent F1 and confusions as JSON."""
    result = evaluate(y_true, y_pred, intent_names)
    low, high = interval(bootstrap(y_true, y_pred, groups))
    mlflow.log_metrics({
        f"{prefix}_accuracy": result["accuracy"],
        f"{prefix}_macro_f1": result["macro_f1"],
        f"{prefix}_macro_f1_ci_low": low,
        f"{prefix}_macro_f1_ci_high": high,
    })
    mlflow.log_dict(result["per_intent_f1"], f"evaluation/{prefix}_per_intent_f1.json")
    mlflow.log_dict(result["top_confusions"], f"evaluation/{prefix}_top_confusions.json")
    return result


def flatten(config: dict, prefix: str = "") -> dict:
    """{"tfidf": {"min_df": 2}} -> {"tfidf.min_df": 2}, for MLflow params."""
    flat = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flat |= flatten(value, f"{prefix}{key}.")
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def log_val_predictions(val: pd.DataFrame, predicted: ArrayLike) -> None:
    """Keep the validation predictions, so later comparisons (the leak check, the paired gate) can
    re-score without retraining."""
    predictions = val.assign(predicted=predicted)
    predictions = predictions[[c for c in ("qid", "label", "predicted") if c in predictions]]
    assert "predicted" in predictions, "validation predictions are missing from the artifact"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "val_predictions.parquet"
        predictions.to_parquet(path, index=False)
        mlflow.log_artifact(str(path), "evaluation")
