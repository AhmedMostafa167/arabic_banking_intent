"""The model registry: the checks a run must pass to be served, and the record a version carries."""

import shutil
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities import Run
from mlflow.models import Model

from arabic_banking_intents.models.enums.DataEnum import DataVariantEnum
from arabic_banking_intents.models.enums.TrackingEnum import AliasEnum, ExperimentEnum, TagEnum

EVIDENCE_METRICS = ["val_macro_f1", "val_macro_f1_ci_low", "val_macro_f1_ci_high", "latency_p95_ms"]


def candidate_problems(client: MlflowClient, run: Run) -> list[str]:
    """Why a run must not be served; empty if it may be.

    Never served: runs outside intent-classifier (the leak check, smoke tests), models trained on the
    leaky official split, and runs from uncommitted code.
    """
    problems = []
    experiment = client.get_experiment(run.info.experiment_id).name
    if experiment != ExperimentEnum.INTENT_CLASSIFIER:
        problems.append(f"run is in '{experiment}'; only {ExperimentEnum.INTENT_CLASSIFIER} runs are candidates")
    if run.data.tags.get(TagEnum.DATA_VARIANT) != DataVariantEnum.GROUPED:
        problems.append("run was not trained on our grouped split")
    if run.data.tags.get(TagEnum.GIT_DIRTY) != "false":
        problems.append("run used uncommitted code (git_dirty is not 'false'); commit and re-run it")
    missing = [m for m in EVIDENCE_METRICS if m not in run.data.metrics]
    if missing:
        problems.append(f"run is missing metrics: {missing}")
    return problems


def logged_model_uri(run: Run) -> str:
    logged = mlflow.search_logged_models(
        experiment_ids=[run.info.experiment_id], filter_string=f"source_run_id = '{run.info.run_id}'", output_format="list"
    )
    if len(logged) != 1:
        raise SystemExit(f"expected one logged model in run {run.info.run_id}, found {len(logged)}")
    return logged[0].model_uri


def download_model_folder(model_uri: str, destination: Path) -> Path:
    """The model folder inside a logged pyfunc model (its `model_dir` artifact), copied to `destination` once."""
    if not destination.exists():
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=tmp))
            # MLmodel maps each artifact key to where it was stored (named after the source folder, not the key)
            stored = Model.load(local / "MLmodel").flavors["python_function"]["artifacts"]["model_dir"]["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(local / stored, destination)
    return destination


def val_predictions(run_id: str) -> pd.DataFrame:
    """The validation predictions a run saved (qid, label, predicted), in validation-set order."""
    with tempfile.TemporaryDirectory() as tmp:
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="evaluation/val_predictions.parquet", dst_path=tmp
        )
        return pd.read_parquet(path)


def register(
    client: MlflowClient,
    run: Run,
    model_name: str,
    alias: AliasEnum | None = AliasEnum.CHAMPION,
    note: str = "",
    extra_evidence: dict[str, str] | None = None,
    model_uri: str | None = None,
) -> str:
    """Register the run's model, record the evidence it was promoted on, and point `alias` at it.

    `model_uri` picks one model when the run holds several (a re-logged copy); `alias=None` registers
    the version without moving any alias.
    """
    metrics, tags = run.data.metrics, run.data.tags
    evidence = {
        "val_macro_f1": f"{metrics['val_macro_f1']:.4f}",
        "val_macro_f1_95ci": f"{metrics['val_macro_f1_ci_low']:.4f}-{metrics['val_macro_f1_ci_high']:.4f}",
        "latency_p95_ms": f"{metrics['latency_p95_ms']:.1f}",
        TagEnum.MODEL_FAMILY: tags.get(TagEnum.MODEL_FAMILY, ""),
        TagEnum.GIT_COMMIT: tags[TagEnum.GIT_COMMIT],
        TagEnum.DATA_VERSION: tags[TagEnum.DATA_VERSION],
        **(extra_evidence or {}),
    }
    version = mlflow.register_model(model_uri or logged_model_uri(run), model_name, tags=evidence)
    description = "; ".join(f"{key} {value}" for key, value in evidence.items())
    client.update_model_version(model_name, version.version, description=f"{note}\n\n{description}".strip())
    if alias is None:
        return f"{model_name} v{version.version} registered (no alias): {description}"
    client.set_registered_model_alias(model_name, alias, version.version)
    return f"{model_name} v{version.version} is now @{alias}: {description}"
