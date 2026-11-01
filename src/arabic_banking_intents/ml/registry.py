"""The model registry: the checks a run must pass to be served, and the record a version carries."""

import tempfile

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities import Run

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
    alias: AliasEnum = AliasEnum.CHAMPION,
    note: str = "",
    extra_evidence: dict[str, str] | None = None,
) -> str:
    """Register the run's model, record the evidence it was promoted on, and point `alias` at it."""
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
    version = mlflow.register_model(logged_model_uri(run), model_name, tags=evidence)
    description = "; ".join(f"{key} {value}" for key, value in evidence.items())
    client.update_model_version(model_name, version.version, description=f"{note}\n\n{description}".strip())
    client.set_registered_model_alias(model_name, alias, version.version)
    return f"{model_name} v{version.version} is now @{alias}: {description}"
