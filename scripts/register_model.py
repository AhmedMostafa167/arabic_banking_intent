"""Register a candidate run's model and point an alias (default: champion) at the new version.

    uv run python scripts/register_model.py --run-id <run id> [--note "why it was promoted"]

Refuses runs that must never be served: anything from the leak-check experiment, anything trained
on the leaky official split, and anything run from uncommitted code. The version's description
and tags record the evidence it was promoted on.
"""

import argparse

import mlflow
from mlflow import MlflowClient

from arabic_banking_intents.helpers.config import get_settings
from arabic_banking_intents.ml.tracking import connect
from arabic_banking_intents.models.enums.DataEnum import DataVariantEnum
from arabic_banking_intents.models.enums.TrackingEnum import AliasEnum, ExperimentEnum, TagEnum

EVIDENCE_METRICS = ["val_macro_f1", "val_macro_f1_ci_low", "val_macro_f1_ci_high", "latency_p95_ms"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a candidate model and set an alias.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--alias", type=AliasEnum, choices=list(AliasEnum), default=AliasEnum.CHAMPION)
    parser.add_argument("--note", default="", help="why this version is promoted")
    args = parser.parse_args()

    settings = get_settings()
    connect(settings.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    run = client.get_run(args.run_id)
    tags, metrics = run.data.tags, run.data.metrics

    experiment = client.get_experiment(run.info.experiment_id).name
    if experiment != ExperimentEnum.INTENT_CLASSIFIER:
        raise SystemExit(f"run is in '{experiment}'; only {ExperimentEnum.INTENT_CLASSIFIER} runs are candidates")
    if tags.get(TagEnum.DATA_VARIANT) != DataVariantEnum.GROUPED:
        raise SystemExit("run was not trained on our grouped split")
    if tags.get(TagEnum.GIT_DIRTY) != "false":
        raise SystemExit("run used uncommitted code (git_dirty is not 'false'); commit and re-run it")
    missing = [m for m in EVIDENCE_METRICS if m not in metrics]
    if missing:
        raise SystemExit(f"run is missing metrics: {missing}")

    logged = mlflow.search_logged_models(
        experiment_ids=[run.info.experiment_id], filter_string=f"source_run_id = '{run.info.run_id}'", output_format="list"
    )
    if len(logged) != 1:
        raise SystemExit(f"expected one logged model in the run, found {len(logged)}")

    evidence = {
        "val_macro_f1": f"{metrics['val_macro_f1']:.4f}",
        "val_macro_f1_95ci": f"{metrics['val_macro_f1_ci_low']:.4f}-{metrics['val_macro_f1_ci_high']:.4f}",
        "latency_p95_ms": f"{metrics['latency_p95_ms']:.1f}",
        TagEnum.GIT_COMMIT: tags[TagEnum.GIT_COMMIT],
        TagEnum.DATA_VERSION: tags[TagEnum.DATA_VERSION],
    }
    version = mlflow.register_model(logged[0].model_uri, settings.REGISTERED_MODEL_NAME, tags=evidence)
    description = "; ".join(f"{key} {value}" for key, value in evidence.items())
    client.update_model_version(settings.REGISTERED_MODEL_NAME, version.version, description=f"{args.note}\n\n{description}".strip())
    client.set_registered_model_alias(settings.REGISTERED_MODEL_NAME, args.alias, version.version)
    print(f"{settings.REGISTERED_MODEL_NAME} v{version.version} is now @{args.alias}: {description}")


if __name__ == "__main__":
    main()
