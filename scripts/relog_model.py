"""Log a registered model again, unchanged, the way models are logged now, and register the copy.

    uv run python scripts/relog_model.py --version 4 --alias champion --note "why"
    uv run python scripts/relog_model.py --version 3 --note "why"          # registered, no alias

For a change in how models are logged, not in the model itself. The copy goes into the original
run, so it keeps that run's lineage and evidence, and it is registered only if it reproduces the
run's saved validation predictions exactly. Commit first: the version records this commit as
relog_commit, next to the training commit it already had.
"""

import argparse
import tempfile
from pathlib import Path

import mlflow
import numpy as np
from mlflow import MlflowClient
from mlflow.models import Model

from arabic_banking_intents.helpers.config import PROJECT_ROOT, get_settings
from arabic_banking_intents.models.enums.DataEnum import DataVariantEnum
from arabic_banking_intents.models.enums.TrackingEnum import AliasEnum, ModelFamilyEnum, ModelMetadataEnum, TagEnum
from arabic_banking_intents.training import arabert_model
from arabic_banking_intents.training.baseline import TEXT_COLUMN
from arabic_banking_intents.training.data import load_label_names, load_train_val
from arabic_banking_intents.training.registry import candidate_problems, register, val_predictions
from arabic_banking_intents.training.tracking import connect, lineage_tags


def log_copy(source_uri: str, family: str, example, intent_names: list[str], tmp: Path) -> str:
    """Log the model at `source_uri` again into the active run; returns the new model's URI."""
    metadata = {ModelMetadataEnum.INTENT_NAMES.value: intent_names}
    if family == ModelFamilyEnum.ARABERT:
        local = Path(mlflow.artifacts.download_artifacts(artifact_uri=source_uri, dst_path=str(tmp)))
        # MLmodel maps each artifact key to where it was stored (named after the source folder, not the key)
        stored = Model.load(local / "MLmodel").flavors["python_function"]["artifacts"]["model_dir"]["path"]
        model_dir = local / stored
        assert (model_dir / arabert_model.MODEL_INFO_FILE).exists(), f"no AraBERT model folder in {local}"
        info = mlflow.pyfunc.log_model(name="model", python_model=arabert_model.__file__,
                                       artifacts={"model_dir": str(model_dir)}, input_example=example, metadata=metadata)
    elif family == ModelFamilyEnum.TFIDF_LOGREG:
        info = mlflow.sklearn.log_model(mlflow.sklearn.load_model(source_uri), name="model", input_example=example,
                                        pyfunc_predict_fn="predict_proba", metadata=metadata)
    else:
        raise SystemExit(f"don't know how to log model family '{family}'")
    return info.model_uri


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-log a registered model unchanged and register the copy.")
    parser.add_argument("--version", required=True, help="the registered version to copy")
    parser.add_argument("--alias", type=AliasEnum, choices=list(AliasEnum), default=None,
                        help="point this alias at the copy (default: none)")
    parser.add_argument("--note", required=True, help="why the copy is registered")
    args = parser.parse_args()

    settings = get_settings()
    lineage = lineage_tags(PROJECT_ROOT, settings.DATA_RAW_DIR, settings.DATA_PROCESSED_DIR)
    if lineage[TagEnum.GIT_DIRTY] != "false":
        raise SystemExit("commit first: the copy records the commit that logged it")
    connect(settings.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    source = client.get_model_version(settings.REGISTERED_MODEL_NAME, args.version)
    run = client.get_run(source.run_id)
    problems = candidate_problems(client, run)
    if problems:
        raise SystemExit("; ".join(problems))

    family = run.data.tags[TagEnum.MODEL_FAMILY]
    _, val = load_train_val(DataVariantEnum.GROUPED, settings.DATA_RAW_DIR, settings.DATA_PROCESSED_DIR)
    intent_names = load_label_names(settings.DATA_PROCESSED_DIR)
    with tempfile.TemporaryDirectory() as tmp:
        mlflow.start_run(run_id=run.info.run_id)
        try:
            model_uri = log_copy(source.source, family, val[[TEXT_COLUMN]].head(3), intent_names, Path(tmp))
        finally:
            mlflow.end_run()  # FINISHED even if the copy failed: that says nothing about the training run

    # The copy must give the same validation predictions the run recorded, row for row
    copy = mlflow.pyfunc.load_model(model_uri)
    assert copy.metadata.metadata[ModelMetadataEnum.INTENT_NAMES] == intent_names
    predicted = np.asarray(copy.predict(val[[TEXT_COLUMN]])).argmax(axis=1)
    saved = val_predictions(run.info.run_id)
    assert np.array_equal(saved["label"].to_numpy(), val["label"].to_numpy()), "validation rows differ"
    differ = int((predicted != saved["predicted"].to_numpy()).sum())
    if differ:
        raise SystemExit(f"the copy ({model_uri}) disagrees with the run on {differ} validation texts; not registered")

    evidence = {TagEnum.RELOGGED_FROM: f"v{args.version}", TagEnum.RELOG_COMMIT: lineage[TagEnum.GIT_COMMIT]}
    print(register(client, run, settings.REGISTERED_MODEL_NAME, args.alias, args.note, evidence, model_uri))


if __name__ == "__main__":
    main()
