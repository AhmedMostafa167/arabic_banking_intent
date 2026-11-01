"""The served model: loaded from the registry once, at startup, and used for every request.

No FastAPI here: the routers handle HTTP, this class handles the model.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from arabic_banking_intents.helpers.config import Settings
from arabic_banking_intents.models.enums.TrackingEnum import ModelMetadataEnum, TagEnum
from arabic_banking_intents.serving import metrics

WARM_UP_TEXT = "أريد تفعيل بطاقتي"  # "I want to activate my card": one real prediction before any request


@dataclass(frozen=True)
class ServedModel:
    """Which model answers: what /info reports, and the version every metric is labelled with."""

    name: str
    alias: str
    version: str
    run_id: str
    model_family: str
    git_commit: str
    intents: int
    loaded_at: str


class PredictController:
    def __init__(self, model: mlflow.pyfunc.PyFuncModel, intent_names: list[str], info: ServedModel, top_k: int):
        self.model = model
        self.intent_names = intent_names
        self.info = info
        self.top_k = top_k
        self.input_column = model.metadata.get_input_schema().input_names()[0]  # the model declares its input

    @classmethod
    def from_registry(cls, settings: Settings) -> "PredictController":
        """Resolve the alias to one version, load it and check it; raises, naming the problem, if anything is wrong."""
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        name = settings.REGISTERED_MODEL_NAME
        try:
            version = MlflowClient().get_model_version_by_alias(name, settings.MODEL_ALIAS)
        except MlflowException as error:
            raise RuntimeError(f"no version of '{name}' has the alias @{settings.MODEL_ALIAS} "
                               f"at {settings.MLFLOW_TRACKING_URI}") from error
        # Load the version, not the alias: moving the alias later must not change what this process serves
        model = mlflow.pyfunc.load_model(f"models:/{name}/{version.version}")
        intent_names = (model.metadata.metadata or {}).get(ModelMetadataEnum.INTENT_NAMES)
        if not intent_names:
            raise RuntimeError(f"{name} v{version.version} carries no intent names; re-log it (scripts/relog_model.py)")
        info = ServedModel(
            name=name,
            alias=settings.MODEL_ALIAS,
            version=version.version,
            run_id=version.run_id,
            model_family=version.tags.get(TagEnum.MODEL_FAMILY, ""),
            git_commit=version.tags.get(TagEnum.GIT_COMMIT, ""),
            intents=len(intent_names),
            loaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        controller = cls(model, intent_names, info, settings.TOP_K)
        controller.check()
        return controller

    def check(self) -> None:
        """One real prediction: the model must return one probability per intent name."""
        probabilities = self.probabilities(WARM_UP_TEXT)
        if probabilities.shape != (len(self.intent_names),):
            raise RuntimeError(f"v{self.info.version} returns {probabilities.shape[0]} probabilities "
                               f"for {len(self.intent_names)} intent names")

    def probabilities(self, text: str) -> np.ndarray:
        return np.asarray(self.model.predict(pd.DataFrame({self.input_column: [text]})))[0]

    def predict(self, text: str) -> list[tuple[str, float]]:
        """The top-k intents with their probabilities, most likely first."""
        start = time.perf_counter()
        probabilities = self.probabilities(text)
        metrics.INFERENCE_SECONDS.labels(self.info.version).observe(time.perf_counter() - start)
        top = np.argsort(probabilities)[::-1][: self.top_k]
        metrics.PREDICTED_INTENTS.labels(self.info.version, self.intent_names[top[0]]).inc()
        metrics.TOP_CONFIDENCE.labels(self.info.version).observe(float(probabilities[top[0]]))
        return [(self.intent_names[i], float(probabilities[i])) for i in top]
