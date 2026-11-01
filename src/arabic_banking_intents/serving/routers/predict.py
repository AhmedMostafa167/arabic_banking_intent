from typing import Annotated

from fastapi import APIRouter, Depends

from arabic_banking_intents.models.enums.ServingEnum import RequestStatusEnum
from arabic_banking_intents.serving import metrics
from arabic_banking_intents.serving.controllers.PredictController import PredictController
from arabic_banking_intents.serving.routers.dependencies import get_predictor
from arabic_banking_intents.serving.routers.schemas.predict import IntentScore, PredictRequest, PredictResponse

predict_router = APIRouter(prefix="/api/v1", tags=["Predict"])


@predict_router.post("/predict")
def predict(request: PredictRequest, predictor: Annotated[PredictController, Depends(get_predictor)]) -> PredictResponse:
    # A plain `def`: the model call blocks, so FastAPI runs this in its thread pool, not on the event loop
    version = predictor.info.version
    with metrics.IN_FLIGHT.track_inprogress():
        try:
            intents = predictor.predict(request.text)
        except Exception:
            metrics.REQUESTS.labels(version, RequestStatusEnum.ERROR).inc()
            raise
    metrics.REQUESTS.labels(version, RequestStatusEnum.OK).inc()
    return PredictResponse(
        intents=[IntentScore(intent=name, confidence=probability) for name, probability in intents],
        model_version=version,
    )
