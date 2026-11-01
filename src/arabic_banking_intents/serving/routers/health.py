from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends

from arabic_banking_intents.serving.controllers.PredictController import PredictController
from arabic_banking_intents.serving.routers.dependencies import get_predictor

health_router = APIRouter(prefix="/api/v1", tags=["Health"])


@health_router.get("/health")
async def health() -> dict:
    """Liveness: the process answers (restart it if this fails)."""
    return {"status": "ok"}


@health_router.get("/ready")
async def ready(predictor: Annotated[PredictController, Depends(get_predictor)]) -> dict:
    """Readiness: a model is loaded and passed its startup check (503 otherwise: send no traffic)."""
    return {"status": "ready", "model_version": predictor.info.version}


@health_router.get("/info")
async def info(predictor: Annotated[PredictController, Depends(get_predictor)]) -> dict:
    """Which model answers requests: name, alias, version, run, family, commit."""
    return asdict(predictor.info)
