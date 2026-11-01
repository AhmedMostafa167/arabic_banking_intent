"""The prediction API: loads the model the alias points to once, at startup, and serves it.

    uv run uvicorn arabic_banking_intents.serving.main:app      # needs the MLflow server (see README)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from arabic_banking_intents.helpers.config import get_settings
from arabic_banking_intents.serving.controllers.PredictController import PredictController
from arabic_banking_intents.serving.routers.health import health_router
from arabic_banking_intents.serving.routers.metrics import metrics_router
from arabic_banking_intents.serving.routers.predict import predict_router

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load before accepting any request. If this raises, the server stops with the reason,
    # instead of starting and then failing every request.
    predictor = PredictController.from_registry(get_settings())
    info = predictor.info
    logger.info(f"serving {info.name} v{info.version} (@{info.alias}, {info.model_family}, run {info.run_id})")
    app.state.predictor = predictor
    yield


app = FastAPI(title="Arabic banking intents", lifespan=lifespan)
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(metrics_router)
