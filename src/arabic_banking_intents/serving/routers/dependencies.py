from fastapi import HTTPException, Request, status

from arabic_banking_intents.serving.controllers.PredictController import PredictController


def get_predictor(request: Request) -> PredictController:
    """The controller loaded at startup; 503 while there is none. Tests override this."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "no model loaded")
    return predictor
