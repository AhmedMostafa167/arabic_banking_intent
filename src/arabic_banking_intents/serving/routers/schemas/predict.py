from pydantic import BaseModel, ConfigDict, Field

from arabic_banking_intents.helpers.config import get_settings


class PredictRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)  # "   " becomes "" and is rejected

    text: str = Field(min_length=1, max_length=get_settings().MAX_TEXT_CHARS, description="One customer message")


class IntentScore(BaseModel):
    intent: str
    confidence: float = Field(ge=0, le=1)


class PredictResponse(BaseModel):
    intents: list[IntentScore]  # most likely first
    model_version: str
