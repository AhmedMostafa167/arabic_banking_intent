"""Typed settings, read once from the environment and `.env` (template: `.env.example`).

Paths are settings rather than constants so another machine (a Kaggle notebook, a container) can
point them elsewhere without a code change.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from arabic_banking_intents.models.enums.TrackingEnum import AliasEnum

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    MLFLOW_TRACKING_URI: str = "http://127.0.0.1:5000"
    REGISTERED_MODEL_NAME: str = "arabic-banking-intents"

    # Serving: the API loads the version this alias points to, once, at startup
    MODEL_ALIAS: AliasEnum = AliasEnum.CHAMPION
    TOP_K: int = Field(default=3, ge=1, le=77)
    MAX_TEXT_CHARS: int = Field(default=1000, ge=1)  # the longest message in the data has 346 characters

    DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
    CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
    RUNS_DIR: Path = PROJECT_ROOT / "runs"  # Kaggle uploads and downloaded outputs; not in git

    KAGGLE_USERNAME: str | None = None  # required by the Kaggle scripts
    KAGGLE_DATASET: str = "arabic-banking-intents-inputs"
    KAGGLE_KERNEL: str = "arabic-banking-intents-train"

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
