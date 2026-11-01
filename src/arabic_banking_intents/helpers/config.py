"""Typed settings, read once from the environment and `.env` (template: `.env.example`).

Paths are settings rather than constants so another machine (a Kaggle notebook, a container) can
point them elsewhere without a code change.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    MLFLOW_TRACKING_URI: str = "http://127.0.0.1:5000"
    REGISTERED_MODEL_NAME: str = "arabic-banking-intents"

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
