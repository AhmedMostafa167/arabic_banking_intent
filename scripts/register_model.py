"""Register a candidate run's model and point an alias (default: champion) at the new version.

    uv run python scripts/register_model.py --run-id <run id> [--note "why it was registered"]

For the first champion, or to re-register the same model. A new model family replacing the
champion goes through scripts/promote.py, which applies the agreed gate first.
"""

import argparse

from mlflow import MlflowClient

from arabic_banking_intents.helpers.config import get_settings
from arabic_banking_intents.models.enums.TrackingEnum import AliasEnum
from arabic_banking_intents.training.registry import candidate_problems, register
from arabic_banking_intents.training.tracking import connect


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a candidate model and set an alias.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--alias", type=AliasEnum, choices=list(AliasEnum), default=AliasEnum.CHAMPION)
    parser.add_argument("--note", default="", help="why this version is registered")
    args = parser.parse_args()

    settings = get_settings()
    connect(settings.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    run = client.get_run(args.run_id)
    problems = candidate_problems(client, run)
    if problems:
        raise SystemExit("; ".join(problems))
    print(register(client, run, settings.REGISTERED_MODEL_NAME, args.alias, args.note))


if __name__ == "__main__":
    main()
