# Arabic banking intent classifier

Classifies Arabic banking customer messages into 77 intents (ArBanking77), with a production-standard
system around the model.

**Status:** step 1 (data) done; step 2 (baseline and experiment tracking) in progress. See
[docs/PLAN.md](docs/PLAN.md).

Data: [ArBanking77](https://github.com/SinaLab/ArBanking77) by SinaLab, Birzeit University (CC BY-SA 4.0).

## Running

```bash
uv sync                                     # install the environment
uv run python scripts/prepare_data.py       # rebuild data/processed/ from data/raw/
uv run pytest                               # run the tests

# MLflow tracking server and model registry: SQLite + local files, at http://127.0.0.1:5000
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts \
    --host 127.0.0.1 --port 5000 --workers 1

# The TF-IDF baseline (settings in configs/baseline.toml); commit first, so the run records clean code
uv run python scripts/train_baseline.py --data grouped --experiment intent-classifier
```

Settings such as the tracking URI and data paths live in `src/arabic_banking_intents/helpers/config.py`;
copy `.env.example` to `.env` to change them.
