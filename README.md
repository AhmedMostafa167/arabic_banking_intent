# Arabic banking intent classifier

Classifies Arabic banking customer messages into 77 intents (ArBanking77), with a production-standard
system around the model.

**Status:** data pipeline, TF-IDF baseline and AraBERT fine-tuning done. Fine-tuned AraBERT is the
registered champion (validation macro-F1 0.918 against the baseline's 0.887). Next: serving it.
Measured results are in [eval/IMPROVEMENT_LOG.md](eval/IMPROVEMENT_LOG.md).

Data: [ArBanking77](https://github.com/SinaLab/ArBanking77) by SinaLab, Birzeit University (CC BY-SA 4.0).

## Running

```bash
uv sync                                     # install the environment
uv run python scripts/prepare_data.py       # rebuild data/processed/ from data/raw/
uv run pytest                               # run the tests

# MLflow tracking server and model registry: SQLite + local files, at http://127.0.0.1:5000
# (job execution off: MLflow 3's GenAI background workers, unused here, take about 1.2 GB)
MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false uv run mlflow server --backend-store-uri sqlite:///mlflow.db \
    --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000 --workers 1

# The TF-IDF baseline (settings in configs/baseline.toml); commit first, so the run records clean code
uv run python scripts/train_baseline.py --data grouped --experiment intent-classifier

# AraBERT on a Kaggle GPU (settings in configs/arabert.toml; needs the Kaggle CLI and KAGGLE_USERNAME)
uv run python scripts/kaggle_push.py            # upload the inputs and start the training notebook
uv run python scripts/log_kaggle_runs.py        # wait for it, download the output, log every run
uv run python scripts/promote.py --runs <seed run IDs>   # the promotion gate against @champion

# Log a registered version again, unchanged, the way models are logged now (checked against its run)
uv run python scripts/relog_model.py --version <n> [--alias champion] --note "why"
```

## Serving

The API loads the model the `champion` alias points to, once, at startup; the model is not in the image.

```bash
docker compose up --build          # MLflow + the API; stop any MLflow server on the host first
curl -s localhost:8000/api/v1/info # which model version is serving
curl -s localhost:8000/api/v1/predict -H 'content-type: application/json' -d '{"text": "وين بطاقتي الجديدة؟"}'
uv run python scripts/check_serving.py   # the API must answer like the model did offline
uv run python benchmarks/run_benchmark.py --rung "0 pytorch fp32"   # load test, logged to MLflow

# The optimisation ladder: ONNX variants of an AraBERT version, each checked and timed
uv run python scripts/build_variants.py --from-version 5
uv run python scripts/build_variants.py --from-version 5 --register <rung> --alias champion --note "why"

# Or the API alone, on the host, against a host MLflow server
uv run uvicorn arabic_banking_intents.serving.main:app
```

Endpoints: `POST /api/v1/predict` (top 3 intents with probabilities), `GET /api/v1/health` (liveness),
`/api/v1/ready` (readiness: model loaded), `/api/v1/info` (the version serving), `/metrics` (Prometheus).
To serve another version, move the alias and restart the `api` container; nothing is rebuilt.

Settings such as the tracking URI and data paths live in `src/arabic_banking_intents/helpers/config.py`;
copy `.env.example` to `.env` to change them.
