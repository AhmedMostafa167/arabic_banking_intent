# Arabic banking intent classifier

Classifies Arabic banking customer messages into 77 intents (ArBanking77), with a production-standard
system around the model: versioned data, tracked runs, a model registry with promotion gates, a served
and load-tested API, and the measured trade-offs of every optimisation.

**Status:** complete. The model is fine-tuned AraBERT, served as ONNX fp16 by a FastAPI container
without PyTorch and monitored with Prometheus and Grafana. It is published on the Hugging Face Hub as
[Ahmed167/arabert-arbanking77-intents](https://huggingface.co/Ahmed167/arabert-arbanking77-intents) (model card:
[docs/MODEL_CARD.md](docs/MODEL_CARD.md)); the data card is [docs/DATA_CARD.md](docs/DATA_CARD.md); every
measured change, with the prediction written before it, is in [eval/IMPROVEMENT_LOG.md](eval/IMPROVEMENT_LOG.md).

## Results

Macro-F1 on the official test sets and on a validation split that, unlike the official one, shares no
source question with training (the official split inflates scores by about 3 points):

| | Validation | MSA | Palestinian | Saudi | Moroccan | Tunisian |
|---|---|---|---|---|---|---|
| TF-IDF + logistic regression | 0.887 | 0.881 | 0.852 | 0.719 | 0.612 | 0.499 |
| Fine-tuned AraBERT (served) | **0.919** | **0.919** | **0.899** | **0.843** | **0.779** | **0.599** |

Training data covers MSA and Palestinian only; Saudi, Moroccan and Tunisian are unseen dialects.

| Serving, on a 2014 laptop CPU in a 3-CPU container | PyTorch | ONNX fp16 |
|---|---|---|
| Container image | 2.26 GB | 0.95 GB |
| Memory when idle | 1.23 GB | 0.86 GB |
| Median latency, 4 concurrent users | 670 ms | 530 ms |

What was learned along the way, briefly:
- The official validation split leaks paraphrases, so a grouped split was built.
- AraBERT beat the baseline beyond the noise for every seed (a paired bootstrap gate), with the
  largest gains on unseen dialects.
- int8 quantisation did not pay off on a CPU without VNNI.
- In a container, ONNX Runtime's default thread count made latency collapse.
- The share of low-confidence answers flags unfamiliar traffic without labels: 1.5% on MSA, 25% on
  Tunisian, which raised the drift alert.

Details: [eval/IMPROVEMENT_LOG.md](eval/IMPROVEMENT_LOG.md); bugs and surprises:
[docs/log.md](docs/log.md).

## How it fits together

```
data/raw ──prepare_data.py──► data/processed (DVC) ──► training: TF-IDF on the laptop, AraBERT on Kaggle
                                                              │ runs, metrics, lineage tags
                                                              ▼
            MLflow tracking + registry (aliases: @champion) ◄── promotion gates (paired bootstrap)
                                                              │ loaded at startup by alias
                                                              ▼
                       FastAPI container (/api/v1/predict, /metrics) ◄── Locust load tests
                                                              │ scraped every 15 s
                                                              ▼
                       Prometheus (alert rules) ──► Grafana: model dashboard, system dashboard
```

Data: [ArBanking77](https://github.com/SinaLab/ArBanking77) by SinaLab, Birzeit University (CC BY-SA 4.0).
A related public study of ArBanking77's dialects:
[AmrKhalil9/arabic-banking-intent-dialects](https://github.com/AmrKhalil9/arabic-banking-intent-dialects).

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

## Monitoring

`docker compose up` also starts Prometheus (localhost:9090; alerts at `/alerts`) and Grafana
(localhost:3000, folder "Intent API"). Both dashboards are provisioned from `monitoring/` at every start.

- **Model dashboard:** requests per second by version and status, inference p50/p95/p99, requests in
  flight, the share of low-confidence answers, the most predicted intents, errors.
- **System dashboard:** the machine's CPU, memory, swap, disk and network (node-exporter), and the
  API process's CPU and memory.
- **Alerts** (`monitoring/prometheus/alerts.yml`):
  - `ApiDown`: no scrape for 3 minutes, longer than a cold start;
  - `HighErrorRate`: over 5% of requests failing for 5 minutes;
  - `LowConfidenceShareHigh`: over 10% of answers with a top probability under 0.5, for 2 minutes.

```bash
# See the drift alert fire: traffic in a dialect the model never saw (about 10 minutes)
TEST_SETS=tunisian uv run locust -f benchmarks/locustfile.py --headless --host http://127.0.0.1:8000 -u 4 -r 4 -t 10m
```

Settings such as the tracking URI and data paths live in `src/arabic_banking_intents/helpers/config.py`;
copy `.env.example` to `.env` to change them.
