"""One simulated customer: sends a real message from one of the five test sets, then waits 1-2 s.

Started by benchmarks/run_benchmark.py; to watch it in the browser instead:
    uv run locust -f benchmarks/locustfile.py --host http://127.0.0.1:8000
"""

import os
import random

import pandas as pd
from locust import HttpUser, between, task

from arabic_banking_intents.helpers.config import get_settings

# Real messages: one text sent over and over would measure a cache, not the model. All five test sets by
# default; TEST_SETS=tunisian sends only one dialect, to see how monitoring reacts to unfamiliar traffic
TEST_SETS = os.environ.get("TEST_SETS", "msa,pal,saudi,moroccan,tunisian").split(",")
TEXTS = pd.concat(
    pd.read_parquet(get_settings().DATA_PROCESSED_DIR / f"test_{name}.parquet", columns=["text"]) for name in TEST_SETS
)["text"].tolist()


class Customer(HttpUser):
    wait_time = between(1, 2)  # a person reading the answer, not a loop flooding the server

    @task
    def predict(self) -> None:
        self.client.post("/api/v1/predict", json={"text": random.choice(TEXTS)}, name="/api/v1/predict")
