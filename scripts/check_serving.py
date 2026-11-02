"""Check that the running API answers like its model did offline.

    uv run python scripts/check_serving.py [--url http://127.0.0.1:8000] [--texts 300]

Sends validation texts over HTTP and compares each top intent with the prediction the served
version's training run recorded for the same text. Anything short of full agreement means serving
and training disagree somewhere between the request and the model (input handling, the wrong
version, a broken load), even though every response looks plausible.
"""

import argparse
import time

import httpx
import numpy as np

from arabic_banking_intents.helpers.config import get_settings
from arabic_banking_intents.models.enums.DataEnum import DataVariantEnum
from arabic_banking_intents.training.baseline import TEXT_COLUMN
from arabic_banking_intents.training.data import load_label_names, load_train_val
from arabic_banking_intents.training.registry import val_predictions
from arabic_banking_intents.training.tracking import connect


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the API's answers with the model's offline predictions.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--texts", type=int, default=300, help="validation texts to send")
    args = parser.parse_args()

    settings = get_settings()
    connect(settings.MLFLOW_TRACKING_URI)
    with httpx.Client(base_url=args.url, timeout=60) as client:
        response = client.get("/api/v1/info")
        response.raise_for_status()
        info = response.json()

        _, val = load_train_val(DataVariantEnum.GROUPED, settings.DATA_RAW_DIR, settings.DATA_PROCESSED_DIR)
        saved = val_predictions(info["run_id"])
        assert np.array_equal(saved["label"].to_numpy(), val["label"].to_numpy()), "validation rows differ"
        names = load_label_names(settings.DATA_PROCESSED_DIR)
        sample = val.assign(offline=saved["predicted"].to_numpy()).sample(n=args.texts, random_state=0)

        agree, latencies = 0, []
        for text, offline in zip(sample[TEXT_COLUMN], sample["offline"]):
            start = time.perf_counter()
            response = client.post("/api/v1/predict", json={"text": text})
            latencies.append((time.perf_counter() - start) * 1000)
            response.raise_for_status()
            agree += response.json()["intents"][0]["intent"] == names[offline]

    p50, p95 = np.percentile(latencies, [50, 95])
    print(f"v{info['version']} ({info['model_family']}): {agree}/{len(sample)} top intents match the run's "
          f"validation predictions; one request at a time over HTTP: p50 {p50:.0f} ms, p95 {p95:.0f} ms")
    if agree != len(sample):
        raise SystemExit("serving disagrees with the model's offline predictions")


if __name__ == "__main__":
    main()
