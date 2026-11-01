"""Load-test the running API and log every load level to MLflow (experiment serving-benchmarks).

    uv run python benchmarks/run_benchmark.py --rung "0 pytorch fp32"     # 1, 2, 4, 8 users; 90 s; 3 repeats
    uv run python benchmarks/run_benchmark.py --rung smoke --users 1 --seconds 20 --repeats 1

Needs the stack from docker-compose.yml. First the cold start: the API container is restarted and
timed until it is ready. Then, for each repeat and user count, Locust runs headless while
`docker stats` samples the API container, and each (repeat, users) becomes one MLflow run with the
latency percentiles, throughput, errors, CPU and memory, tagged with the version being served.
Run it on a quiet machine: anything else using the CPU shows up in the numbers.
"""

import argparse
import csv
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import httpx
import mlflow
import numpy as np

from arabic_banking_intents.helpers.config import PROJECT_ROOT, get_settings
from arabic_banking_intents.training.tracking import connect, lineage_tags
from arabic_banking_intents.models.enums.TrackingEnum import ExperimentEnum, TagEnum

LOCUSTFILE = Path(__file__).with_name("locustfile.py")
READY_TIMEOUT_SECONDS = 600
WARM_UP_REQUESTS = 10  # the first requests after a start are slower; keep them out of the numbers
WARM_UP_TEXT = "أريد تفعيل بطاقتي"
STATS_INTERVAL_SECONDS = 2
PAUSE_SECONDS = 5  # between load levels, so one level's queue does not spill into the next


def run(*args: str) -> str:
    return subprocess.run(list(args), cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def to_mb(size: str) -> float:
    """A docker size such as '1.18GiB' or '390.4MiB', in MB."""
    units = {"KiB": 1024 / 1e6, "MiB": 1024**2 / 1e6, "GiB": 1024**3 / 1e6, "kB": 1e-3, "MB": 1.0, "GB": 1e3, "B": 1e-6}
    for unit, factor in units.items():
        if size.endswith(unit):
            return float(size[: -len(unit)]) * factor
    raise ValueError(f"unknown size: {size}")


def container_memory_mb(container: str) -> float:
    return to_mb(run("docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container).split("/")[0].strip())


class StatsSampler(threading.Thread):
    """Samples the API container's CPU (100% = one core) and memory while a load test runs."""

    def __init__(self, container: str):
        super().__init__(daemon=True)
        self.container = container
        self.cpu_pct: list[float] = []
        self.memory_mb: list[float] = []
        self.stopped = threading.Event()

    def run(self) -> None:
        while not self.stopped.is_set():
            cpu, memory = run("docker", "stats", "--no-stream", "--format", "{{.CPUPerc}};{{.MemUsage}}",
                              self.container).split(";")
            self.cpu_pct.append(float(cpu.rstrip("%")))
            self.memory_mb.append(to_mb(memory.split("/")[0].strip()))
            self.stopped.wait(STATS_INTERVAL_SECONDS)


def wait_ready(url: str) -> None:
    deadline = time.time() + READY_TIMEOUT_SECONDS
    while time.time() < deadline:
        try:
            if httpx.get(f"{url}/api/v1/ready", timeout=5).status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(1)
    raise SystemExit(f"the API was not ready {READY_TIMEOUT_SECONDS} s after a restart")


def cold_start_seconds(url: str) -> float:
    """Restart the API container; seconds from its start to ready (model downloaded, loaded, checked)."""
    run("docker", "compose", "restart", "api")
    start = time.perf_counter()
    wait_ready(url)
    return time.perf_counter() - start


def load_test(url: str, users: int, seconds: int, out: Path) -> dict[str, float]:
    """One headless Locust run; its aggregated latency percentiles, throughput and failures."""
    prefix = out / f"u{users}"
    run("locust", "-f", str(LOCUSTFILE), "--headless", "--host", url, "--users", str(users),
        "--spawn-rate", str(users), "--run-time", f"{seconds}s", "--csv", str(prefix), "--only-summary",
        "--exit-code-on-error", "0")  # failures are a result to record, not a crash
    with open(f"{prefix}_stats.csv") as file:
        row = next(r for r in csv.DictReader(file) if r["Name"] == "Aggregated")
    requests, failures = int(row["Request Count"]), int(row["Failure Count"])
    return {
        "requests": requests,
        "failure_rate": failures / requests if requests else 1.0,
        "requests_per_s": float(row["Requests/s"]),
        "p50_ms": float(row["50%"]),
        "p95_ms": float(row["95%"]),
        "p99_ms": float(row["99%"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Load-test the running API and log the results to MLflow.")
    parser.add_argument("--rung", required=True, help='what is being measured, e.g. "0 pytorch fp32"')
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--users", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seconds", type=int, default=90, help="per load level")
    parser.add_argument("--repeats", type=int, default=3, help="to see the noise between identical runs")
    args = parser.parse_args()

    settings = get_settings()
    connect(settings.MLFLOW_TRACKING_URI)
    lineage = lineage_tags(PROJECT_ROOT, settings.DATA_RAW_DIR, settings.DATA_PROCESSED_DIR)
    container = run("docker", "compose", "ps", "-q", "api")
    if not container:
        raise SystemExit("the api container is not running: docker compose up -d")
    image_mb = int(run("docker", "image", "inspect", run("docker", "inspect", "--format", "{{.Image}}", container),
                       "--format", "{{.Size}}")) / 1e6
    api_cpus = int(run("docker", "inspect", "--format", "{{.HostConfig.NanoCpus}}", container)) / 1e9

    cold_start = cold_start_seconds(args.url)
    idle_mb = container_memory_mb(container)
    info = httpx.get(f"{args.url}/api/v1/info").json()
    for _ in range(WARM_UP_REQUESTS):
        httpx.post(f"{args.url}/api/v1/predict", json={"text": WARM_UP_TEXT}, timeout=60).raise_for_status()

    mlflow.set_experiment(ExperimentEnum.SERVING_BENCHMARKS)
    tags = {**{str(key): value for key, value in lineage.items()}, TagEnum.RUNG: args.rung,
            TagEnum.MODEL_VERSION: info["version"], TagEnum.MODEL_FAMILY: info["model_family"]}
    with mlflow.start_run(run_name=f"{args.rung} | cold start", tags=tags):
        mlflow.log_params({"api_cpus": api_cpus})
        mlflow.log_metrics({"cold_start_s": cold_start, "image_size_mb": image_mb, "idle_memory_mb": idle_mb})
    print(f"v{info['version']} ({info['model_family']}): cold start {cold_start:.0f} s, image {image_mb:.0f} MB, "
          f"idle {idle_mb:.0f} MB, {api_cpus:g} CPUs")

    results: dict[int, list[dict[str, float]]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for repeat in range(1, args.repeats + 1):
            for users in args.users:  # every level once per repeat, so slow drifts spread over all levels
                sampler = StatsSampler(container)
                sampler.start()
                metrics = load_test(args.url, users, args.seconds, Path(tmp))
                sampler.stopped.set()
                sampler.join()
                metrics |= {"cpu_mean_pct": float(np.mean(sampler.cpu_pct)), "cpu_peak_pct": float(np.max(sampler.cpu_pct)),
                            "memory_peak_mb": float(np.max(sampler.memory_mb))}
                with mlflow.start_run(run_name=f"{args.rung} | {users} users | repeat {repeat}", tags=tags):
                    mlflow.log_params({"users": users, "seconds": args.seconds, "repeat": repeat, "wait_between_s": "1-2",
                                       "api_cpus": api_cpus, "api_workers": 1})
                    mlflow.log_metrics(metrics)
                results.setdefault(users, []).append(metrics)
                print(f"repeat {repeat}, {users} users: p50 {metrics['p50_ms']:.0f} ms, p95 {metrics['p95_ms']:.0f} ms, "
                      f"p99 {metrics['p99_ms']:.0f} ms, {metrics['requests_per_s']:.2f} req/s, "
                      f"failures {metrics['failure_rate']:.1%}, CPU {metrics['cpu_mean_pct']:.0f}% mean", flush=True)
                time.sleep(PAUSE_SECONDS)

    print("\nmedian over repeats (p95 range across repeats):")
    for users, runs in results.items():
        p95 = [m["p95_ms"] for m in runs]
        print(f"  {users} users: p50 {np.median([m['p50_ms'] for m in runs]):.0f} ms, p95 {np.median(p95):.0f} ms "
              f"({min(p95):.0f}-{max(p95):.0f}), p99 {np.median([m['p99_ms'] for m in runs]):.0f} ms, "
              f"{np.median([m['requests_per_s'] for m in runs]):.2f} req/s, CPU {np.median([m['cpu_mean_pct'] for m in runs]):.0f}%")


if __name__ == "__main__":
    main()
