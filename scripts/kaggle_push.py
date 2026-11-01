"""Upload the inputs for AraBERT training to Kaggle and start the training notebook.

    uv run python scripts/kaggle_push.py            # the runs in configs/arabert.toml
    uv run python scripts/kaggle_push.py --smoke    # 1% of the data, 1 epoch: checks the whole loop
    uv run python scripts/kaggle_push.py --dry-run  # only build runs/kaggle_upload, to inspect it

The inputs are one private Kaggle dataset version: our data files, the official train/validation
files with our label IDs, the config, and a run request carrying the git commit, the data versions
and the library versions to install on Kaggle. Refuses uncommitted code, so every Kaggle run traces
back to a commit. Afterwards: uv run python scripts/log_kaggle_runs.py
"""

import argparse
import json
import shutil
import subprocess
import time
from importlib.metadata import version

from arabic_banking_intents.helpers.config import PROJECT_ROOT, get_settings
from arabic_banking_intents.training.data import load_labels, load_official
from arabic_banking_intents.training.tracking import lineage_tags
from arabic_banking_intents.models.enums.TrackingEnum import TagEnum

TRAINING_SCRIPT = PROJECT_ROOT / "kaggle" / "train_arabert.py"
PINNED_PACKAGES = ["transformers", "tokenizers", "safetensors", "accelerate"]  # Kaggle keeps its own CUDA torch
ACCELERATOR = "NvidiaTeslaT4"
READY_TIMEOUT_SECONDS = 900


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{result.stdout}{result.stderr}")
    return result


def dataset_state(dataset: str) -> dict | None:
    """{"status": ..., "current_version_number": ...}, or None if the dataset doesn't exist yet."""
    result = kaggle("datasets", "status", dataset, "--format", "json", check=False)
    return json.loads(result.stdout) if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload the training inputs and start the Kaggle notebook.")
    parser.add_argument("--smoke", action="store_true", help="1%% of the data and 1 epoch, to check the loop")
    parser.add_argument("--dry-run", action="store_true", help="build the upload folder only; nothing is sent")
    args = parser.parse_args()

    settings = get_settings()
    lineage = lineage_tags(PROJECT_ROOT, settings.DATA_RAW_DIR, settings.DATA_PROCESSED_DIR)
    if not args.dry_run:
        if not settings.KAGGLE_USERNAME:
            raise SystemExit("set KAGGLE_USERNAME in .env (see .env.example)")
        if lineage[TagEnum.GIT_DIRTY] != "false":
            raise SystemExit("commit first: every Kaggle run must trace back to a commit")

    # 1. The inputs, as flat files: Kaggle datasets keep folders unreliably
    upload = settings.RUNS_DIR / "kaggle_upload"
    shutil.rmtree(upload, ignore_errors=True)
    upload.mkdir(parents=True)
    for path in settings.DATA_PROCESSED_DIR.iterdir():
        shutil.copy2(path, upload)
    labels = load_labels(settings.DATA_RAW_DIR / "Banking77_intents.csv")
    for split in ("train", "val"):
        official = load_official(settings.DATA_RAW_DIR / f"Banking77_Arabized_MSA_PAL_{split}.csv", labels)
        official.to_parquet(upload / f"official_{split}.parquet", index=False)
    shutil.copy2(settings.CONFIGS_DIR / "arabert.toml", upload)
    request = {
        "smoke": args.smoke,
        "lineage": {str(key): value for key, value in lineage.items()},
        "packages": {package: version(package) for package in PINNED_PACKAGES},
    }
    (upload / "run_request.json").write_text(json.dumps(request, indent=2))
    if args.dry_run:
        print(f"built {upload}: " + ", ".join(sorted(path.name for path in upload.iterdir())))
        return

    # 2. A new version of the private dataset, and wait until Kaggle has processed it
    dataset = f"{settings.KAGGLE_USERNAME}/{settings.KAGGLE_DATASET}"
    (upload / "dataset-metadata.json").write_text(json.dumps(
        {"title": settings.KAGGLE_DATASET, "id": dataset, "licenses": [{"name": "CC-BY-SA-4.0"}]}, indent=2))
    before = dataset_state(dataset)
    commit, data_version = lineage[TagEnum.GIT_COMMIT], lineage[TagEnum.DATA_VERSION]
    note = f"commit {commit[:7]}, data {data_version[:8]}" + (", smoke" if args.smoke else "")
    if before is None:  # -t keeps parquet as parquet; Kaggle converts tables to CSV by default
        kaggle("datasets", "create", "-p", str(upload), "-t")
    else:
        kaggle("datasets", "version", "-p", str(upload), "-m", note, "-t")
    previous_version = before["current_version_number"] if before else 0
    deadline = time.time() + READY_TIMEOUT_SECONDS
    while True:
        state = dataset_state(dataset)
        if state and state["status"].lower() == "ready" and state["current_version_number"] > previous_version:
            break
        if time.time() > deadline:
            raise SystemExit(f"dataset {dataset} not ready after {READY_TIMEOUT_SECONDS // 60} min: {state}")
        time.sleep(15)
    print(f"dataset {dataset} version {state['current_version_number']} ready ({note})")

    # 3. The notebook: the training script, on one T4, with internet for the model and pinned packages
    kernel_dir = settings.RUNS_DIR / "kaggle_kernel"
    shutil.rmtree(kernel_dir, ignore_errors=True)
    kernel_dir.mkdir(parents=True)
    shutil.copy2(TRAINING_SCRIPT, kernel_dir)
    kernel = f"{settings.KAGGLE_USERNAME}/{settings.KAGGLE_KERNEL}"
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel,
        "title": settings.KAGGLE_KERNEL,
        "code_file": TRAINING_SCRIPT.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": ACCELERATOR,
        "dataset_sources": [dataset],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=2))
    print(kaggle("kernels", "push", "-p", str(kernel_dir), "--accelerator", ACCELERATOR).stdout.strip())

    pending = {"kernel": kernel, "git_commit": commit, "smoke": args.smoke, "pushed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (settings.RUNS_DIR / "pending.json").write_text(json.dumps(pending, indent=2))
    print(f"started {kernel}; when it finishes: uv run python scripts/log_kaggle_runs.py")


if __name__ == "__main__":
    main()
