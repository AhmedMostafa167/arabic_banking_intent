"""Publish the model to the Hugging Face Hub, from the registry, with the model card.

    hf auth login                       # once; the token stays in ~/.cache/huggingface, never in this repo
    uv run python scripts/publish_model.py --repo-id <user>/<name> --dry-run   # build the folder, upload nothing
    uv run python scripts/publish_model.py --repo-id <user>/<name> [--private]

The repository gets the PyTorch weights of --pytorch-version at its root (loadable with
transformers), the ONNX graph of --onnx-version in onnx/, and docs/MODEL_CARD.md as its README. Both
versions must hold the same model; the card states the measurements behind that.
"""

import argparse
import shutil

from huggingface_hub import HfApi
from mlflow import MlflowClient

from arabic_banking_intents.helpers.config import PROJECT_ROOT, get_settings
from arabic_banking_intents.training.registry import download_model_folder
from arabic_banking_intents.training.tracking import connect
from arabic_banking_intents.models.enums.TrackingEnum import TagEnum

PYTORCH_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
ONNX_FILES = ["model.onnx", "tokenizer.json", "tokenizer_config.json", "intent_model.json"]
MODEL_CARD = PROJECT_ROOT / "docs" / "MODEL_CARD.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the registered model to the Hugging Face Hub.")
    parser.add_argument("--repo-id", required=True, help="<user>/<name> on the Hub")
    parser.add_argument("--pytorch-version", default="5")
    parser.add_argument("--onnx-version", default="7")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="build runs/hub/<name> and stop")
    args = parser.parse_args()

    settings = get_settings()
    connect(settings.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    pytorch = client.get_model_version(settings.REGISTERED_MODEL_NAME, args.pytorch_version)
    onnx = client.get_model_version(settings.REGISTERED_MODEL_NAME, args.onnx_version)
    if onnx.tags.get(TagEnum.DERIVED_FROM) != f"v{args.pytorch_version}":
        raise SystemExit(f"v{args.onnx_version} is not derived from v{args.pytorch_version}; the card would be wrong")

    staging = settings.RUNS_DIR / "hub" / args.repo_id.split("/")[-1]
    shutil.rmtree(staging, ignore_errors=True)
    (staging / "onnx").mkdir(parents=True)
    cache = settings.RUNS_DIR / "hub" / "cache"
    pytorch_dir = download_model_folder(pytorch.source, cache / f"v{args.pytorch_version}")
    onnx_dir = download_model_folder(onnx.source, cache / f"v{args.onnx_version}")
    for name in PYTORCH_FILES:
        shutil.copy2(pytorch_dir / name, staging / name)
    for name in ONNX_FILES:
        shutil.copy2(onnx_dir / name, staging / "onnx" / name)
    (staging / "README.md").write_text(MODEL_CARD.read_text().replace("<this repository>", args.repo_id))
    sizes = {path.relative_to(staging).as_posix(): f"{path.stat().st_size / 1e6:.0f} MB"
             for path in sorted(staging.rglob("*")) if path.is_file()}
    print(f"built {staging}: {sizes}")
    if args.dry_run:
        return

    api = HfApi()
    api.create_repo(args.repo_id, private=args.private, exist_ok=True)
    commit = pytorch.tags.get(TagEnum.GIT_COMMIT, "")[:7]
    api.upload_folder(folder_path=str(staging), repo_id=args.repo_id,
                      commit_message=f"v{args.pytorch_version} (PyTorch) and v{args.onnx_version} (ONNX fp16) "
                                     f"from the registry; training commit {commit}")
    print(f"published https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
