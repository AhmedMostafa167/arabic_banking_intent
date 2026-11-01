"""Fine-tune AraBERT for every run in the config, and save files for the laptop to log.

Runs on a Kaggle GPU: scripts/kaggle_push.py uploads the inputs as a Kaggle dataset (data files,
config, and a run request with the git commit) and starts this script. Nothing here talks to MLflow
or computes the reported numbers: scripts/log_kaggle_runs.py does both on the laptop, from the
predictions saved here, with the same evaluation code as the baseline.

Locally, as a quick check with a tiny model:
    uv run python kaggle/train_arabert.py --inputs <dir> --output <dir> --model hf-internal-testing/tiny-random-bert
"""

import os

# Kaggle's T4 machine has two GPUs, and the Trainer would split every batch across both: 2 x 64 = 128
# per step instead of the paper's 64. One GPU keeps the batch as configured. Set before torch loads.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_OUTPUT = Path("/kaggle/working")  # files left here become the notebook's output
TEST_SETS = ["msa", "pal", "saudi", "moroccan", "tunisian"]
TRAIN_FILES = {"grouped": ("train.parquet", "val.parquet"), "official": ("official_train.parquet", "official_val.parquet")}
MODEL_INFO_FILE = "intent_model.json"  # read by arabic_banking_intents.ml.arabert_model


# ---------------------------------------------------------------- inputs

def find_inputs(root: Path) -> Path:
    """The folder holding run_request.json; Kaggle mounts the dataset somewhere under /kaggle/input."""
    found = list(root.rglob("run_request.json"))
    assert len(found) == 1, f"expected one run_request.json under {root}, found {len(found)}"
    return found[0].parent


def pin_packages(packages: dict[str, str]) -> None:
    """Install the laptop's library versions, so the model trained here loads identically there."""
    pins = [f"{name}=={pinned}" for name, pinned in packages.items() if version(name) != pinned]
    if pins:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *pins], check=True)


def select_runs(config: dict, smoke: bool) -> list[dict]:
    """All configured runs, or for a smoke test the first run of each variant on a small slice."""
    if not smoke:
        return config["runs"]
    config["max_epochs"] = config["smoke"]["max_epochs"]
    first_of_each = {}
    for run in config["runs"]:
        first_of_each.setdefault(run["variant"], run)
    return list(first_of_each.values())


# ---------------------------------------------------------------- one training run

def encode(tokenizer, frame: pd.DataFrame, max_length: int) -> list[dict]:
    """Token IDs per message, without padding: the collator pads each batch to its longest message."""
    encoded = tokenizer(frame["text"].tolist(), truncation=True, max_length=max_length)
    return [
        {"input_ids": ids, "attention_mask": mask, "labels": int(label)}
        for ids, mask, label in zip(encoded["input_ids"], encoded["attention_mask"], frame["label"])
    ]


def macro_f1(eval_prediction) -> dict:
    """Validation macro-F1 for picking the best epoch. Same definition as
    arabic_banking_intents.ml.evaluate.macro_f1: averaged over the intents present."""
    logits, labels = eval_prediction
    predicted = logits.argmax(axis=-1)
    return {"macro_f1": f1_score(labels, predicted, labels=np.unique(labels), average="macro", zero_division=0)}


def train_run(run: dict, config: dict, inputs: Path, output_root: Path, request: dict) -> None:
    from transformers import (  # imported after pin_packages, so the pinned versions load
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    name = f"{run['variant']}-seed{run['seed']}"
    out = output_root / "runs" / name
    out.mkdir(parents=True)
    set_seed(run["seed"])  # Python, NumPy and PyTorch: the classifier layer's start and the batch order

    # Data: the train/validation pair for this variant, the five official test sets, the label names
    train_file, val_file = TRAIN_FILES[run["variant"]]
    train, val = pd.read_parquet(inputs / train_file), pd.read_parquet(inputs / val_file)
    if request["smoke"]:
        train = train.sample(frac=config["smoke"]["train_fraction"], random_state=run["seed"])
    tests = {t: pd.read_parquet(inputs / f"test_{t}.parquet") for t in TEST_SETS}
    labels = json.loads((inputs / "labels.json").read_text(encoding="utf-8"))
    intent_names = [entry["intent"] for entry in sorted(labels, key=lambda entry: entry["label"])]

    # Model: the pretrained encoder plus a new 77-way classification layer on top
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=len(intent_names),
        id2label=dict(enumerate(intent_names)),
        label2id={intent: i for i, intent in enumerate(intent_names)},
    )

    max_length = config["max_length"]
    train_lengths = [len(ids) for ids in tokenizer(train["text"].tolist())["input_ids"]]
    truncated_share = float(np.mean(np.array(train_lengths) > max_length))
    train_set, val_set = encode(tokenizer, train, max_length), encode(tokenizer, val, max_length)

    checkpoints = Path(tempfile.mkdtemp())  # outside /kaggle/working, so checkpoints are not outputs
    args = TrainingArguments(
        output_dir=str(checkpoints),
        num_train_epochs=config["max_epochs"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"] * 2,
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"],
        fp16=config["fp16"] and torch.cuda.is_available(),
        eval_strategy="epoch",  # score validation after every epoch...
        save_strategy="epoch",  # ...save a checkpoint after every epoch...
        load_best_model_at_end=True,  # ...and finish with the epoch that scored best
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,  # the best and the latest checkpoint
        save_only_model=True,  # no optimizer state: smaller checkpoints
        logging_steps=50,
        report_to="none",
        seed=run["seed"],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
        compute_metrics=macro_f1,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"])],
    )

    start = time.perf_counter()
    trainer.train()
    training_seconds = time.perf_counter() - start

    # Predictions with the best epoch's weights: probabilities for validation and every test set
    (out / "predictions").mkdir()
    for split, frame in {"val": val, **{f"test_{t}": tests[t] for t in TEST_SETS}}.items():
        logits = trainer.predict(encode(tokenizer, frame, max_length)).predictions
        probabilities = torch.softmax(torch.from_numpy(logits).float(), dim=-1).numpy()
        np.savez_compressed(out / "predictions" / f"{split}.npz",
                            probabilities=probabilities.astype(np.float16), labels=frame["label"].to_numpy())

    if run["save_model"]:
        model_dir = out / "model"
        if torch.cuda.is_available():
            trainer.model.to(torch.float16)  # halves the download; the laptop loads it back as fp32
        trainer.model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        (model_dir / MODEL_INFO_FILE).write_text(json.dumps(
            {"framework": "transformers", "base_model": config["model_name"], "max_length": max_length}, indent=2))

    evaluations = [entry for entry in trainer.state.log_history if "eval_macro_f1" in entry]
    best = max(evaluations, key=lambda entry: entry["eval_macro_f1"])
    run_info = {
        "name": name,
        **run,
        "smoke": request["smoke"],
        "lineage": request["lineage"],
        "config": {key: value for key, value in config.items() if key not in ("runs", "smoke")},
        "train_rows": len(train),
        "val_rows": len(val),
        "truncated_share": truncated_share,
        "best_epoch": best["epoch"],
        "epochs_run": evaluations[-1]["epoch"],
        "training_seconds": training_seconds,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "packages": {package: version(package) for package in ("torch", "transformers", "tokenizers")},
    }
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2))
    (out / "history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    shutil.rmtree(checkpoints)
    print(f"{name}: best epoch {best['epoch']:.0f}, validation macro-F1 {best['eval_macro_f1']:.4f}, "
          f"{training_seconds / 60:.1f} min", flush=True)


# ---------------------------------------------------------------- entry point

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune AraBERT for every configured run.")
    parser.add_argument("--inputs", type=Path, help="folder with run_request.json (default: found under /kaggle/input)")
    parser.add_argument("--output", type=Path, default=KAGGLE_OUTPUT)
    parser.add_argument("--model", help="override the configured model, e.g. a tiny one for a local check")
    args = parser.parse_args()

    inputs = args.inputs or find_inputs(KAGGLE_INPUT)
    request = json.loads((inputs / "run_request.json").read_text())
    if KAGGLE_INPUT.exists():
        pin_packages(request["packages"])
    config = tomllib.loads((inputs / "arabert.toml").read_text())
    if args.model:
        config["model_name"] = args.model

    for run in select_runs(config, request["smoke"]):
        train_run(run, config, inputs, args.output, request)


if __name__ == "__main__":
    main()
