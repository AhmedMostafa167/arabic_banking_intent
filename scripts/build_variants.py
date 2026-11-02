"""Build the ONNX variants of a registered AraBERT version and measure each: the optimisation ladder.

    uv run python scripts/build_variants.py --from-version 5                          # every rung
    uv run python scripts/build_variants.py --from-version 5 --rungs onnx-int8-static # only some
    uv run python scripts/build_variants.py --from-version 5 --register onnx-int8-dynamic --alias champion --note "why"

The variants go to runs/variants/<rung>/ (an existing one is reused). Each is measured the same
way as the PyTorch original next to it: model size, latency per message (200 validation texts, one
at a time) and quality on every validation text. The fp32 rungs must give the source's labels; the
reduced-precision rungs must pass the non-inferiority gate: the paired 95% interval of
(variant - source) validation macro-F1 above -0.5 points. Then a thread sweep on the optimised graph.
One MLflow run per rung in the `optimisation` experiment.

With --register, a rung that passed is logged as a model in a run of intent-classifier, which keeps
its evidence, and registered. Commit first.
"""

import argparse
import gc
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlflow
import numpy as np
from mlflow import MlflowClient

from arabic_banking_intents.helpers.config import PROJECT_ROOT, get_settings
from arabic_banking_intents.training import onnx_model, variants
from arabic_banking_intents.training.arabert_model import AraBertClassifier
from arabic_banking_intents.training.baseline import TEXT_COLUMN
from arabic_banking_intents.training.data import load_label_names, load_train_val
from arabic_banking_intents.training.evaluate import interval, latency_ms, macro_f1, paired_bootstrap
from arabic_banking_intents.training.onnx_model import OnnxClassifier
from arabic_banking_intents.training.registry import download_model_folder, register, val_predictions
from arabic_banking_intents.training.tracking import connect, lineage_tags, log_evaluation, log_val_predictions
from arabic_banking_intents.models.enums.DataEnum import DataVariantEnum
from arabic_banking_intents.models.enums.TrackingEnum import (AliasEnum, ExperimentEnum, ModelFamilyEnum,
                                                              ModelMetadataEnum, TagEnum)

GATE_MARGIN_POINTS = 0.5  # agreed 2026-11-01: an optimised copy may lose at most this much macro-F1
SAME_MODEL_AGREEMENT = 0.999  # an fp32 conversion may flip a near-tie or two, not more
LATENCY_SAMPLE = 200
SWEEP_SAMPLE = 100
CALIBRATION_PER_GROUP = 2  # train texts per (intent, dialect): 77 x 2 x 2 = 308 for int8-static
CALIBRATION_BATCH = 16
REFERENCE = "pytorch-fp32"
RUNGS = ["onnx-fp32", "onnx-optimised", "onnx-fp16", "onnx-int8-dynamic", "onnx-int8-static", "onnx-int8-selective"]
SAME_MODEL_RUNGS = {"onnx-fp32", "onnx-optimised"}


def folder_mb(path: Path) -> float:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) / 1e6


def build(rung: str, root: Path, source: Path, calibration_texts: list[str]) -> None:
    out = root / rung
    if (out / "model.onnx").exists():
        return
    if rung == "onnx-fp32":
        variants.export_fp32(source, out)
    elif rung == "onnx-optimised":
        print("  fused:", variants.optimise(root / "onnx-fp32", out), flush=True)
    elif rung == "onnx-fp16":
        variants.to_fp16(root / "onnx-fp32", out)
    elif rung == "onnx-int8-dynamic":
        variants.int8_dynamic(root / "onnx-fp32", out)
    elif rung == "onnx-int8-selective":
        variants.int8_selective(root / "onnx-fp32", out)
    elif rung == "onnx-int8-static":
        reader = OnnxClassifier(root / "onnx-fp32")
        batches = [reader.feed(calibration_texts[i:i + CALIBRATION_BATCH])
                   for i in range(0, len(calibration_texts), CALIBRATION_BATCH)]
        variants.int8_static(root / "onnx-fp32", out, batches)


def measure(rung: str, predict_proba, size_mb: float, val, saved: np.ndarray) -> dict:
    """Size, latency per message, and quality against the source version's validation predictions."""
    texts = val[TEXT_COLUMN].tolist()
    predicted = predict_proba(texts).argmax(axis=1)
    p50, p95 = latency_ms(lambda text: predict_proba([text]), texts[:LATENCY_SAMPLE])
    agreement = float(np.mean(predicted == saved))
    low, high = interval(paired_bootstrap(val["label"], predicted, saved, groups=val["qid"]) * 100)
    if rung in SAME_MODEL_RUNGS:
        gate = "same model" if agreement >= SAME_MODEL_AGREEMENT else "differs"
    else:
        gate = "pass" if low > -GATE_MARGIN_POINTS else "fail"
    return {"rung": rung, "gate": gate, "predicted": predicted, "model_size_mb": size_mb, "latency_p50_ms": p50,
            "latency_p95_ms": p95, "val_macro_f1": macro_f1(val["label"], predicted), "label_agreement": agreement,
            "gain_ci_low_points": low, "gain_ci_high_points": high}


def thread_sweep(model_dir: Path, texts: list[str]) -> dict[str, float]:
    """Per-message latency alone, and throughput with 4 simultaneous callers, at 1-4 threads per call."""
    results = {}
    for threads in (1, 2, 3, 4):
        classifier = OnnxClassifier(model_dir, threads=threads)
        one = lambda text: classifier.predict_proba([text])  # noqa: E731
        results[f"threads_{threads}_alone_p50_ms"] = latency_ms(one, texts)[0]
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(one, texts))
        results[f"threads_{threads}_4_callers_msgs_per_s"] = len(texts) / (time.perf_counter() - start)
        print(f"  {threads} threads: alone p50 {results[f'threads_{threads}_alone_p50_ms']:.0f} ms, "
              f"4 callers {results[f'threads_{threads}_4_callers_msgs_per_s']:.2f} msgs/s", flush=True)
    return results


def log_rung(row: dict, tags: dict) -> None:
    with mlflow.start_run(run_name=row["rung"], tags={**tags, TagEnum.RUNG: row["rung"], "gate": row["gate"]}):
        mlflow.log_metrics({key: value for key, value in row.items() if isinstance(value, float)})


def register_variant(args, client, settings, lineage, source, val, saved, intent_names) -> None:
    rung_dir = settings.RUNS_DIR / "variants" / args.register
    if not (rung_dir / "model.onnx").exists():
        raise SystemExit(f"build {args.register} first: uv run python scripts/build_variants.py --from-version {args.from_version}")
    if lineage[TagEnum.GIT_DIRTY] != "false":
        raise SystemExit("commit first: a registered version must trace back to a commit")
    row = measure(args.register, OnnxClassifier(rung_dir).predict_proba, folder_mb(rung_dir), val, saved)
    if row["gate"] not in ("same model", "pass"):
        raise SystemExit(f"{args.register} did not pass its check ({row['gate']}); not registered")
    mlflow.set_experiment(ExperimentEnum.INTENT_CLASSIFIER)
    tags = {**lineage, TagEnum.DATA_VARIANT: DataVariantEnum.GROUPED, TagEnum.MODEL_FAMILY: ModelFamilyEnum.ARABERT,
            TagEnum.RUNG: args.register, TagEnum.DERIVED_FROM: f"v{source.version}"}
    with mlflow.start_run(run_name=f"arabert-{args.register}", tags=tags) as run:
        log_evaluation("val", val["label"], row["predicted"], intent_names, val["qid"])
        mlflow.log_metrics({key: row[key] for key in ("latency_p50_ms", "latency_p95_ms", "model_size_mb",
                                                      "gain_ci_low_points", "gain_ci_high_points")})
        log_val_predictions(val, row["predicted"])
        info = mlflow.pyfunc.log_model(name="model", python_model=onnx_model.__file__,
                                       artifacts={"model_dir": str(rung_dir)}, input_example=val[[TEXT_COLUMN]].head(3),
                                       metadata={ModelMetadataEnum.INTENT_NAMES.value: intent_names})
    evidence = {TagEnum.RUNG: args.register, TagEnum.DERIVED_FROM: f"v{source.version}", "gate": row["gate"],
                "gain_ci_low_points": f"{row['gain_ci_low_points']:+.2f}"}
    print(register(client, client.get_run(run.info.run_id), settings.REGISTERED_MODEL_NAME, args.alias, args.note,
                   evidence, info.model_uri))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and measure the ONNX variants of an AraBERT version.")
    parser.add_argument("--from-version", required=True)
    parser.add_argument("--rungs", nargs="+", choices=[REFERENCE, *RUNGS, "speed", "threads"], default=[REFERENCE, *RUNGS, "speed", "threads"],
                        help="measure only these (built variants are reused)")
    parser.add_argument("--register", choices=RUNGS, help="log and register this rung (built, and passing its check)")
    parser.add_argument("--alias", type=AliasEnum, choices=list(AliasEnum), default=None)
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    settings = get_settings()
    connect(settings.MLFLOW_TRACKING_URI)
    lineage = {str(key): value for key, value in
               lineage_tags(PROJECT_ROOT, settings.DATA_RAW_DIR, settings.DATA_PROCESSED_DIR).items()}
    client = MlflowClient()
    source = client.get_model_version(settings.REGISTERED_MODEL_NAME, args.from_version)
    if source.tags.get(TagEnum.MODEL_FAMILY) != ModelFamilyEnum.ARABERT:
        raise SystemExit(f"v{args.from_version} is not an AraBERT model")
    train, val = load_train_val(DataVariantEnum.GROUPED, settings.DATA_RAW_DIR, settings.DATA_PROCESSED_DIR)
    saved = val_predictions(source.run_id)
    assert np.array_equal(saved["label"].to_numpy(), val["label"].to_numpy()), "validation rows differ"
    saved = saved["predicted"].to_numpy()
    intent_names = load_label_names(settings.DATA_PROCESSED_DIR)
    root = settings.RUNS_DIR / "variants"
    source_dir = download_model_folder(source.source, root / f"v{args.from_version}-pytorch")

    if args.register:
        register_variant(args, client, settings, lineage, source, val, saved, intent_names)
        return

    mlflow.set_experiment(ExperimentEnum.OPTIMISATION)
    tags = {**lineage, TagEnum.DERIVED_FROM: f"v{args.from_version}"}
    calibration = train.groupby(["label", "dialect"]).sample(n=CALIBRATION_PER_GROUP, random_state=0)[TEXT_COLUMN].tolist()
    rows = []
    if REFERENCE in args.rungs:
        # The original: its validation predictions are the saved ones; its latency is measured like every variant's
        reference = AraBertClassifier(source_dir)
        p50, p95 = latency_ms(lambda text: reference.predict_proba([text]), val[TEXT_COLUMN].tolist()[:LATENCY_SAMPLE])
        rows.append({"rung": REFERENCE, "gate": "reference", "model_size_mb": folder_mb(source_dir), "latency_p50_ms": p50,
                     "latency_p95_ms": p95, "val_macro_f1": macro_f1(val["label"], saved), "label_agreement": 1.0,
                     "gain_ci_low_points": 0.0, "gain_ci_high_points": 0.0})
        log_rung(rows[-1], tags)
        del reference
    for rung in [rung for rung in RUNGS if rung in args.rungs]:
        print(f"{rung}: building and measuring", flush=True)
        try:
            build(rung, root, source_dir, calibration)
            classifier = OnnxClassifier(root / rung)
            rows.append(measure(rung, classifier.predict_proba, folder_mb(root / rung), val, saved))
            del classifier  # one model in memory at a time: this laptop has little to spare
        except Exception as error:  # a rung that cannot run on this CPU is a result too
            rows.append({"rung": rung, "gate": f"could not run: {type(error).__name__}: {str(error)[:120]}"})
        log_rung(rows[-1], tags)  # at once, so a crash later loses only the rung in progress
        gc.collect()
    if "speed" in args.rungs:
        # Every built variant timed back to back on the same messages, so all share the machine's conditions
        print("speed, back to back:", flush=True)
        speed = {}
        for rung in [REFERENCE, *RUNGS]:
            folder = source_dir if rung == REFERENCE else root / rung
            if rung != REFERENCE and not (folder / "model.onnx").exists():
                continue
            classifier = AraBertClassifier(folder) if rung == REFERENCE else OnnxClassifier(folder)
            p50, p95 = latency_ms(lambda text: classifier.predict_proba([text]), val[TEXT_COLUMN].tolist()[:LATENCY_SAMPLE])
            speed |= {f"{rung}_p50_ms": p50, f"{rung}_p95_ms": p95}
            print(f"  {rung:20} p50 {p50:5.0f} ms, p95 {p95:5.0f} ms", flush=True)
            del classifier
            gc.collect()
        with mlflow.start_run(run_name="speed, back to back", tags={**tags, TagEnum.RUNG: "speed"}):
            mlflow.log_metrics(speed)
    if "threads" in args.rungs:
        print("thread sweep on onnx-optimised:", flush=True)
        sweep = thread_sweep(root / "onnx-optimised", val[TEXT_COLUMN].tolist()[:SWEEP_SAMPLE])
        with mlflow.start_run(run_name="thread sweep (onnx-optimised)", tags={**tags, TagEnum.RUNG: "threads"}):
            mlflow.log_metrics(sweep)

    print(f"\n{'rung':20} {'size MB':>8} {'p50 ms':>7} {'p95 ms':>7} {'macro-F1':>9} {'same labels':>11} "
          f"{'change, 95% (points)':>21}  check")
    for row in rows:
        if "latency_p50_ms" not in row:
            print(f"{row['rung']:20} {row['gate']}")
            continue
        print(f"{row['rung']:20} {row['model_size_mb']:8.0f} {row['latency_p50_ms']:7.0f} {row['latency_p95_ms']:7.0f} "
              f"{row['val_macro_f1']:9.4f} {row['label_agreement']:11.2%} "
              f"{row['gain_ci_low_points']:+10.2f} to {row['gain_ci_high_points']:+.2f}  {row['gate']}")


if __name__ == "__main__":
    main()
