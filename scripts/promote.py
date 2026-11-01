"""Promote a model family if every one of its seed runs passes the gate against the current champion.

    uv run python scripts/promote.py --runs <run id> <run id> <run id> [--note "..."]

The gate, agreed before AraBERT was trained: for each seed, the paired 95% bootstrap interval of
(candidate − champion) macro-F1 on our validation split lies entirely above 0. If every seed passes,
the median seed is registered and becomes @champion; otherwise nothing changes. Latency is reported
in the version's record but does not decide.
"""

import argparse

import numpy as np
from mlflow import MlflowClient

from arabic_banking_intents.helpers.config import get_settings
from arabic_banking_intents.training.evaluate import interval, macro_f1, paired_bootstrap
from arabic_banking_intents.training.registry import candidate_problems, register, val_predictions
from arabic_banking_intents.training.tracking import connect
from arabic_banking_intents.models.enums.TrackingEnum import AliasEnum


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the promotion gate to a family's seed runs.")
    parser.add_argument("--runs", nargs="+", required=True, help="the candidate's seed run IDs")
    parser.add_argument("--note", default="", help="why this family is being promoted")
    args = parser.parse_args()

    settings = get_settings()
    connect(settings.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    champion = client.get_model_version_by_alias(settings.REGISTERED_MODEL_NAME, AliasEnum.CHAMPION)
    champion_val = val_predictions(champion.run_id)

    results = []
    for run_id in args.runs:
        run = client.get_run(run_id)
        problems = candidate_problems(client, run)
        if problems:
            raise SystemExit(f"{run_id}: " + "; ".join(problems))
        candidate_val = val_predictions(run_id)
        same_rows = np.array_equal(candidate_val[["qid", "label"]].to_numpy(), champion_val[["qid", "label"]].to_numpy())
        if not same_rows:
            raise SystemExit(f"{run_id}: candidate and champion were scored on different validation rows")

        diffs = paired_bootstrap(candidate_val["label"], candidate_val["predicted"], champion_val["predicted"],
                                 groups=candidate_val["qid"])
        low, high = interval(diffs)
        gain = macro_f1(candidate_val["label"], candidate_val["predicted"]) - macro_f1(champion_val["label"], champion_val["predicted"])
        results.append({"run": run, "seed": run.data.params.get("seed", "?"), "gain": gain, "low": low, "high": high})
        print(f"seed {results[-1]['seed']}: val macro-F1 {run.data.metrics['val_macro_f1']:.4f}, "
              f"gain over v{champion.version} {gain * 100:+.2f} points (95% interval {low * 100:+.2f} to {high * 100:+.2f}): "
              f"{'pass' if low > 0 else 'FAIL'}")

    if any(r["low"] <= 0 for r in results):
        raise SystemExit(f"gate failed: not every seed beats v{champion.version} beyond the noise; @champion unchanged")

    by_score = sorted(results, key=lambda r: r["run"].data.metrics["val_macro_f1"])
    median = by_score[len(by_score) // 2]
    evidence = {
        "gate": f"passed by all {len(results)} seeds against v{champion.version}",
        "gate_min_ci_low_points": f"{min(r['low'] for r in results) * 100:+.2f}",
        "seed": str(median["seed"]),
    }
    print(register(client, median["run"], settings.REGISTERED_MODEL_NAME, AliasEnum.CHAMPION, args.note, evidence))


if __name__ == "__main__":
    main()
