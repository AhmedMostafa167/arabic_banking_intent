"""Scoring shared by every model, so every comparison is made the same way on the same data.

Macro-F1 averages over the intents present in `y_true`. The Tunisian test set has 27 of the 77
intents, and counting the other 50 as zeros would score a model on intents it was never asked.

Bootstrap intervals show how much a score depends on which examples happen to be in the evaluation
set. They resample whole questions (`groups`), because versions of one question are not
independent evidence.
"""

from collections.abc import Callable, Iterator, Sequence

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, f1_score

Metric = Callable[[np.ndarray, np.ndarray], float]


def macro_f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """F1 averaged over the intents that appear in `y_true`."""
    return float(f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro", zero_division=0))


def evaluate(y_true: ArrayLike, y_pred: ArrayLike, intent_names: Sequence[str], n_confusions: int = 10) -> dict:
    """Accuracy, macro-F1, F1 per intent, and the most-confused (true, predicted) intent pairs."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    present = np.unique(y_true)
    per_intent = f1_score(y_true, y_pred, labels=present, average=None, zero_division=0)

    confusions = []
    wrong = y_true != y_pred
    if wrong.any():
        pairs, counts = np.unique(np.column_stack([y_true[wrong], y_pred[wrong]]), axis=0, return_counts=True)
        for i in np.argsort(-counts, kind="stable")[:n_confusions]:
            true, predicted = pairs[i]
            confusions.append({"true": intent_names[true], "predicted": intent_names[predicted], "count": int(counts[i])})

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": macro_f1(y_true, y_pred),
        "per_intent_f1": {intent_names[i]: float(f1) for i, f1 in zip(present, per_intent)},
        "top_confusions": confusions,
    }


def _resamples(n_rows: int, groups: ArrayLike | None, n_resamples: int, seed: int) -> Iterator[np.ndarray]:
    """Row indices of each resample: as many groups as there are, drawn with replacement."""
    rng = np.random.default_rng(seed)
    if groups is None:
        for _ in range(n_resamples):
            yield rng.integers(n_rows, size=n_rows)
        return

    _, group_of_row = np.unique(np.asarray(groups), return_inverse=True)
    rows_by_group = np.split(np.argsort(group_of_row, kind="stable"), np.cumsum(np.bincount(group_of_row))[:-1])
    for _ in range(n_resamples):
        picked = rng.integers(len(rows_by_group), size=len(rows_by_group))
        yield np.concatenate([rows_by_group[g] for g in picked])


def bootstrap(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    groups: ArrayLike | None = None,
    metric: Metric = macro_f1,
    n_resamples: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """The metric on `n_resamples` resamples of the evaluation set.

    For two models scored on different sets (the leak check), subtract their two bootstrap arrays,
    drawn with different seeds, and take the `interval` of the difference.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.array([metric(y_true[idx], y_pred[idx]) for idx in _resamples(len(y_true), groups, n_resamples, seed)])


def paired_bootstrap(
    y_true: ArrayLike,
    pred_a: ArrayLike,
    pred_b: ArrayLike,
    groups: ArrayLike | None = None,
    metric: Metric = macro_f1,
    n_resamples: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """metric(A) - metric(B) for two models scored on the same resamples of the same set.

    Sharper than comparing two separate intervals, because both models face the same easy and hard
    examples in every resample.
    """
    y_true, pred_a, pred_b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    return np.array([
        metric(y_true[idx], pred_a[idx]) - metric(y_true[idx], pred_b[idx])
        for idx in _resamples(len(y_true), groups, n_resamples, seed)
    ])


def interval(samples: ArrayLike, level: float = 0.95) -> tuple[float, float]:
    """The range holding the central `level` share of the bootstrap samples."""
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(samples, [tail, 100 - tail])
    return float(low), float(high)
