import numpy as np
import pytest

from arabic_banking_intents.ml.evaluate import bootstrap, evaluate, interval, macro_f1, paired_bootstrap

NAMES = [f"intent_{i}" for i in range(77)]


def test_hand_calculated_example():
    # intent 0: precision 1/2, recall 1/2 -> F1 0.5
    # intent 1: precision 2/3, recall 1   -> F1 0.8
    # intent 2: never predicted           -> F1 0
    result = evaluate([0, 0, 1, 1, 2], [0, 1, 1, 1, 0], NAMES)

    assert result["accuracy"] == pytest.approx(3 / 5)
    assert result["macro_f1"] == pytest.approx((0.5 + 0.8 + 0) / 3)
    assert {(c["true"], c["predicted"]) for c in result["top_confusions"]} == {
        ("intent_0", "intent_1"),
        ("intent_2", "intent_0"),
    }


def test_macro_f1_averages_only_over_intents_in_the_test_set():
    # Intent 5 is predicted once but never appears in y_true (as in the Tunisian set, which has
    # 27 of 77 intents). intent 0: F1 1; intent 1: precision 1, recall 1/2 -> F1 2/3.
    # Averaging over all 77 intents would give 0.02; over the predicted ones too, 0.56.
    assert macro_f1([0, 0, 1, 1], [0, 0, 1, 5]) == pytest.approx((1 + 2 / 3) / 2)


def test_bootstrap_is_reproducible_and_brackets_the_score():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, size=200)
    y_pred = np.where(rng.random(200) < 0.8, y_true, rng.integers(0, 5, size=200))
    groups = np.arange(200) // 2  # two versions per question

    samples = bootstrap(y_true, y_pred, groups, n_resamples=200, seed=1)
    low, high = interval(samples)

    assert np.array_equal(samples, bootstrap(y_true, y_pred, groups, n_resamples=200, seed=1))
    assert low < macro_f1(y_true, y_pred) < high
    # a model compared with itself differs by exactly 0 on every resample
    assert np.all(paired_bootstrap(y_true, y_pred, y_pred, groups, n_resamples=50) == 0)
