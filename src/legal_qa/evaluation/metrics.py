"""Answer-quality, retrieval and significance metrics (numpy/scipy only, no torch).

All answer metrics compare normalised strings, so the same code works for
yes/no targets and for free-text targets.
"""

import re
import string
from collections import Counter
from typing import Sequence

import numpy as np

LABELS = ("yes", "no")


def normalize_text(s: str) -> str:
    """SQuAD-style normalisation: lowercase, strip punctuation and articles."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def normalize_label(s: str) -> str:
    """Map a generated string to 'yes' / 'no', or 'invalid' if it is neither."""
    a = normalize_text(s)
    return a if a in LABELS else "invalid"


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_text(pred) == normalize_text(gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_text(pred).split(), normalize_text(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def f1_for_class(y_true: Sequence[str], y_pred: Sequence[str], cls: str) -> float:
    tp = sum(t == cls and p == cls for t, p in zip(y_true, y_pred))
    fp = sum(t != cls and p == cls for t, p in zip(y_true, y_pred))
    fn = sum(t == cls and p != cls for t, p in zip(y_true, y_pred))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def mcc(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Binary Matthews correlation; 'invalid' predictions count as wrong (neither class)."""
    tp = sum(t == "yes" and p == "yes" for t, p in zip(y_true, y_pred))
    tn = sum(t == "no" and p == "no" for t, p in zip(y_true, y_pred))
    fp = sum(t == "no" and p == "yes" for t, p in zip(y_true, y_pred))
    fn = sum(t == "yes" and p != "yes" for t, p in zip(y_true, y_pred))
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return 0.0 if den == 0 else (tp * tn - fp * fn) / den


def answer_metrics(golds: Sequence[str], preds: Sequence[str]) -> dict:
    """Metrics for one system. `preds` are raw generated strings."""
    y_true = [normalize_label(g) for g in golds]
    y_pred = [normalize_label(p) for p in preds]
    n = len(y_true)
    return {
        "n": n,
        "accuracy": float(np.mean([t == p for t, p in zip(y_true, y_pred)])),
        "exact_match": float(np.mean([exact_match(p, g) for p, g in zip(preds, golds)])),
        "token_f1": float(np.mean([token_f1(p, g) for p, g in zip(preds, golds)])),
        "f1_yes": f1_for_class(y_true, y_pred, "yes"),
        "f1_no": f1_for_class(y_true, y_pred, "no"),
        "macro_f1": (f1_for_class(y_true, y_pred, "yes") + f1_for_class(y_true, y_pred, "no")) / 2,
        "mcc": mcc(y_true, y_pred),
        "pred_yes_rate": float(np.mean([p == "yes" for p in y_pred])),
        "invalid_rate": float(np.mean([p == "invalid" for p in y_pred])),
    }


def majority_baseline(train_golds: Sequence[str], test_golds: Sequence[str]) -> dict:
    """Always predict the most frequent training label."""
    label = Counter(normalize_label(g) for g in train_golds).most_common(1)[0][0]
    return {"label": label, **answer_metrics(test_golds, [label] * len(test_golds))}


def cluster_bootstrap_ci(
    correct: np.ndarray, groups: Sequence, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    """CI for a mean of per-question scores, resampling whole `question_group`s.

    The test set has only ~23 distinct question texts (each asked in ~30 states), so rows are
    not independent; resampling rows would give intervals that are far too narrow.
    """
    correct = np.asarray(correct, dtype=float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    sums = np.array([correct[groups == g].sum() for g in uniq])
    cnts = np.array([(groups == g).sum() for g in uniq])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def paired_cluster_bootstrap(
    correct_a: np.ndarray, correct_b: np.ndarray, groups: Sequence, n_boot: int = 10_000, seed: int = 42
) -> dict:
    """Difference in accuracy (a - b) with a group-resampled CI and a two-sided bootstrap p-value."""
    d = np.asarray(correct_a, dtype=float) - np.asarray(correct_b, dtype=float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    sums = np.array([d[groups == g].sum() for g in uniq])
    cnts = np.array([(groups == g).sum() for g in uniq])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = sums[idx].sum(1) / cnts[idx].sum(1)
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"diff": float(d.mean()), "ci_low": float(np.quantile(boot, 0.025)),
            "ci_high": float(np.quantile(boot, 0.975)), "p_value": float(min(1.0, p))}


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar test on per-question correctness (ignores group structure; report with the bootstrap)."""
    from scipy.stats import binomtest

    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(only_a, n, 0.5).pvalue)
    return {"a_only_correct": only_a, "b_only_correct": only_b, "p_value": p}


# ---------------------------------------------------------------- retrieval
def statute_ranking(hits: list[dict]) -> list[int]:
    """Unique statute ids in rank order from a ranked list of chunk hits."""
    return list(dict.fromkeys(h["statute_idx"] for h in hits))


def retrieval_metrics(ranked_statutes: list[list[int]], golds: list[list[int]], ks=(1, 3, 5, 10)) -> dict:
    """Statute-level Recall@k (any gold in top-k), full recall@k (all gold) and MRR."""
    out = {}
    for k in ks:
        out[f"hit@{k}"] = float(np.mean([bool(set(r[:k]) & set(g)) for r, g in zip(ranked_statutes, golds)]))
        out[f"recall@{k}"] = float(np.mean([len(set(r[:k]) & set(g)) / len(set(g)) for r, g in zip(ranked_statutes, golds)]))
    rr = []
    for r, g in zip(ranked_statutes, golds):
        rank = next((i + 1 for i, s in enumerate(r) if s in set(g)), None)
        rr.append(0.0 if rank is None else 1.0 / rank)
    out["mrr"] = float(np.mean(rr))
    return out
