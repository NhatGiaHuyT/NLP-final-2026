from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from legal_qa.data.preprocessing import qa_key, resolve_duplicates

SPLITS = ("train", "val", "test")
GROUP_KEY = "question_group"


def group_split(
    records: list[dict], val_size: float = 0.1, test_size: float = 0.1, seed: int = 42, group_key: str = GROUP_KEY
) -> dict[str, list[dict]]:
    """Two-stage GroupShuffleSplit; sizes are fractions of groups."""
    groups = np.array([r[group_key] for r in records])
    idx = np.arange(len(records))

    outer = GroupShuffleSplit(n_splits=1, test_size=val_size + test_size, random_state=seed)
    train_idx, rest_idx = next(outer.split(idx, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=test_size / (val_size + test_size), random_state=seed)
    val_rel, test_rel = next(inner.split(rest_idx, groups=groups[rest_idx]))

    parts = {"train": train_idx, "val": rest_idx[val_rel], "test": rest_idx[test_rel]}
    return {name: [records[i] for i in sorted(ix)] for name, ix in parts.items()}


def check_no_leakage(splits: dict[str, list[dict]], group_key: str = GROUP_KEY) -> None:
    for field in (group_key, "question", "id"):
        values = {name: {r[field] for r in rows} for name, rows in splits.items()}
        for a, b in combinations(values, 2):
            overlap = values[a] & values[b]
            if overlap:
                raise AssertionError(f"{len(overlap)} `{field}` values shared by {a} and {b}")


def aux_train_extension(aux_rows: list[dict], train_groups: set[int], exclude_keys: set[tuple[str, str]]) -> list[dict]:
    """questions_aux rows not in `questions`, from train groups only."""
    candidates = [r for r in aux_rows if qa_key(r) not in exclude_keys]
    kept, _ = resolve_duplicates(candidates)
    return [r for r in kept if r[GROUP_KEY] in train_groups]


def split_summary(splits: dict[str, list[dict]]) -> pd.DataFrame:
    total = sum(len(rows) for rows in splits.values())
    rows = []
    for name, recs in splits.items():
        n_yes = sum(r["answer"] == "yes" for r in recs)
        rows.append({
            "split": name,
            "n_questions": len(recs),
            "share": round(len(recs) / total, 3),
            "n_groups": len({r[GROUP_KEY] for r in recs}),
            "n_states": len({r["state"] for r in recs}),
            "n_yes": n_yes,
            "n_no": len(recs) - n_yes,
            "yes_ratio": round(n_yes / len(recs), 3),
        })
    return pd.DataFrame(rows)
