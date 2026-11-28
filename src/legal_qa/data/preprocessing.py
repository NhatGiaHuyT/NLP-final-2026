from collections import defaultdict
from typing import Iterable

import numpy as np

MODEL_NAME = "google/flan-t5-small"
LABELS = ("yes", "no")


def normalize_answer(answer: str) -> str:
    a = answer.strip().lower().rstrip(".")
    if a not in LABELS:
        raise ValueError(f"Not a yes/no answer: {answer!r}")
    return a


def format_input(state: str, question: str) -> str:
    # Questions repeat across states, only the state tells them apart.
    return f"state: {state} question: {' '.join(question.split())}"


def qa_key(raw: dict) -> tuple[str, str]:
    return raw["state"], " ".join(raw["question"].split())


def resolve_duplicates(rows: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """One row per (state, question); keys with conflicting answers are dropped."""
    by_key = defaultdict(list)
    for r in rows:
        by_key[qa_key(r)].append(r)
    kept, dropped = [], []
    for dup in by_key.values():
        if len({normalize_answer(r["answer"]) for r in dup}) == 1:
            kept.append(dup[0])
        else:
            dropped.extend(dup)
    return kept, dropped


def to_record(raw: dict, source: str = "questions") -> dict:
    answer = normalize_answer(raw["answer"])
    prefix = "q" if source == "questions" else "aux"
    return {
        "id": f"{prefix}-{raw['idx']}",
        "source": source,
        "state": raw["state"],
        "question_group": int(raw["question_group"]),
        "question": " ".join(raw["question"].split()),
        "answer": answer,
        "input_text": format_input(raw["state"], raw["question"]),
        "target_text": answer,
        # questions_aux has no statute_idx.
        "gold_statute_idx": list(dict.fromkeys(s["statute_idx"] for s in raw["statutes"] if "statute_idx" in s)),
        "statutes": raw["statutes"],
        "original_question": raw["original_question"],
        "caveats": [c for c in raw["caveats"] if c and c.strip()],
    }


def token_lengths(texts: Iterable[str], tokenizer) -> np.ndarray:
    return np.array([len(ids) for ids in tokenizer(list(texts))["input_ids"]])
