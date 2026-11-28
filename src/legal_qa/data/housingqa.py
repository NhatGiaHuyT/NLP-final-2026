"""HousingQA adapter: `load_dataset` first, raw zip fallback (its loading script fails on datasets>=4)."""

import json
import zipfile
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from legal_qa.data.base import LegalQADataset
from legal_qa.data.io import read_jsonl
from legal_qa.paths import PROCESSED_DIR, RAW_DIR

REPO_ID = "reglab/housing_qa"
RAW_FILES = {
    "questions": "data/questions.json.zip",
    "questions_aux": "data/questions_aux.json.zip",
    "statutes": "data/statutes.tsv.zip",
}
HOUSINGQA_RAW_DIR = RAW_DIR / "housing_qa"


def raw_path(subset: str) -> Path:
    return HOUSINGQA_RAW_DIR / RAW_FILES[subset]


def download(subsets: Iterable[str] = RAW_FILES, force: bool = False) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    paths = {}
    for subset in subsets:
        if force or not raw_path(subset).exists():
            hf_hub_download(REPO_ID, RAW_FILES[subset], repo_type="dataset", local_dir=HOUSINGQA_RAW_DIR)
        paths[subset] = raw_path(subset)
    return paths


def try_load_dataset(subset: str):
    """`load_dataset` result, or None if it fails."""
    try:
        import datasets

        split = "corpus" if subset == "statutes" else "test"
        # Flag was removed in datasets>=4.
        kwargs = {"trust_remote_code": True} if int(datasets.__version__.split(".")[0]) < 4 else {}
        ds = datasets.load_dataset(REPO_ID, subset, split=split, **kwargs)
    except Exception as e:
        print(f"[housing_qa] load_dataset('{subset}') failed ({type(e).__name__}: {str(e)[:120]}) -> using raw zip")
        return None
    print(f"[housing_qa] '{subset}' loaded with load_dataset")
    return ds


def _questions_from_zip(subset: str) -> list[dict]:
    download([subset])
    with zipfile.ZipFile(raw_path(subset)) as z:
        return json.loads(z.read(f"{subset}.json"))


def load_questions(subset: str = "questions", use_datasets: bool = True) -> list[dict]:
    """Raw rows of `questions` (with statute_idx) or `questions_aux` (without)."""
    if subset not in ("questions", "questions_aux"):
        raise ValueError(f"Unknown question subset: {subset}")
    ds = try_load_dataset(subset) if use_datasets else None
    if ds is None:
        return _questions_from_zip(subset)
    rows = ds.to_list()
    for r in rows:
        # Sequence(dict) features come back as a dict of lists.
        if isinstance(r["statutes"], dict):
            r["statutes"] = [dict(zip(r["statutes"], values)) for values in zip(*r["statutes"].values())]
    return rows


def state_key(state: str) -> str:
    # The statutes corpus uses lowercase state names.
    return state.strip().lower()


def iter_statutes(
    states: Iterable[str] | None = None, chunksize: int = 200_000, use_datasets: bool = True
) -> Iterator[pd.DataFrame]:
    """Statutes (citation, path, state, text, idx) in DataFrame batches."""
    keys = {state_key(s) for s in states} if states is not None else None
    ds = try_load_dataset("statutes") if use_datasets else None
    if ds is not None:
        batches = (pd.DataFrame(b).astype({"idx": "int64"}) for b in ds.iter(batch_size=chunksize))
    else:
        download(["statutes"])
        batches = pd.read_csv(
            raw_path("statutes"), sep="\t", chunksize=chunksize, dtype={"idx": "int64"}, keep_default_na=False
        )
    for df in batches:
        yield df if keys is None else df[df["state"].isin(keys)]


def load_statutes(states: Iterable[str] | None = None, use_datasets: bool = True) -> pd.DataFrame:
    return pd.concat(iter_statutes(states, use_datasets=use_datasets), ignore_index=True)


class HousingQA(LegalQADataset):
    name = "housing_qa"

    def __init__(self, processed_dir: str | Path = PROCESSED_DIR):
        self.processed_dir = Path(processed_dir)

    def splits(self, include_aux: bool = False) -> dict[str, list[dict]]:
        out = {s: read_jsonl(self.processed_dir / f"{s}.jsonl") for s in ("train", "val", "test")}
        if include_aux:
            out["train"] = out["train"] + read_jsonl(self.processed_dir / "train_aux.jsonl")
        return out

    def corpus(self, states: list[str] | None = None) -> Iterator[pd.DataFrame]:
        return iter_statutes(states)

    def gold_ids(self, record: dict) -> list[int]:
        return record["gold_statute_idx"]
