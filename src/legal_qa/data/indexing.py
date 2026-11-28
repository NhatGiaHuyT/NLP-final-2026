"""Per-state BM25 indexes in data/index/<state-slug>/."""

import time
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import bm25s
import pandas as pd
import Stemmer

from legal_qa.data.corpus import CHUNK_OVERLAP, CHUNK_WORDS, chunk_statutes, find_question_leaks, index_text
from legal_qa.data.housingqa import state_key
from legal_qa.data.io import write_json
from legal_qa.paths import INDEX_DIR

STOPWORDS = "en"
STEMMER = "english"


@lru_cache(maxsize=1)
def _stemmer() -> Stemmer.Stemmer:
    return Stemmer.Stemmer(STEMMER)


def tokenize(texts: Iterable[str], return_ids: bool = True):
    return bm25s.tokenize(list(texts), stopwords=STOPWORDS, stemmer=_stemmer(), return_ids=return_ids, show_progress=False)


def state_slug(state: str) -> str:
    return state_key(state).replace(" ", "-")


def state_index_dir(state: str, index_dir: str | Path = INDEX_DIR) -> Path:
    return Path(index_dir) / state_slug(state)


def build_state_index(chunks: list[dict], out_dir: str | Path) -> None:
    retriever = bm25s.BM25()
    retriever.index(tokenize(index_text(c) for c in chunks), show_progress=False)
    retriever.save(out_dir, corpus=chunks, show_progress=False)


def load_state_index(state: str, index_dir: str | Path = INDEX_DIR, mmap: bool = True) -> bm25s.BM25:
    return bm25s.BM25.load(state_index_dir(state, index_dir), load_corpus=True, mmap=mmap)


def search(retriever: bm25s.BM25, query: str, k: int = 5) -> list[dict]:
    """Top-k chunks (metadata + BM25 score) for one query."""
    docs, scores = retriever.retrieve(tokenize([query], return_ids=False), k=k, show_progress=False)
    return [{**doc, "score": float(score)} for doc, score in zip(docs[0], scores[0])]


def build_corpus_indexes(
    statutes: pd.DataFrame,
    states: Iterable[str],
    index_dir: str | Path = INDEX_DIR,
    leak_check_questions: dict[str, list[str]] | None = None,
    size: int = CHUNK_WORDS,
    overlap: int = CHUNK_OVERLAP,
) -> pd.DataFrame:
    """Chunk + index every state, write manifest.json, return per-state stats."""
    index_dir = Path(index_dir)
    rows_by_state = statutes.groupby("state").indices  # copy one state at a time
    stats = []
    for state in sorted(states):
        t0 = time.time()
        df = statutes.iloc[rows_by_state[state_key(state)]]
        chunks = chunk_statutes(df, size, overlap)
        leaks = find_question_leaks({state: leak_check_questions.get(state, [])}, chunks) if leak_check_questions else {}
        build_state_index(chunks, state_index_dir(state, index_dir))
        stats.append({
            "state": state,
            "slug": state_slug(state),
            "n_statutes": len(df),
            "n_chunks": len(chunks),
            "chunks_per_statute": round(len(chunks) / len(df), 2),
            "question_leaks": sum(len(v) for v in leaks.values()),
            "build_seconds": round(time.time() - t0, 1),
        })
        print(f"{state:22s} statutes={len(df):7d} chunks={len(chunks):7d} {stats[-1]['build_seconds']}s", flush=True)
    stats_df = pd.DataFrame(stats)
    write_json(
        {
            "chunk_words": size,
            "chunk_overlap": overlap,
            "bm25_stopwords": STOPWORDS,
            "bm25_stemmer": STEMMER,
            "indexed_text": "path headings (without first part) + chunk text",
            "states": stats_df.to_dict(orient="records"),
        },
        index_dir / "manifest.json",
    )
    return stats_df
