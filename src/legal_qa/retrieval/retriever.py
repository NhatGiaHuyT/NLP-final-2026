"""M3 retrieval: per-state BM25 (teammate's index) -> ranked hits cached to disk -> RAG model inputs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from tqdm.auto import tqdm

from legal_qa.data.indexing import load_state_index, search
from legal_qa.data.io import read_jsonl, write_jsonl
from legal_qa.paths import INDEX_DIR

KEEP = ("chunk_id", "statute_idx", "chunk_no", "citation", "path", "text", "score")


def retrieve_records(records: list[dict], n_chunks: int = 50, index_dir: str | Path = INDEX_DIR) -> list[dict]:
    """Top `n_chunks` BM25 chunks for each record's question, searched only in the question's state.

    Only `question` is used as the query (never the answer or gold statutes), so the result is
    legitimate for validation and test records.
    """
    by_state = defaultdict(list)
    for i, r in enumerate(records):
        by_state[r["state"]].append(i)
    out = [None] * len(records)
    for state, idxs in tqdm(sorted(by_state.items()), desc="states"):
        retriever = load_state_index(state, index_dir)
        for i in idxs:
            hits = search(retriever, records[i]["question"], k=n_chunks)
            out[i] = {"id": records[i]["id"], "hits": [{k: h[k] for k in KEEP} for h in hits]}
        del retriever
    return out


def cache_retrieval(records: list[dict], path: str | Path, n_chunks: int = 50, index_dir=INDEX_DIR) -> list[dict]:
    path = Path(path)
    if path.exists():
        cached = read_jsonl(path)
        if [c["id"] for c in cached] == [r["id"] for r in records]:
            return cached
    res = retrieve_records(records, n_chunks, index_dir)
    write_jsonl(res, path)
    return res


def top_statute_hits(hits: list[dict], k: int) -> list[dict]:
    """Best-scoring chunk of each of the first k distinct statutes (so k passages = k provisions)."""
    seen, out = set(), []
    for h in hits:
        if h["statute_idx"] not in seen:
            seen.add(h["statute_idx"])
            out.append(h)
            if len(out) == k:
                break
    return out


def build_rag_input(record: dict, hits: list[dict], tokenizer, k: int, max_len: int = 512) -> str:
    """`<M2 input> context: [1] <passage> [2] <passage> ...`, with the question first so it is never cut.

    The token budget left after the question is split equally among the k passages (each is truncated
    to its share), so the whole input fits `max_len` instead of silently dropping the later passages.
    """
    prefix = f"{record['input_text']} context:"
    if k == 0:
        return prefix
    budget = max_len - len(tokenizer(prefix)["input_ids"])  # includes the final EOS
    per = max(budget // k - 4, 8)                           # -4: room for the "[i]" marker
    parts = [prefix]
    for i, h in enumerate(top_statute_hits(hits, k), 1):
        ids = tokenizer(h["text"], add_special_tokens=False)["input_ids"][:per]
        parts.append(f"[{i}] " + tokenizer.decode(ids, skip_special_tokens=True))
    return " ".join(parts)


def make_rag_records(records: list[dict], retrieved: list[dict], tokenizer, k: int, max_len: int = 512) -> list[dict]:
    assert [r["id"] for r in records] == [x["id"] for x in retrieved], "retrieval cache misaligned"
    return [{**r, "input_text": build_rag_input(r, x["hits"], tokenizer, k, max_len)} for r, x in zip(records, retrieved)]


def cite(hits: list[dict], k: int = 1) -> list[dict]:
    """Citation(s) for the extension: the k top-ranked distinct statutes (id, citation string, section path)."""
    return [{"statute_idx": h["statute_idx"], "citation": h["citation"], "path": h["path"], "score": h["score"]}
            for h in top_statute_hits(hits, k)]
