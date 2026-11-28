"""Statute chunking for the M3 retrieval index."""

from collections import defaultdict

import pandas as pd

# ~1.45 tokens/word: 3 chunks (~435 tokens) + the question fit Flan-T5's 512 limit.
CHUNK_WORDS = 100
CHUNK_OVERLAP = 20


def clean_text(text: str) -> str:
    return " ".join(text.split())


def chunk_words(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    stride = size - overlap
    return [" ".join(words[i : i + size]) for i in range(0, len(words) - overlap, stride)]


def chunk_statutes(df: pd.DataFrame, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    chunks = []
    for row in df.itertuples(index=False):
        for j, piece in enumerate(chunk_words(clean_text(row.text), size, overlap)):
            chunks.append({
                "chunk_id": f"{row.idx}-{j}",
                "statute_idx": int(row.idx),
                "chunk_no": j,
                "state": row.state,
                "citation": row.citation,
                "path": row.path,
                "text": piece,
            })
    return chunks


def index_text(chunk: dict) -> str:
    # The first path part is the code name, identical for the whole state.
    heading = " ".join(chunk["path"].split("|")[1:])
    return f"{heading} {chunk['text']}"


def find_question_leaks(questions_by_state: dict[str, list[str]], chunks: list[dict]) -> dict[str, list[str]]:
    """Questions found verbatim in a chunk of the same state."""
    texts = defaultdict(list)
    for c in chunks:
        texts[c["state"]].append(c["text"].lower())
    blobs = {state: "\n".join(t) for state, t in texts.items()}
    leaks = {}
    for state, questions in questions_by_state.items():
        blob = blobs.get(state.lower(), "")
        hits = [q for q in questions if " ".join(q.lower().split()) in blob]
        if hits:
            leaks[state] = hits
    return leaks
