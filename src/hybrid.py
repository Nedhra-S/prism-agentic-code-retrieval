"""Local-only experiment harness for hybrid (BM25 + dense) retrieval.

NOT wired into encoder.py / the official MTEB submission. MTEB's evaluate()
does nearest-neighbour search over your encoder's embeddings itself; true
hybrid fusion inside that exact loop needs a custom search/retrieval hook,
which is more mteb-internals plumbing than this scaffold attempts.

Use this instead to answer, on a small local sample: "does combining BM25
with dense embeddings actually beat pure dense search here?" If yes, that's
your team's signal that hybrid search is worth the extra engineering to
wire into the real pipeline (or into demo.py, which the judges will also
look at during the hands-on round even though it's not what MTEB scores).

Expects a JSONL file of snippets: {"id": ..., "text": ...} per line.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from preprocess import clean_query, clean_snippet


def load_snippets(path: str) -> list[dict]:
    snippets = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                snippets.append(json.loads(line))
    return snippets


def _tokenize(text: str) -> list[str]:
    # Simple whitespace/punctuation split — good enough for BM25 over code.
    # TODO (team): consider a code-aware tokenizer (split on camelCase,
    # snake_case, punctuation) if BM25 alone looks weak on identifiers.
    import re

    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\S", text)


class HybridRetriever:
    def __init__(self, snippets: list[dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.snippets = snippets
        self.texts = [clean_snippet(s["text"]) for s in snippets]

        self.bm25 = BM25Okapi([_tokenize(t) for t in self.texts])

        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            self.texts, show_progress_bar=False, convert_to_numpy=True
        )
        # normalize for cosine similarity via dot product
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = self.embeddings / norms

    def _dense_scores(self, query: str) -> np.ndarray:
        q_emb = self.model.encode([clean_query(query)], convert_to_numpy=True)[0]
        q_emb = q_emb / max(np.linalg.norm(q_emb), 1e-8)
        return self.embeddings @ q_emb

    def _bm25_scores(self, query: str) -> np.ndarray:
        return np.array(self.bm25.get_scores(_tokenize(clean_query(query))))

    def search(self, query: str, top_k: int = 10, k_rrf: int = 60) -> list[tuple[dict, float]]:
        """Reciprocal Rank Fusion of BM25 and dense rankings.

        RRF score for an item = sum over each ranker of 1 / (k + rank),
        rank is 1-indexed. k_rrf=60 is the commonly-cited default in the
        RRF literature, not tuned for this dataset — worth sweeping.
        """
        dense = self._dense_scores(query)
        bm25 = self._bm25_scores(query)

        dense_ranks = np.argsort(-dense)
        bm25_ranks = np.argsort(-bm25)

        rrf_scores = np.zeros(len(self.snippets))
        for rank, idx in enumerate(dense_ranks, start=1):
            rrf_scores[idx] += 1.0 / (k_rrf + rank)
        for rank, idx in enumerate(bm25_ranks, start=1):
            rrf_scores[idx] += 1.0 / (k_rrf + rank)

        order = np.argsort(-rrf_scores)[:top_k]
        return [(self.snippets[i], float(rrf_scores[i])) for i in order]


def _demo():
    """Manual smoke test — run `python src/hybrid.py` with a sample file."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--snippets", required=True, help="Path to JSONL file")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if not Path(args.snippets).exists():
        raise SystemExit(f"Snippets file not found: {args.snippets}")

    snippets = load_snippets(args.snippets)
    retriever = HybridRetriever(snippets)
    results = retriever.search(args.query, top_k=args.top_k)

    for rank, (snippet, score) in enumerate(results, start=1):
        print(f"{rank}. [{score:.4f}] id={snippet.get('id')}")
        print(snippet["text"][:200].replace("\n", " "))
        print()


if __name__ == "__main__":
    _demo()
