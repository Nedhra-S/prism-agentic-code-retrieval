"""P1: retrieval across versions.

Requirement (from the theme guideline): the corpus changes over time
(commits), so the solution must be able to rebuild any index/cache for a
new version "in a reasonable amount of time" — implying incremental, not
full, re-embedding where possible.

Approach here: hash each snippet's content. Only snippets whose hash isn't
already in the cache get re-embedded when a new version is indexed. Each
version gets its own index (list of snippet ids + embedding matrix), built
from the shared embedding cache.

The "Evolutionary Retrieval" bonus (retrieving/ranking across ALL versions
of a snippet, where versions are near-duplicates and thus hard to
distinguish) is NOT implemented — see the TODO in `Snippet` and at the
bottom of this file. Only attempt it after P0 and this P1 baseline are
solid; the guideline explicitly ranks it below both.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Snippet:
    id: str
    text: str
    # TODO (team, bonus only): a `lineage_id` field here, shared across
    # versions of "the same" snippet even as its content/hash changes,
    # is what evolutionary retrieval would need to link versions together.
    file_path: str | None = None


@dataclass
class VersionIndex:
    version: str
    snippet_ids: list[str]
    embeddings: np.ndarray  # shape (n, dim)
    build_seconds: float


class EmbeddingCache:
    """Disk-backed cache: content_hash -> embedding vector.

    Reused across versions so only snippets that actually changed get
    re-embedded. Stored as a single .npz (ids + matrix) plus a small JSON
    index; fine for a few thousand snippets, would want a real vector DB
    (e.g. FAISS on-disk, or sqlite) at bigger scale.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.cache_dir / "cache_index.json"
        self._vecs_path = self.cache_dir / "cache_vectors.npz"
        self._hash_to_row: dict[str, int] = {}
        self._matrix: np.ndarray | None = None
        self._load()

    def _load(self):
        if self._meta_path.exists() and self._vecs_path.exists():
            self._hash_to_row = json.loads(self._meta_path.read_text())
            self._matrix = np.load(self._vecs_path)["embeddings"]

    def _save(self):
        self._meta_path.write_text(json.dumps(self._hash_to_row))
        if self._matrix is not None:
            np.savez_compressed(self._vecs_path, embeddings=self._matrix)

    def missing_hashes(self, hashes: list[str]) -> list[str]:
        return [h for h in hashes if h not in self._hash_to_row]

    def add(self, hashes: list[str], embeddings: np.ndarray):
        """Append newly-computed embeddings for hashes not already cached."""
        if len(hashes) == 0:
            return
        start_row = 0 if self._matrix is None else self._matrix.shape[0]
        if self._matrix is None:
            self._matrix = embeddings
        else:
            self._matrix = np.vstack([self._matrix, embeddings])
        for i, h in enumerate(hashes):
            self._hash_to_row[h] = start_row + i
        self._save()

    def get(self, hash_: str) -> np.ndarray:
        return self._matrix[self._hash_to_row[hash_]]

    def get_many(self, hashes: list[str]) -> np.ndarray:
        rows = [self._hash_to_row[h] for h in hashes]
        return self._matrix[rows]


def build_version_index(
    version: str,
    snippets: list[Snippet],
    cache: EmbeddingCache,
    embed_fn,
) -> VersionIndex:
    """Build (or incrementally update) an index for one corpus version.

    `embed_fn(texts: list[str]) -> np.ndarray` should be your encoder's
    batch-encode function (e.g. wrap PrePostPipelineEncoder.encode).

    Only snippets whose content hash isn't already cached get embedded —
    that's the "rebuild in a reasonable amount of time" requirement.
    """
    start = time.time()

    hashes = [content_hash(s.text) for s in snippets]
    to_embed_idx = [i for i, h in enumerate(hashes) if h in cache.missing_hashes(hashes)]

    if to_embed_idx:
        texts_to_embed = [snippets[i].text for i in to_embed_idx]
        new_embeddings = embed_fn(texts_to_embed)
        new_hashes = [hashes[i] for i in to_embed_idx]
        cache.add(new_hashes, np.asarray(new_embeddings))

    embeddings = cache.get_many(hashes)
    build_seconds = time.time() - start

    return VersionIndex(
        version=version,
        snippet_ids=[s.id for s in snippets],
        embeddings=embeddings,
        build_seconds=build_seconds,
    )


if __name__ == "__main__":
    # Minimal smoke test with a tiny in-memory example. Real embed_fn would
    # come from encoder.PrePostPipelineEncoder — kept decoupled here so
    # this file has no hard mteb dependency.
    def fake_embed_fn(texts):
        # deterministic fake embeddings for a dependency-free smoke test
        return np.array([[hash(t) % 100, len(t)] for t in texts], dtype=float)

    cache = EmbeddingCache(".cache_smoketest")

    v1 = [Snippet(id="a", text="def foo(): pass"), Snippet(id="b", text="def bar(): pass")]
    idx1 = build_version_index("v1", v1, cache, fake_embed_fn)
    print(f"v1: {len(idx1.snippet_ids)} snippets, {idx1.build_seconds:.4f}s (all new)")

    # v2 changes only snippet "b" — "a" should be reused from cache
    v2 = [Snippet(id="a", text="def foo(): pass"), Snippet(id="b", text="def bar(): return 1")]
    idx2 = build_version_index("v2", v2, cache, fake_embed_fn)
    print(f"v2: {len(idx2.snippet_ids)} snippets, {idx2.build_seconds:.4f}s (1 new, 1 reused)")
