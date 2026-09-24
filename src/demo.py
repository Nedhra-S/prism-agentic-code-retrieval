"""
Interactive demo for the final Multi-View Code Retrieval system.

Pipeline:

    Natural-language query
            ↓
    Core / Input / Output views
            ↓
    Multi-view embeddings
            ↓
    0.50 / 0.25 / 0.25 fusion
            ↓
    Similarity retrieval
            ↓
    Ranked code snippets
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT FINAL ENCODER
# ============================================================

from src.encoder import MultiViewCodeEncoder


# ============================================================
# LOAD JSONL SNIPPETS
# ============================================================

def load_snippets(path: str) -> list[dict]:
    """
    Load code snippets from a JSONL file.

    The loader accepts common fields such as:
        text
        code
        content

    It also preserves:
        file
        path
        filename
        id
    when available.
    """

    snippets: list[dict] = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            if not isinstance(item, dict):
                item = {
                    "text": str(item)
                }

            text = (
                item.get("text")
                or item.get("code")
                or item.get("content")
                or ""
            )

            if not text:
                raise ValueError(
                    f"Snippet on line {line_number} "
                    "does not contain text/code/content."
                )

            snippet = dict(item)
            snippet["text"] = str(text)

            snippets.append(snippet)

    if not snippets:
        raise ValueError(
            "No snippets were found in the input file."
        )

    return snippets


# ============================================================
# DISPLAY LABEL
# ============================================================

def get_label(
    snippet: dict,
    index: int,
) -> str:

    for key in (
        "file",
        "path",
        "filename",
        "name",
        "id",
    ):

        value = snippet.get(key)

        if value:
            return str(value)

    return f"snippet_{index + 1}"


# ============================================================
# PREVIEW
# ============================================================

def get_preview(
    text: str,
    max_lines: int = 4,
) -> str:

    lines = text.strip().splitlines()

    if not lines:
        return ""

    preview_lines = lines[:max_lines]

    preview = "\n".join(
        preview_lines
    )

    if len(lines) > max_lines:
        preview += "\n..."

    return preview


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Multi-View Code Retrieval Demo"
        )
    )

    parser.add_argument(
        "--snippets",
        required=True,
        help="Path to JSONL code snippet file.",
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Natural-language programming query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of ranked results to display.",
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError(
            "--top-k must be greater than zero."
        )

    # ========================================================
    # LOAD DATA
    # ========================================================

    print()
    print("=" * 76)
    print("MULTI-VIEW CODE RETRIEVAL DEMO")
    print("=" * 76)

    print()
    print("Loading snippets...")

    snippets = load_snippets(
        args.snippets
    )

    print(
        f"Loaded {len(snippets)} code snippets."
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print("Loading final retrieval encoder...")

    encoder = MultiViewCodeEncoder()

    # ========================================================
    # ENCODE CORPUS
    # ========================================================

    print()
    print("Building code embedding index...")

    start_index = time.perf_counter()

    corpus_embeddings = encoder.encode_corpus(
        snippets,
        batch_size=64,
    )

    index_time = (
        time.perf_counter()
        - start_index
    )

    print(
        f"Index built in {index_time:.2f}s"
    )

    # ========================================================
    # DISPLAY QUERY
    # ========================================================

    print()
    print("Query:")
    print(args.query)

    # ========================================================
    # ENCODE QUERY
    # ========================================================

    start_query = time.perf_counter()

    query_embedding = encoder.encode_queries(
        [args.query],
        batch_size=1,
    )

    # ========================================================
    # RETRIEVE
    # ========================================================

    scores = (
        query_embedding @ corpus_embeddings.T
    )

    scores = np.asarray(
        scores[0],
        dtype=np.float32,
    )

    ranking = np.argsort(
        -scores
    )

    query_time = (
        time.perf_counter()
        - start_query
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    top_k = min(
        args.top_k,
        len(snippets),
    )

    print()
    print(
        f"Top {top_k} Retrieved Code Snippets:"
    )

    print("-" * 76)

    for rank, snippet_index in enumerate(
        ranking[:top_k],
        start=1,
    ):

        snippet = snippets[
            int(snippet_index)
        ]

        score = float(
            scores[snippet_index]
        )

        label = get_label(
            snippet,
            int(snippet_index),
        )

        preview = get_preview(
            str(snippet["text"])
        )

        print(
            f"{rank}. "
            f"[{score:.4f}] "
            f"{label}"
        )

        print(
            f"   {preview}"
        )

        print()

    print("-" * 76)

    print(
        f"Query retrieval time: "
        f"{query_time * 1000:.1f} ms"
    )

    print()
    print("=" * 76)
    print("DEMO COMPLETE")
    print("=" * 76)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()