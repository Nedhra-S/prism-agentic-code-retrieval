"""
Quick sanity check for the final Multi-View Code Retrieval encoder.
"""

from __future__ import annotations

import sys
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
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print("MULTI-VIEW ENCODER QUICK CHECK")
    print("=" * 70)

    # --------------------------------------------------------
    # Load encoder
    # --------------------------------------------------------

    print()
    print("Loading encoder...")

    encoder = MultiViewCodeEncoder()

    # --------------------------------------------------------
    # Sample query
    # --------------------------------------------------------

    query = [
        "How can I validate whether the input string contains only digits?"
    ]

    # --------------------------------------------------------
    # Sample code snippets
    # --------------------------------------------------------

    corpus = [
        {
            "text": (
                "def validate_input(s):\n"
                "    return s.isdigit()"
            )
        },
        {
            "text": (
                "def add(a, b):\n"
                "    return a + b"
            )
        },
    ]

    # --------------------------------------------------------
    # Encode query
    # --------------------------------------------------------

    print()
    print("Encoding test query...")

    query_embeddings = encoder.encode_queries(
        query,
        batch_size=2,
    )

    # --------------------------------------------------------
    # Encode corpus
    # --------------------------------------------------------

    print()
    print("Encoding test corpus...")

    corpus_embeddings = encoder.encode_corpus(
        corpus,
        batch_size=2,
    )

    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------

    expected_dimension = 384

    assert query_embeddings.shape == (
        1,
        expected_dimension,
    ), (
        f"Unexpected query shape: "
        f"{query_embeddings.shape}"
    )

    assert corpus_embeddings.shape == (
        2,
        expected_dimension,
    ), (
        f"Unexpected corpus shape: "
        f"{corpus_embeddings.shape}"
    )

    # --------------------------------------------------------
    # Similarity check
    # --------------------------------------------------------

    similarity_scores = (
        query_embeddings @ corpus_embeddings.T
    )

    best_index = int(
        np.argmax(similarity_scores[0])
    )

    print()
    print("Embedding shapes OK:")
    print("Query :", query_embeddings.shape)
    print("Corpus:", corpus_embeddings.shape)

    print()
    print("Similarity scores:")

    for index, score in enumerate(
        similarity_scores[0]
    ):
        print(
            f"  Snippet {index + 1}: "
            f"{float(score):.4f}"
        )

    print()
    print(
        "Best matching snippet:",
        best_index + 1,
    )

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("quick_check passed.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()