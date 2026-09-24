from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mteb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from preprocess import clean_query, clean_snippet


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

NUM_QUERIES = 100
TOP_K = 10


def tokenize(text):
    import re
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\S", text)


def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def main():

    print("Loading AppsRetrieval...")

    task = mteb.get_task("AppsRetrieval")
    task.load_data()

    test_data = task.dataset["default"]["test"]

    corpus = test_data["corpus"]
    queries = test_data["queries"]
    qrels = test_data["relevant_docs"]

    corpus_ids = [
        item["id"]
        for item in corpus
    ]

    corpus_texts = [
        clean_snippet(item["text"])
        for item in corpus
    ]

    selected_queries = queries.select(
        range(NUM_QUERIES)
    )

    query_ids = [
        item["id"]
        for item in selected_queries
    ]

    query_texts = [
        clean_query(item["text"])
        for item in selected_queries
    ]

    print("\nLoading MiniLM...")

    model = SentenceTransformer(MODEL_NAME)

    print("Encoding corpus...")

    dense_embeddings = model.encode(
        corpus_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    dense_embeddings = normalize(dense_embeddings)

    print("\nBuilding BM25...")

    bm25 = BM25Okapi([
        tokenize(text)
        for text in corpus_texts
    ])

    print("\nEncoding queries...")

    query_embeddings = model.encode(
        query_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    query_embeddings = normalize(query_embeddings)

    # Counters
    dense_found = 0
    bm25_found = 0
    both_found = 0
    neither_found = 0

    high_overlap = 0
    medium_overlap = 0
    low_overlap = 0

    # Store useful cases
    useful_bm25 = []
    harmful_bm25 = []

    print("\nAnalyzing retrieval signals...")

    for row, query_id in enumerate(query_ids):

        query = query_texts[row]
        relevant_docs = qrels.get(query_id, {})

        # -----------------------------------------
        # Dense top 10
        # -----------------------------------------

        dense_scores = (
            dense_embeddings @ query_embeddings[row]
        )

        dense_top_indices = np.argsort(
            -dense_scores
        )[:TOP_K]

        dense_top_ids = {
            corpus_ids[i]
            for i in dense_top_indices
        }

        # -----------------------------------------
        # BM25 top 10
        # -----------------------------------------

        bm25_scores = np.array(
            bm25.get_scores(
                tokenize(query)
            )
        )

        bm25_top_indices = np.argsort(
            -bm25_scores
        )[:TOP_K]

        bm25_top_ids = {
            corpus_ids[i]
            for i in bm25_top_indices
        }

        # -----------------------------------------
        # Relevant documents
        # -----------------------------------------

        relevant_ids = set(relevant_docs.keys())

        dense_hit = bool(
            dense_top_ids & relevant_ids
        )

        bm25_hit = bool(
            bm25_top_ids & relevant_ids
        )

        if dense_hit:
            dense_found += 1

        if bm25_hit:
            bm25_found += 1

        if dense_hit and bm25_hit:
            both_found += 1

        if not dense_hit and not bm25_hit:
            neither_found += 1

        # -----------------------------------------
        # Top-10 overlap
        # -----------------------------------------

        overlap = len(
            dense_top_ids & bm25_top_ids
        )

        if overlap >= 7:
            high_overlap += 1
            overlap_type = "HIGH"

        elif overlap >= 3:
            medium_overlap += 1
            overlap_type = "MEDIUM"

        else:
            low_overlap += 1
            overlap_type = "LOW"

        # -----------------------------------------
        # Find BM25-only useful cases
        # -----------------------------------------

        if bm25_hit and not dense_hit:

            useful_bm25.append({
                "query_id": query_id,
                "query": query.replace("\n", " ")[:220],
                "overlap": overlap,
            })

        # -----------------------------------------
        # Find Dense-only cases
        # -----------------------------------------

        if dense_hit and not bm25_hit:

            harmful_bm25.append({
                "query_id": query_id,
                "query": query.replace("\n", " ")[:220],
                "overlap": overlap,
            })

    # -----------------------------------------
    # Print summary
    # -----------------------------------------

    print()

    print("=" * 65)
    print("RETRIEVAL SIGNAL ANALYSIS")
    print("=" * 65)

    print("Queries analyzed :", NUM_QUERIES)

    print()
    print("Top-10 relevance coverage")
    print("  Dense found relevant :", dense_found)
    print("  BM25 found relevant  :", bm25_found)
    print("  Both found relevant  :", both_found)
    print("  Neither found        :", neither_found)

    print()
    print("Dense/BM25 Top-10 overlap")
    print("  High overlap (7-10) :", high_overlap)
    print("  Medium overlap (3-6):", medium_overlap)
    print("  Low overlap (0-2)   :", low_overlap)

    print()
    print("Potentially useful BM25 cases")
    print("  BM25 found relevant, Dense did not:",
          len(useful_bm25))

    print()
    print("Potentially risky BM25 cases")
    print("  Dense found relevant, BM25 did not:",
          len(harmful_bm25))

    print()
    print("=" * 65)
    print("BM25-ONLY USEFUL EXAMPLES")
    print("=" * 65)

    for case in useful_bm25[:5]:
        print()
        print("Query ID:", case["query_id"])
        print("Overlap :", case["overlap"])
        print("Query   :", case["query"])

    print()
    print("=" * 65)
    print("DENSE-ONLY EXAMPLES")
    print("=" * 65)

    for case in harmful_bm25[:5]:
        print()
        print("Query ID:", case["query_id"])
        print("Overlap :", case["overlap"])
        print("Query   :", case["query"])

    print()
    print("=" * 65)


if __name__ == "__main__":
    main()