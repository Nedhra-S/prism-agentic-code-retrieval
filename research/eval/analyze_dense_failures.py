from __future__ import annotations

import re
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


def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def tokenize(text):
    return set(
        re.findall(
            r"[A-Za-z_][A-Za-z0-9_]*",
            text.lower()
        )
    )


def rank_of_relevant_doc(ranking, corpus_ids, relevant_ids):
    relevant_ids = set(relevant_ids)

    for rank, index in enumerate(ranking, start=1):
        if corpus_ids[index] in relevant_ids:
            return rank

    return None


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

    corpus_by_id = {
        item["id"]: item["text"]
        for item in corpus
    }

    selected_queries = queries.select(
        range(NUM_QUERIES)
    )

    print("\nLoading MiniLM...")

    model = SentenceTransformer(MODEL_NAME)

    print("Encoding corpus...")

    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    corpus_embeddings = normalize(
        corpus_embeddings
    )

    print("\nBuilding BM25...")

    bm25 = BM25Okapi([
        list(tokenize(text))
        for text in corpus_texts
    ])

    print("\nEncoding queries...")

    query_texts = [
        clean_query(item["text"])
        for item in selected_queries
    ]

    query_embeddings = model.encode(
        query_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    query_embeddings = normalize(
        query_embeddings
    )

    failures = []

    for row, query_item in enumerate(selected_queries):

        query_id = query_item["id"]
        query = query_texts[row]

        relevant_docs = qrels.get(
            query_id,
            {}
        )

        relevant_ids = list(
            relevant_docs.keys()
        )

        if not relevant_ids:
            continue

        # ------------------------------
        # Dense ranking
        # ------------------------------

        dense_scores = (
            corpus_embeddings
            @ query_embeddings[row]
        )

        dense_ranking = np.argsort(
            -dense_scores
        )

        dense_top10 = dense_ranking[:TOP_K]

        dense_top10_ids = {
            corpus_ids[i]
            for i in dense_top10
        }

        # Only analyze cases where
        # Dense missed all relevant docs.
        if dense_top10_ids & set(relevant_ids):
            continue

        dense_rank = rank_of_relevant_doc(
            dense_ranking,
            corpus_ids,
            relevant_ids
        )

        # ------------------------------
        # BM25 ranking
        # ------------------------------

        bm25_scores = np.array(
            bm25.get_scores(
                list(tokenize(query))
            )
        )

        bm25_ranking = np.argsort(
            -bm25_scores
        )

        bm25_rank = rank_of_relevant_doc(
            bm25_ranking,
            corpus_ids,
            relevant_ids
        )

        # ------------------------------
        # Choose the best relevant code
        # ------------------------------

        best_relevant_id = min(
            relevant_ids,
            key=lambda doc_id: (
                bm25_rank
                if doc_id in corpus_ids[:0]
                else 0
            )
        )

        # Use the first relevant document
        # for simple inspection.
        best_relevant_id = relevant_ids[0]

        relevant_code = corpus_by_id.get(
            best_relevant_id,
            ""
        )

        query_tokens = tokenize(query)
        code_tokens = tokenize(relevant_code)

        overlap = (
            query_tokens & code_tokens
        )

        overlap_count = len(overlap)

        failures.append({
            "query_id": query_id,
            "query": query.replace("\n", " ")[:350],
            "relevant_id": best_relevant_id,
            "dense_rank": dense_rank,
            "bm25_rank": bm25_rank,
            "overlap_count": overlap_count,
            "query_token_count": len(query_tokens),
            "code_token_count": len(code_tokens),
            "overlap_tokens": sorted(overlap)[:20],
        })

    # ------------------------------------------
    # Sort failures by Dense rank
    # ------------------------------------------

    failures.sort(
        key=lambda x: (
            x["dense_rank"]
            if x["dense_rank"] is not None
            else 999999
        )
    )

    # ------------------------------------------
    # Print summary
    # ------------------------------------------

    print()
    print("=" * 75)
    print("DENSE FAILURE ANALYSIS")
    print("=" * 75)

    print(
        "Queries where Dense missed Top-10:",
        len(failures)
    )

    print()
    print("Showing up to 10 failure cases.")

    for i, case in enumerate(
        failures[:10],
        start=1
    ):

        print()
        print(f"CASE {i}")
        print("-" * 75)

        print(
            "Query ID:",
            case["query_id"]
        )

        print(
            "Query:",
            case["query"]
        )

        print(
            "Relevant document:",
            case["relevant_id"]
        )

        print(
            "Dense rank:",
            case["dense_rank"]
        )

        print(
            "BM25 rank:",
            case["bm25_rank"]
        )

        print(
            "Token overlap:",
            case["overlap_count"],
            "/",
            case["query_token_count"]
        )

        print(
            "Shared tokens:",
            ", ".join(
                case["overlap_tokens"]
            )
        )

    print()
    print("=" * 75)


if __name__ == "__main__":
    main()