from __future__ import annotations

import sys
from pathlib import Path
from math import log2

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
RRF_K = 60


def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def tokenize(text):
    import re
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\S", text)


def ndcg_at_10(ranked_ids, relevant_docs):

    ranked_ids = ranked_ids[:10]

    dcg = 0.0

    for rank, doc_id in enumerate(ranked_ids, start=1):

        relevance = relevant_docs.get(doc_id, 0)

        if relevance > 0:
            gain = (2 ** relevance) - 1
            dcg += gain / log2(rank + 1)

    ideal = sorted(
        [r for r in relevant_docs.values() if r > 0],
        reverse=True
    )[:10]

    idcg = 0.0

    for rank, relevance in enumerate(ideal, start=1):

        gain = (2 ** relevance) - 1
        idcg += gain / log2(rank + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def reciprocal_rank(ranked_ids, relevant_docs):

    for rank, doc_id in enumerate(ranked_ids, start=1):

        if relevant_docs.get(doc_id, 0) > 0:
            return 1.0 / rank

    return 0.0


def main():

    print("Loading AppsRetrieval dataset...")

    task = mteb.get_task("AppsRetrieval")
    task.load_data()

    test_data = task.dataset["default"]["test"]

    corpus = test_data["corpus"]
    queries = test_data["queries"]
    qrels = test_data["relevant_docs"]

    print("Corpus size:", len(corpus))
    print("Queries used:", NUM_QUERIES)

    # --------------------------------------------------
    # Prepare corpus
    # --------------------------------------------------

    corpus_ids = [
        item["id"]
        for item in corpus
    ]

    corpus_texts = [
        clean_snippet(item["text"])
        for item in corpus
    ]

    # --------------------------------------------------
    # Load embedding model
    # --------------------------------------------------

    print("\nLoading MiniLM...")

    model = SentenceTransformer(MODEL_NAME)

    # --------------------------------------------------
    # Dense corpus embeddings
    # --------------------------------------------------

    print("Encoding corpus...")

    dense_embeddings = model.encode(
        corpus_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    dense_embeddings = normalize(dense_embeddings)

    # --------------------------------------------------
    # BM25 index
    # --------------------------------------------------

    print("\nBuilding BM25 index...")

    bm25 = BM25Okapi(
        [
            tokenize(text)
            for text in corpus_texts
        ]
    )

    # --------------------------------------------------
    # Queries
    # --------------------------------------------------

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

    print("\nEncoding queries...")

    query_embeddings = model.encode(
        query_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    query_embeddings = normalize(query_embeddings)

    # --------------------------------------------------
    # Counters
    # --------------------------------------------------

    dense_better = 0
    hybrid_better = 0
    same_score = 0

    dense_ndcg_total = 0.0
    hybrid_ndcg_total = 0.0

    dense_mrr_total = 0.0
    hybrid_mrr_total = 0.0

    # --------------------------------------------------
    # Compare each query
    # --------------------------------------------------

    print("\nComparing Dense vs Hybrid...")

    for row, query_id in enumerate(query_ids):

        query = query_texts[row]

        relevant_docs = qrels.get(
            query_id,
            {}
        )

        # ------------------------------
        # Dense ranking
        # ------------------------------

        dense_scores = (
            dense_embeddings
            @ query_embeddings[row]
        )

        dense_indices = np.argsort(
            -dense_scores
        )[:TOP_K]

        dense_ranked_ids = [
            corpus_ids[index]
            for index in dense_indices
        ]

        # ------------------------------
        # BM25 ranking
        # ------------------------------

        bm25_scores = np.array(
            bm25.get_scores(
                tokenize(query)
            )
        )

        bm25_indices = np.argsort(
            -bm25_scores
        )

        # ------------------------------
        # RRF hybrid ranking
        # ------------------------------

        rrf_scores = np.zeros(
            len(corpus)
        )

        # Dense contribution
        dense_full_ranking = np.argsort(
            -dense_scores
        )

        for rank, index in enumerate(
            dense_full_ranking,
            start=1
        ):
            rrf_scores[index] += (
                1.0 / (RRF_K + rank)
            )

        # BM25 contribution
        for rank, index in enumerate(
            bm25_indices,
            start=1
        ):
            rrf_scores[index] += (
                1.0 / (RRF_K + rank)
            )

        hybrid_indices = np.argsort(
            -rrf_scores
        )[:TOP_K]

        hybrid_ranked_ids = [
            corpus_ids[index]
            for index in hybrid_indices
        ]

        # ------------------------------
        # Scores
        # ------------------------------

        dense_ndcg = ndcg_at_10(
            dense_ranked_ids,
            relevant_docs
        )

        hybrid_ndcg = ndcg_at_10(
            hybrid_ranked_ids,
            relevant_docs
        )

        dense_mrr = reciprocal_rank(
            dense_ranked_ids,
            relevant_docs
        )

        hybrid_mrr = reciprocal_rank(
            hybrid_ranked_ids,
            relevant_docs
        )

        dense_ndcg_total += dense_ndcg
        hybrid_ndcg_total += hybrid_ndcg

        dense_mrr_total += dense_mrr
        hybrid_mrr_total += hybrid_mrr

        # ------------------------------
        # Determine winner per query
        # ------------------------------

        if dense_ndcg > hybrid_ndcg:
            dense_better += 1

        elif hybrid_ndcg > dense_ndcg:
            hybrid_better += 1

        else:
            same_score += 1

    # --------------------------------------------------
    # Final averages
    # --------------------------------------------------

    dense_mean_ndcg = (
        dense_ndcg_total / NUM_QUERIES
    )

    hybrid_mean_ndcg = (
        hybrid_ndcg_total / NUM_QUERIES
    )

    dense_mean_mrr = (
        dense_mrr_total / NUM_QUERIES
    )

    hybrid_mean_mrr = (
        hybrid_mrr_total / NUM_QUERIES
    )

    # --------------------------------------------------
    # Print result
    # --------------------------------------------------

    print()

    print("=" * 60)
    print("DENSE vs HYBRID QUERY-BY-QUERY COMPARISON")
    print("=" * 60)

    print("Queries evaluated :", NUM_QUERIES)

    print()
    print("NDCG@10")
    print("  Dense  :", f"{dense_mean_ndcg:.4f}")
    print("  Hybrid :", f"{hybrid_mean_ndcg:.4f}")

    print()
    print("MRR")
    print("  Dense  :", f"{dense_mean_mrr:.4f}")
    print("  Hybrid :", f"{hybrid_mean_mrr:.4f}")

    print()
    print("Per-query comparison using NDCG@10")
    print("  Dense better :", dense_better)
    print("  Hybrid better:", hybrid_better)
    print("  Same          :", same_score)

    print("=" * 60)


if __name__ == "__main__":
    main()