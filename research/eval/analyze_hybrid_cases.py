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


def first_relevant_rank(ranked_ids, relevant_docs):

    for rank, doc_id in enumerate(ranked_ids, start=1):

        if relevant_docs.get(doc_id, 0) > 0:
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

    print("Loading MiniLM...")

    model = SentenceTransformer(MODEL_NAME)

    print("Encoding corpus...")

    dense_embeddings = model.encode(
        corpus_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    dense_embeddings = normalize(dense_embeddings)

    print("Building BM25...")

    bm25 = BM25Okapi([
        tokenize(text)
        for text in corpus_texts
    ])

    print("Encoding queries...")

    query_embeddings = model.encode(
        query_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    query_embeddings = normalize(query_embeddings)

    changed_cases = []

    for row, query_id in enumerate(query_ids):

        query = query_texts[row]

        relevant_docs = qrels.get(
            query_id,
            {}
        )

        # -----------------------------
        # Dense ranking
        # -----------------------------

        dense_scores = (
            dense_embeddings
            @ query_embeddings[row]
        )

        dense_ranking = np.argsort(
            -dense_scores
        )

        dense_top10 = dense_ranking[:TOP_K]

        dense_ids = [
            corpus_ids[i]
            for i in dense_top10
        ]

        # -----------------------------
        # BM25 ranking
        # -----------------------------

        bm25_scores = np.array(
            bm25.get_scores(
                tokenize(query)
            )
        )

        bm25_ranking = np.argsort(
            -bm25_scores
        )

        # -----------------------------
        # Hybrid RRF
        # -----------------------------

        rrf_scores = np.zeros(
            len(corpus)
        )

        for rank, index in enumerate(
            dense_ranking,
            start=1
        ):
            rrf_scores[index] += (
                1.0 / (RRF_K + rank)
            )

        for rank, index in enumerate(
            bm25_ranking,
            start=1
        ):
            rrf_scores[index] += (
                1.0 / (RRF_K + rank)
            )

        hybrid_ranking = np.argsort(
            -rrf_scores
        )

        hybrid_top10 = hybrid_ranking[:TOP_K]

        hybrid_ids = [
            corpus_ids[i]
            for i in hybrid_top10
        ]

        # -----------------------------
        # Compare
        # -----------------------------

        dense_ndcg = ndcg_at_10(
            dense_ids,
            relevant_docs
        )

        hybrid_ndcg = ndcg_at_10(
            hybrid_ids,
            relevant_docs
        )

        dense_rank = first_relevant_rank(
            dense_ids,
            relevant_docs
        )

        hybrid_rank = first_relevant_rank(
            hybrid_ids,
            relevant_docs
        )

        if dense_ndcg != hybrid_ndcg:

            # Show only a short portion of
            # the long benchmark question.
            short_query = (
                query.replace("\n", " ")
                .replace("\r", " ")
                [:300]
            )

            changed_cases.append({
                "query_id": query_id,
                "query": short_query,
                "dense_ndcg": dense_ndcg,
                "hybrid_ndcg": hybrid_ndcg,
                "dense_rank": dense_rank,
                "hybrid_rank": hybrid_rank,
                "dense_top": dense_ids[0],
                "hybrid_top": hybrid_ids[0],
            })

    # ---------------------------------
    # Print only changed cases
    # ---------------------------------

    print()
    print("=" * 70)
    print("QUERIES WHERE DENSE AND HYBRID DIFFER")
    print("=" * 70)

    print("Number of changed cases:", len(changed_cases))

    for i, case in enumerate(changed_cases, start=1):

        print()
        print(f"CASE {i}")
        print("-" * 70)

        print("Query ID:", case["query_id"])
        print("Query:", case["query"])

        print(
            "Dense   -> NDCG:",
            f"{case['dense_ndcg']:.4f}",
            "| first relevant rank:",
            case["dense_rank"],
            "| top result:",
            case["dense_top"]
        )

        print(
            "Hybrid  -> NDCG:",
            f"{case['hybrid_ndcg']:.4f}",
            "| first relevant rank:",
            case["hybrid_rank"],
            "| top result:",
            case["hybrid_top"]
        )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()