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

    corpus_ids = [
        item["id"]
        for item in corpus
    ]

    corpus_texts = [
        clean_snippet(item["text"])
        for item in corpus
    ]

    # -----------------------------
    # Dense retrieval
    # -----------------------------

    print("\nLoading MiniLM...")

    model = SentenceTransformer(MODEL_NAME)

    print("Encoding corpus...")

    dense_embeddings = model.encode(
        corpus_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    dense_embeddings = normalize(dense_embeddings)

    # -----------------------------
    # BM25 retrieval
    # -----------------------------

    print("\nBuilding BM25 index...")

    tokenized_corpus = [
        tokenize(text)
        for text in corpus_texts
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    # -----------------------------
    # Queries
    # -----------------------------

    selected_queries = queries.select(range(NUM_QUERIES))

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

    # -----------------------------
    # Search
    # -----------------------------

    print("\nRunning hybrid retrieval...")

    ndcg_scores = []
    mrr_scores = []

    for row, query_id in enumerate(query_ids):

        query = query_texts[row]

        # Dense scores
        dense_scores = dense_embeddings @ query_embeddings[row]

        # BM25 scores
        bm25_scores = np.array(
            bm25.get_scores(tokenize(query))
        )

        # Rankings
        dense_ranked = np.argsort(-dense_scores)
        bm25_ranked = np.argsort(-bm25_scores)

        # RRF
        rrf_scores = np.zeros(len(corpus))

        for rank, index in enumerate(
            dense_ranked,
            start=1
        ):
            rrf_scores[index] += (
                1.0 / (RRF_K + rank)
            )

        for rank, index in enumerate(
            bm25_ranked,
            start=1
        ):
            rrf_scores[index] += (
                1.0 / (RRF_K + rank)
            )

        top_indices = np.argsort(
            -rrf_scores
        )[:TOP_K]

        ranked_ids = [
            corpus_ids[index]
            for index in top_indices
        ]

        relevant_docs = qrels.get(
            query_id,
            {}
        )

        ndcg_scores.append(
            ndcg_at_10(
                ranked_ids,
                relevant_docs
            )
        )

        mrr_scores.append(
            reciprocal_rank(
                ranked_ids,
                relevant_docs
            )
        )

    print()

    print("=" * 55)
    print("HYBRID BASELINE RESULTS")
    print("=" * 55)

    print("Model             :", MODEL_NAME)
    print("Queries evaluated :", NUM_QUERIES)
    print("Corpus size       :", len(corpus))
    print(f"NDCG@10           : {np.mean(ndcg_scores):.4f}")
    print(f"MRR               : {np.mean(mrr_scores):.4f}")

    print("=" * 55)


if __name__ == "__main__":
    main()