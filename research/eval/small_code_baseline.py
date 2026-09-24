from __future__ import annotations

import numpy as np
import mteb
from model2vec import StaticModel
from math import log2

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from preprocess import clean_query, clean_snippet


MODEL_NAME = "minishlab/potion-code-16M-v2"

NUM_QUERIES = 100
TOP_K = 10


def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


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
    print("Total test queries:", len(queries))
    print("Queries used:", NUM_QUERIES)

    corpus_ids = [
        item["id"]
        for item in corpus
    ]

    corpus_texts = [
        clean_snippet(item["text"])
        for item in corpus
    ]

    print("\nLoading code-oriented model...")

    model = StaticModel.from_pretrained(MODEL_NAME)

    print("\nEncoding code corpus...")

    corpus_embeddings = model.encode(corpus_texts)

    corpus_embeddings = normalize(corpus_embeddings)

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

    query_embeddings = model.encode(query_texts)

    query_embeddings = normalize(query_embeddings)

    print("\nSearching...")

    scores = query_embeddings @ corpus_embeddings.T

    ndcg_scores = []
    mrr_scores = []

    for row, query_id in enumerate(query_ids):

        top_indices = np.argsort(
            -scores[row]
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

    mean_ndcg = np.mean(ndcg_scores)
    mean_mrr = np.mean(mrr_scores)

    print()
    print("=" * 50)
    print("CODE MODEL BASELINE RESULTS")
    print("=" * 50)
    print("Model             :", MODEL_NAME)
    print("Queries evaluated :", NUM_QUERIES)
    print("Corpus size       :", len(corpus))
    print(f"NDCG@10           : {mean_ndcg:.4f}")
    print(f"MRR               : {mean_mrr:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()