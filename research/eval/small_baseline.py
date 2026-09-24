from __future__ import annotations

import sys
import re
from pathlib import Path
from math import log2

import numpy as np
import mteb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"

sys.path.insert(0, str(SRC_DIR))


# Import our three-view query decomposition functions.
from query_views import (
    extract_core,
    extract_input,
    extract_output,
)


# ==========================================================
# SETTINGS
# ==========================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Number of training queries used for weight tuning.
# 1000 is enough for a first development experiment.
TRAIN_QUERY_COUNT = 1000

# Official AppsRetrieval test contains 3765 queries.
TEST_QUERY_COUNT = 3765

TOP_K = 10

BATCH_SIZE = 64


# ==========================================================
# WEIGHT CONFIGURATIONS
# ==========================================================
#
# Format:
# CORE / INPUT / OUTPUT
#
# We use TRAIN data to select the best configuration.
# The selected configuration is then frozen before TEST.
# ==========================================================

WEIGHT_CONFIGS = [

    ("A", 0.50, 0.20, 0.30),

    ("B", 0.40, 0.30, 0.30),

    ("C", 0.33, 0.33, 0.34),

    ("D", 0.50, 0.25, 0.25),

    ("E", 0.25, 0.50, 0.25),

    ("F", 0.25, 0.25, 0.50),

    ("G", 0.45, 0.30, 0.25),

    ("H", 0.35, 0.35, 0.30),
]


# ==========================================================
# TEXT CLEANING
# ==========================================================

def normalize_text(text: str) -> str:
    """
    Normalize repeated whitespace.
    """

    if not text:
        return ""

    text = str(text).strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


# ==========================================================
# EMBEDDING NORMALIZATION
# ==========================================================

def normalize_embeddings(
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    L2-normalize embeddings.

    After normalization, dot product gives cosine similarity.
    """

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    norms[norms == 0] = 1.0

    return (
        embeddings
        / norms
    )


# ==========================================================
# NDCG@10
# ==========================================================

def ndcg_at_10(
    ranked_ids: list[str],
    relevant_docs: dict,
) -> float:
    """
    Calculate NDCG@10 for one query.
    """

    ranked_ids = ranked_ids[:10]

    dcg = 0.0

    for rank, doc_id in enumerate(
        ranked_ids,
        start=1,
    ):

        relevance = relevant_docs.get(
            doc_id,
            0,
        )

        if relevance > 0:

            gain = (
                (2 ** relevance)
                - 1
            )

            dcg += (
                gain
                /
                log2(rank + 1)
            )

    ideal_relevances = sorted(
        [
            relevance
            for relevance in relevant_docs.values()
            if relevance > 0
        ],
        reverse=True,
    )[:10]

    idcg = 0.0

    for rank, relevance in enumerate(
        ideal_relevances,
        start=1,
    ):

        gain = (
            (2 ** relevance)
            - 1
        )

        idcg += (
            gain
            /
            log2(rank + 1)
        )

    if idcg == 0:
        return 0.0

    return dcg / idcg


# ==========================================================
# MRR
# ==========================================================

def reciprocal_rank(
    ranked_ids: list[str],
    relevant_docs: dict,
) -> float:
    """
    Calculate reciprocal rank for one query.
    """

    for rank, doc_id in enumerate(
        ranked_ids,
        start=1,
    ):

        if relevant_docs.get(
            doc_id,
            0,
        ) > 0:

            return 1.0 / rank

    return 0.0


# ==========================================================
# LOAD TRAINING DATA
# ==========================================================

def load_training_data():
    """
    Load the TRAIN query split from the CoIR Apps
    queries/corpus dataset and TRAIN qrels.

    Important:
    apps-queries-corpus uses "queries" and "corpus"
    as SPLITS, not configurations.
    """

    print()
    print("=" * 70)
    print("LOADING TRAINING DATA")
    print("=" * 70)

    # ------------------------------------------------------
    # Load raw query collection.
    # ------------------------------------------------------

    print(
        "Loading Apps queries..."
    )

    queries_dataset = load_dataset(
        "CoIR-Retrieval/apps-queries-corpus",
        split="queries",
    )

    print(
        "Raw queries:",
        len(queries_dataset),
    )

    # ------------------------------------------------------
    # Keep TRAIN queries only.
    # ------------------------------------------------------

    train_queries_dataset = queries_dataset.filter(
        lambda row:
        str(
            row.get(
                "partition",
                "",
            )
        ).lower()
        == "train"
    )

    print(
        "Training queries:",
        len(train_queries_dataset),
    )

    # ------------------------------------------------------
    # Limit development set.
    # ------------------------------------------------------

    train_count = min(
        TRAIN_QUERY_COUNT,
        len(train_queries_dataset),
    )

    train_queries_dataset = (
        train_queries_dataset.select(
            range(train_count)
        )
    )

    # ------------------------------------------------------
    # Convert raw query rows.
    #
    # The raw dataset uses "_id".
    # ------------------------------------------------------

    train_queries = []

    for row in train_queries_dataset:

        query_id = str(
            row.get(
                "_id",
                row.get(
                    "id",
                    "",
                ),
            )
        )

        query_text = str(
            row["text"]
        )

        train_queries.append(
            (
                query_id,
                query_text,
            )
        )

    # ------------------------------------------------------
    # Load TRAIN qrels.
    # ------------------------------------------------------

    print(
        "Loading training relevance judgments..."
    )

    qrels_dataset = load_dataset(
        "CoIR-Retrieval/apps-qrels",
        split="train",
    )

    print(
        "Training qrels:",
        len(qrels_dataset),
    )

    train_qrels = {}

    for row in qrels_dataset:

        query_id = str(
            row["query_id"]
        )

        corpus_id = str(
            row["corpus_id"]
        )

        score = int(
            row["score"]
        )

        if query_id not in train_qrels:

            train_qrels[
                query_id
            ] = {}

        train_qrels[
            query_id
        ][
            corpus_id
        ] = score

    # ------------------------------------------------------
    # Keep only queries that have a qrel.
    # ------------------------------------------------------

    train_queries = [
        item
        for item in train_queries
        if item[0] in train_qrels
    ]

    print(
        "Usable training queries:",
        len(train_queries),
    )

    return (
        train_queries,
        train_qrels,
    )


# ==========================================================
# LOAD CORPUS
# ==========================================================

def load_corpus():
    """
    Load the complete AppsRetrieval corpus.

    The corpus is a separate split of the same
    queries/corpus dataset.
    """

    print()
    print(
        "Loading complete corpus..."
    )

    corpus_dataset = load_dataset(
        "CoIR-Retrieval/apps-queries-corpus",
        split="corpus",
    )

    print(
        "Corpus size:",
        len(corpus_dataset),
    )

    corpus_ids = []

    corpus_texts = []

    for row in corpus_dataset:

        corpus_id = str(
            row.get(
                "_id",
                row.get(
                    "id",
                    "",
                ),
            )
        )

        corpus_text = str(
            row["text"]
        )

        corpus_ids.append(
            corpus_id
        )

        corpus_texts.append(
            normalize_text(
                corpus_text
            )
        )

    return (
        corpus_ids,
        corpus_texts,
    )


# ==========================================================
# LOAD OFFICIAL TEST DATA
# ==========================================================

def load_test_data():
    """
    Load official MTEB AppsRetrieval test data.

    This test set is NOT used for tuning.
    """

    print()
    print(
        "Loading official MTEB test..."
    )

    task = mteb.get_task(
        "AppsRetrieval"
    )

    task.load_data()

    test_data = (
        task.dataset[
            "default"
        ][
            "test"
        ]
    )

    queries = test_data[
        "queries"
    ]

    qrels = test_data[
        "relevant_docs"
    ]

    print(
        "Official test queries:",
        len(queries),
    )

    return (
        queries,
        qrels,
    )


# ==========================================================
# ENCODE CORPUS
# ==========================================================

def encode_corpus(
    model,
    corpus_texts,
):
    """
    Encode the entire corpus ONCE.
    """

    print()
    print(
        "Encoding complete corpus..."
    )

    embeddings = model.encode(
        corpus_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    embeddings = normalize_embeddings(
        embeddings
    )

    return embeddings


# ==========================================================
# CREATE QUERY VIEWS
# ==========================================================

def create_query_views(
    texts,
):
    """
    Create CORE, INPUT and OUTPUT views.
    """

    core_texts = []
    input_texts = []
    output_texts = []

    for text in texts:

        core = extract_core(
            text
        )

        inp = extract_input(
            text
        )

        output = extract_output(
            text
        )

        core_texts.append(
            core
        )

        input_texts.append(
            inp
        )

        output_texts.append(
            output
        )

    return (
        core_texts,
        input_texts,
        output_texts,
    )


# ==========================================================
# ENCODE QUERY VIEWS
# ==========================================================

def encode_views(
    model,
    texts,
    label,
):
    """
    Create and encode three query views.
    """

    (
        core_texts,
        input_texts,
        output_texts,
    ) = create_query_views(
        texts
    )

    print()
    print(
        f"Encoding {label} CORE..."
    )

    core_embeddings = model.encode(
        core_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    core_embeddings = normalize_embeddings(
        core_embeddings
    )

    print(
        f"Encoding {label} INPUT..."
    )

    input_embeddings = model.encode(
        input_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    input_embeddings = normalize_embeddings(
        input_embeddings
    )

    print(
        f"Encoding {label} OUTPUT..."
    )

    output_embeddings = model.encode(
        output_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    output_embeddings = normalize_embeddings(
        output_embeddings
    )

    return (
        core_embeddings,
        input_embeddings,
        output_embeddings,
    )


# ==========================================================
# CALCULATE RETRIEVAL SCORES
# ==========================================================

def calculate_scores(
    corpus_embeddings,
    query_embeddings,
):
    """
    Calculate cosine similarity between query embeddings
    and all corpus embeddings.
    """

    return (
        corpus_embeddings
        @
        query_embeddings.T
    ).T


# ==========================================================
# EVALUATE TRAIN CONFIGURATION
# ==========================================================

def evaluate_weight_configuration(
    corpus_ids,
    corpus_embeddings,
    query_ids,
    qrels,
    core_embeddings,
    input_embeddings,
    output_embeddings,
    core_weight,
    input_weight,
    output_weight,
):
    """
    Evaluate one weight configuration.
    """

    # Calculate all similarities in matrix form.
    core_scores = calculate_scores(
        corpus_embeddings,
        core_embeddings,
    )

    input_scores = calculate_scores(
        corpus_embeddings,
        input_embeddings,
    )

    output_scores = calculate_scores(
        corpus_embeddings,
        output_embeddings,
    )

    combined_scores = (
        core_weight
        * core_scores
        +
        input_weight
        * input_scores
        +
        output_weight
        * output_scores
    )

    ndcg_scores = []
    mrr_scores = []

    for row, query_id in enumerate(
        query_ids
    ):

        relevant_docs = qrels.get(
            query_id,
            {},
        )

        if not relevant_docs:
            continue

        top_indices = np.argpartition(
            -combined_scores[row],
            TOP_K - 1,
        )[:TOP_K]

        top_indices = (
            top_indices[
                np.argsort(
                    -combined_scores[
                        row,
                        top_indices,
                    ]
                )
            ]
        )

        ranked_ids = [
            corpus_ids[index]
            for index in top_indices
        ]

        ndcg_scores.append(
            ndcg_at_10(
                ranked_ids,
                relevant_docs,
            )
        )

        mrr_scores.append(
            reciprocal_rank(
                ranked_ids,
                relevant_docs,
            )
        )

    if not ndcg_scores:
        return (
            0.0,
            0.0,
        )

    return (
        float(
            np.mean(
                ndcg_scores
            )
        ),
        float(
            np.mean(
                mrr_scores
            )
        ),
    )


# ==========================================================
# TRAIN WEIGHT TUNING
# ==========================================================

def tune_weights(
    model,
    corpus_ids,
    corpus_embeddings,
    train_queries,
    train_qrels,
):
    """
    Tune weights using TRAIN data only.
    """

    query_ids = [
        item[0]
        for item in train_queries
    ]

    query_texts = [
        item[1]
        for item in train_queries
    ]

    (
        core_embeddings,
        input_embeddings,
        output_embeddings,
    ) = encode_views(
        model,
        query_texts,
        "TRAIN",
    )

    # ------------------------------------------------------
    # Calculate similarities once.
    # ------------------------------------------------------

    print()
    print(
        "Calculating TRAIN similarities..."
    )

    core_scores = calculate_scores(
        corpus_embeddings,
        core_embeddings,
    )

    input_scores = calculate_scores(
        corpus_embeddings,
        input_embeddings,
    )

    output_scores = calculate_scores(
        corpus_embeddings,
        output_embeddings,
    )

    results = []

    print()
    print(
        "Tuning weight configurations..."
    )

    for (
        name,
        core_weight,
        input_weight,
        output_weight,
    ) in WEIGHT_CONFIGS:

        combined_scores = (
            core_weight
            * core_scores
            +
            input_weight
            * input_scores
            +
            output_weight
            * output_scores
        )

        ndcg_scores = []
        mrr_scores = []

        for row, query_id in enumerate(
            query_ids
        ):

            relevant_docs = train_qrels.get(
                query_id,
                {},
            )

            if not relevant_docs:
                continue

            top_indices = np.argpartition(
                -combined_scores[row],
                TOP_K - 1,
            )[:TOP_K]

            top_indices = (
                top_indices[
                    np.argsort(
                        -combined_scores[
                            row,
                            top_indices,
                        ]
                    )
                ]
            )

            ranked_ids = [
                corpus_ids[index]
                for index in top_indices
            ]

            ndcg_scores.append(
                ndcg_at_10(
                    ranked_ids,
                    relevant_docs,
                )
            )

            mrr_scores.append(
                reciprocal_rank(
                    ranked_ids,
                    relevant_docs,
                )
            )

        mean_ndcg = (
            float(
                np.mean(
                    ndcg_scores
                )
            )
            if ndcg_scores
            else 0.0
        )

        mean_mrr = (
            float(
                np.mean(
                    mrr_scores
                )
            )
            if mrr_scores
            else 0.0
        )

        results.append(
            (
                name,
                core_weight,
                input_weight,
                output_weight,
                mean_ndcg,
                mean_mrr,
            )
        )

    # ------------------------------------------------------
    # Print results.
    # ------------------------------------------------------

    print()
    print(
        "=" * 75
    )

    print(
        "TRAIN WEIGHT-TUNING RESULTS"
    )

    print(
        "=" * 75
    )

    print(
        "Config | CORE | INPUT | OUTPUT | NDCG@10 | MRR"
    )

    print(
        "-" * 75
    )

    for result in results:

        (
            name,
            core_weight,
            input_weight,
            output_weight,
            ndcg,
            mrr,
        ) = result

        print(
            f"{name:6} | "
            f"{core_weight:.2f} | "
            f"{input_weight:.2f}  | "
            f"{output_weight:.2f}   | "
            f"{ndcg:.4f}  | "
            f"{mrr:.4f}"
        )

    # ------------------------------------------------------
    # Select best configuration using NDCG@10.
    # ------------------------------------------------------

    best = max(
        results,
        key=lambda item: item[4],
    )

    print()
    print(
        "BEST TRAIN CONFIGURATION"
    )

    print(
        f"Config : {best[0]}"
    )

    print(
        f"CORE   : {best[1]:.2f}"
    )

    print(
        f"INPUT  : {best[2]:.2f}"
    )

    print(
        f"OUTPUT : {best[3]:.2f}"
    )

    print(
        f"NDCG@10: {best[4]:.4f}"
    )

    print(
        f"MRR    : {best[5]:.4f}"
    )

    print(
        "=" * 75
    )

    return (
        best[1],
        best[2],
        best[3],
    )


# ==========================================================
# FINAL TEST EVALUATION
# ==========================================================

def evaluate_test(
    model,
    corpus_ids,
    corpus_embeddings,
    test_queries,
    test_qrels,
    weights,
):
    """
    Evaluate frozen weights on the complete official
    AppsRetrieval test set.
    """

    (
        core_weight,
        input_weight,
        output_weight,
    ) = weights

    query_count = min(
        TEST_QUERY_COUNT,
        len(test_queries),
    )

    selected_queries = (
        test_queries.select(
            range(query_count)
        )
    )

    query_ids = [
        str(
            item["id"]
        )
        for item in selected_queries
    ]

    query_texts = [
        str(
            item["text"]
        )
        for item in selected_queries
    ]

    # ------------------------------------------------------
    # Original dense baseline
    # ------------------------------------------------------

    print()
    print(
        "Encoding TEST full-query baseline..."
    )

    baseline_embeddings = model.encode(
        query_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    baseline_embeddings = normalize_embeddings(
        baseline_embeddings
    )

    # ------------------------------------------------------
    # Multi-view embeddings
    # ------------------------------------------------------

    (
        core_embeddings,
        input_embeddings,
        output_embeddings,
    ) = encode_views(
        model,
        query_texts,
        "TEST",
    )

    # ------------------------------------------------------
    # Similarity matrices
    # ------------------------------------------------------

    print()
    print(
        "Calculating TEST similarities..."
    )

    dense_scores = calculate_scores(
        corpus_embeddings,
        baseline_embeddings,
    )

    core_scores = calculate_scores(
        corpus_embeddings,
        core_embeddings,
    )

    input_scores = calculate_scores(
        corpus_embeddings,
        input_embeddings,
    )

    output_scores = calculate_scores(
        corpus_embeddings,
        output_embeddings,
    )

    combined_scores = (
        core_weight
        * core_scores
        +
        input_weight
        * input_scores
        +
        output_weight
        * output_scores
    )

    dense_ndcg = []
    dense_mrr = []

    multi_ndcg = []
    multi_mrr = []

    # ------------------------------------------------------
    # Evaluate every test query.
    # ------------------------------------------------------

    print()
    print(
        "Evaluating TEST..."
    )

    for row, query_id in enumerate(
        query_ids
    ):

        relevant_docs = test_qrels.get(
            query_id,
            {},
        )

        if not relevant_docs:
            continue

        # ==================================================
        # ORIGINAL DENSE
        # ==================================================

        dense_top = np.argpartition(
            -dense_scores[row],
            TOP_K - 1,
        )[:TOP_K]

        dense_top = (
            dense_top[
                np.argsort(
                    -dense_scores[
                        row,
                        dense_top,
                    ]
                )
            ]
        )

        dense_ranked_ids = [
            corpus_ids[index]
            for index in dense_top
        ]

        # ==================================================
        # OUR MULTI-VIEW
        # ==================================================

        multi_top = np.argpartition(
            -combined_scores[row],
            TOP_K - 1,
        )[:TOP_K]

        multi_top = (
            multi_top[
                np.argsort(
                    -combined_scores[
                        row,
                        multi_top,
                    ]
                )
            ]
        )

        multi_ranked_ids = [
            corpus_ids[index]
            for index in multi_top
        ]

        # ==================================================
        # Metrics
        # ==================================================

        dense_ndcg.append(
            ndcg_at_10(
                dense_ranked_ids,
                relevant_docs,
            )
        )

        dense_mrr.append(
            reciprocal_rank(
                dense_ranked_ids,
                relevant_docs,
            )
        )

        multi_ndcg.append(
            ndcg_at_10(
                multi_ranked_ids,
                relevant_docs,
            )
        )

        multi_mrr.append(
            reciprocal_rank(
                multi_ranked_ids,
                relevant_docs,
            )
        )

        if (
            (row + 1) % 250
            == 0
        ):

            print(
                f"Processed "
                f"{row + 1}/{query_count} "
                f"test queries"
            )

    # ------------------------------------------------------
    # Averages
    # ------------------------------------------------------

    dense_mean_ndcg = float(
        np.mean(
            dense_ndcg
        )
    )

    dense_mean_mrr = float(
        np.mean(
            dense_mrr
        )
    )

    multi_mean_ndcg = float(
        np.mean(
            multi_ndcg
        )
    )

    multi_mean_mrr = float(
        np.mean(
            multi_mrr
        )
    )

    ndcg_change = (
        multi_mean_ndcg
        -
        dense_mean_ndcg
    )

    mrr_change = (
        multi_mean_mrr
        -
        dense_mean_mrr
    )

    ndcg_percent = (
        (
            ndcg_change
            /
            dense_mean_ndcg
        )
        * 100
        if dense_mean_ndcg != 0
        else 0.0
    )

    mrr_percent = (
        (
            mrr_change
            /
            dense_mean_mrr
        )
        * 100
        if dense_mean_mrr != 0
        else 0.0
    )

    # ------------------------------------------------------
    # Final report
    # ------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "FINAL FULL TEST EVALUATION"
    )

    print(
        "=" * 80
    )

    print(
        f"Test queries evaluated : {query_count}"
    )

    print(
        f"Corpus size             : {len(corpus_ids)}"
    )

    print()

    print(
        "FROZEN TRAIN-SELECTED WEIGHTS"
    )

    print(
        f"CORE   = {core_weight:.2f}"
    )

    print(
        f"INPUT  = {input_weight:.2f}"
    )

    print(
        f"OUTPUT = {output_weight:.2f}"
    )

    print()

    print(
        "ORIGINAL DENSE BASELINE"
    )

    print(
        f"NDCG@10 = {dense_mean_ndcg:.4f}"
    )

    print(
        f"MRR     = {dense_mean_mrr:.4f}"
    )

    print()

    print(
        "OUR MULTI-VIEW METHOD"
    )

    print(
        f"NDCG@10 = {multi_mean_ndcg:.4f}"
    )

    print(
        f"MRR     = {multi_mean_mrr:.4f}"
    )

    print()

    print(
        "CHANGE"
    )

    print(
        f"NDCG@10 = {ndcg_change:+.4f} "
        f"({ndcg_percent:+.2f}%)"
    )

    print(
        f"MRR     = {mrr_change:+.4f} "
        f"({mrr_percent:+.2f}%)"
    )

    print(
        "=" * 80
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print()
    print(
        "=" * 80
    )

    print(
        "TRAIN-TUNED MULTI-VIEW CODE RETRIEVAL"
    )

    print(
        "=" * 80
    )

    # ------------------------------------------------------
    # 1. Training data
    # ------------------------------------------------------

    (
        train_queries,
        train_qrels,
    ) = load_training_data()

    # ------------------------------------------------------
    # 2. Corpus
    # ------------------------------------------------------

    (
        corpus_ids,
        corpus_texts,
    ) = load_corpus()

    # ------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------

    print()
    print(
        "Loading embedding model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    # ------------------------------------------------------
    # 4. Corpus embeddings
    # ------------------------------------------------------

    corpus_embeddings = encode_corpus(
        model,
        corpus_texts,
    )

    # ------------------------------------------------------
    # 5. Tune using TRAIN
    # ------------------------------------------------------

    weights = tune_weights(
        model,
        corpus_ids,
        corpus_embeddings,
        train_queries,
        train_qrels,
    )

    # ------------------------------------------------------
    # 6. Official TEST
    # ------------------------------------------------------

    (
        test_queries,
        test_qrels,
    ) = load_test_data()

    # ------------------------------------------------------
    # 7. Evaluate frozen weights
    # ------------------------------------------------------

    evaluate_test(
        model,
        corpus_ids,
        corpus_embeddings,
        test_queries,
        test_qrels,
        weights,
    )


if __name__ == "__main__":
    main()