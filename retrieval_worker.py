import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PRISM GENAI HACKATHON 3.0
# BACKGROUND RETRIEVAL WORKER
# ============================================================


# ------------------------------------------------------------
# PERFORMANCE SETTINGS
# ------------------------------------------------------------

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Use a reasonable number of CPU threads.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_FILE = (
    ROOT
    / "data"
    / "sample_snippets.jsonl"
)

CACHE_DIR = (
    ROOT
    / "data"
    / ".dashboard_cache"
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EMBED_FILE = (
    CACHE_DIR
    / "sample_embeddings.npy"
)

READY_FILE = (
    CACHE_DIR
    / "worker_ready.json"
)


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

MODEL_FILE = (
    "onnx/model_quint8_avx2.onnx"
)


# ------------------------------------------------------------
# FINAL WEIGHTS
# ------------------------------------------------------------

CORE_WEIGHT = 0.50
INPUT_WEIGHT = 0.25
OUTPUT_WEIGHT = 0.25

TOP_K = 5


# ============================================================
# DATA
# ============================================================

def load_data():

    items = []

    if not DATA_FILE.exists():
        return items

    with DATA_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:

                item = json.loads(line)

                if isinstance(item, dict):
                    items.append(item)

            except Exception:
                continue

    return items


# ============================================================
# EXTRACT CODE
# ============================================================

def get_code(item):

    for key in (
        "code",
        "snippet",
        "text",
        "content",
        "source",
    ):

        value = item.get(key)

        if isinstance(value, str):

            if value.strip():

                return value.strip()

    return ""


# ============================================================
# EXTRACT NAME
# ============================================================

def get_name(item, code):

    for key in (
        "name",
        "title",
        "function",
        "id",
    ):

        value = item.get(key)

        if isinstance(value, str):

            if value.strip():

                return value.strip()


    for line in code.splitlines():

        line = line.strip()

        if line.startswith("def "):

            name = (
                line
                .split("(")[0]
                .replace("def ", "")
                .strip()
            )

            return name + "(...)"


        if line.startswith("class "):

            return (
                line
                .split(":")[0]
                .strip()
            )


    return "Code Snippet"


# ============================================================
# BUILD MODEL
# ============================================================

def load_model():

    print(
        "Loading ONNX retrieval model...",
        file=sys.stderr,
        flush=True
    )

    model = SentenceTransformer(
        MODEL_NAME,
        backend="onnx",
        model_kwargs={
            "file_name": MODEL_FILE,
            "provider": "CPUExecutionProvider",
        },
    )

    print(
        "Model loaded.",
        file=sys.stderr,
        flush=True
    )

    return model


# ============================================================
# BUILD / LOAD CORPUS INDEX
# ============================================================

def build_index(model):

    items = load_data()

    valid_items = []
    codes = []

    for item in items:

        code = get_code(item)

        if code:

            valid_items.append(item)
            codes.append(code)


    if not codes:

        return (
            [],
            np.empty(
                (0, 384),
                dtype=np.float32
            )
        )


    # --------------------------------------------------------
    # TRY EXISTING CACHE
    # --------------------------------------------------------

    if EMBED_FILE.exists():

        try:

            cached = np.load(
                EMBED_FILE,
                allow_pickle=False
            )

            if (
                cached.ndim == 2
                and cached.shape[0] == len(codes)
                and cached.shape[1] == 384
            ):

                return (
                    valid_items,
                    cached.astype(
                        np.float32,
                        copy=False
                    )
                )

        except Exception:

            pass


    # --------------------------------------------------------
    # CREATE INDEX
    # --------------------------------------------------------

    print(
        "Creating code embedding index...",
        file=sys.stderr,
        flush=True
    )

    embeddings = model.encode(
        codes,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )


    try:

        np.save(
            EMBED_FILE,
            embeddings
        )

    except Exception:

        pass


    return (
        valid_items,
        embeddings
    )


# ============================================================
# QUERY VIEWS
# ============================================================

def build_views(query):

    query = " ".join(
        query.strip().split()
    )


    core = (
        "Main programming task: "
        + query
    )


    input_view = (
        "Input requirements: "
        + query
    )


    output_view = (
        "Expected output or required behavior: "
        + query
    )


    return (
        core,
        input_view,
        output_view
    )


# ============================================================
# RETRIEVAL
# ============================================================

def search(
    query,
    items,
    corpus,
    model
):

    start = time.perf_counter()


    # --------------------------------------------------------
    # QUERY VIEWS
    # --------------------------------------------------------

    core, input_view, output_view = (
        build_views(query)
    )


    # --------------------------------------------------------
    # ENCODE ONLY THREE QUERY VIEWS
    # --------------------------------------------------------

    query_vectors = model.encode(
        [
            core,
            input_view,
            output_view
        ],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=3,
    )


    query_vectors = np.asarray(
        query_vectors,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # WEIGHTED FUSION
    # --------------------------------------------------------

    final_query = (
        CORE_WEIGHT * query_vectors[0]
        + INPUT_WEIGHT * query_vectors[1]
        + OUTPUT_WEIGHT * query_vectors[2]
    )


    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    norm = np.linalg.norm(
        final_query
    )

    if norm > 0:

        final_query /= norm


    # --------------------------------------------------------
    # FAST COSINE SIMILARITY
    # --------------------------------------------------------

    scores = (
        corpus @ final_query
    )


    # --------------------------------------------------------
    # TOP K
    # --------------------------------------------------------

    k = min(
        TOP_K,
        len(scores)
    )

    if k == 0:

        return {
            "results": [],
            "views": {
                "core": core,
                "input": input_view,
                "output": output_view,
            },
            "latency_ms": 0.0,
        }


    if k == len(scores):

        indexes = np.argsort(
            -scores
        )

    else:

        indexes = np.argpartition(
            -scores,
            k - 1
        )[:k]

        indexes = indexes[
            np.argsort(
                -scores[indexes]
            )
        ]


    # --------------------------------------------------------
    # FORMAT RESULTS
    # --------------------------------------------------------

    results = []

    for rank, index in enumerate(
        indexes,
        start=1
    ):

        index = int(index)

        code = get_code(
            items[index]
        )

        results.append(
            {
                "rank": rank,
                "name": get_name(
                    items[index],
                    code
                ),
                "code": code,
                "score": float(
                    scores[index]
                ),
            }
        )


    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000.0


    return {
        "results": results,
        "views": {
            "core": core,
            "input": input_view,
            "output": output_view,
        },
        "latency_ms": latency_ms,
    }


# ============================================================
# READY FILE
# ============================================================

def write_ready_file(
    corpus_size
):

    payload = {
        "ready": True,
        "corpus_size": corpus_size,
        "model": MODEL_NAME,
        "backend": "onnx",
        "quantization": "quint8_avx2",
        "dimension": 384,
        "timestamp": time.time(),
    }


    temporary_file = (
        CACHE_DIR
        / "worker_ready.tmp"
    )


    try:

        with temporary_file.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                payload,
                file,
                indent=2
            )


        temporary_file.replace(
            READY_FILE
        )

    except Exception:

        pass


# ============================================================
# MAIN WORKER
# ============================================================

def main():

    # Remove stale status.
    try:

        if READY_FILE.exists():
            READY_FILE.unlink()

    except Exception:

        pass


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = load_model()


    # --------------------------------------------------------
    # LOAD / BUILD INDEX
    # --------------------------------------------------------

    items, corpus = build_index(
        model
    )


    # --------------------------------------------------------
    # SIGNAL READY
    # --------------------------------------------------------

    write_ready_file(
        len(items)
    )


    print(
        "RETRIEVAL_WORKER_READY",
        file=sys.stderr,
        flush=True
    )


    # --------------------------------------------------------
    # WAIT FOR QUERIES
    # --------------------------------------------------------

    for line in sys.stdin:

        line = line.strip()

        if not line:
            continue


        # ----------------------------------------------------
        # REQUEST
        # ----------------------------------------------------

        try:

            request = json.loads(
                line
            )

        except Exception:

            response = {
                "error": "Invalid request."
            }

            print(
                json.dumps(response),
                flush=True
            )

            continue


        # ----------------------------------------------------
        # SHUTDOWN
        # ----------------------------------------------------

        if request.get(
            "command"
        ) == "shutdown":

            break


        # ----------------------------------------------------
        # SEARCH
        # ----------------------------------------------------

        if request.get(
            "command"
        ) == "search":

            query = str(
                request.get(
                    "query",
                    ""
                )
            )


            if not query.strip():

                response = {
                    "error":
                    "Query cannot be empty."
                }

            else:

                try:

                    response = search(
                        query,
                        items,
                        corpus,
                        model
                    )

                except Exception as exc:

                    response = {
                        "error": str(exc)
                    }


            print(
                json.dumps(response),
                flush=True
            )

            continue


        # ----------------------------------------------------
        # UNKNOWN COMMAND
        # ----------------------------------------------------

        response = {
            "error": "Unknown command."
        }

        print(
            json.dumps(response),
            flush=True
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()