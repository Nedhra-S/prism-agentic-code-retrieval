import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PRISM GENAI HACKATHON 3.0
# THEME 01 - AGENTIC CODE INTELLIGENCE
#
# P1: VERSION-AWARE RETRIEVAL
#
# Features:
#   1. Retrieve from any versioned code snapshot
#   2. Detect changed/new snippets using content hashes
#   3. Reuse embeddings for unchanged snippets
#   4. Encode only changed/new snippets
#   5. Rebuild a version index efficiently
#   6. Search any selected version
#
# P0 benchmark code is not modified by this file.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CORE_WEIGHT = 0.50
INPUT_WEIGHT = 0.25
OUTPUT_WEIGHT = 0.25

EMBEDDING_DIMENSION = 384

DEFAULT_TOP_K = 5


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_VERSIONS_DIR = (
    PROJECT_ROOT
    / "data"
    / "versions"
)

CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / ".version_cache"
)

EMBEDDING_CACHE_DIR = (
    CACHE_DIR
    / "embeddings"
)

INDEX_CACHE_DIR = (
    CACHE_DIR
    / "indexes"
)

EMBEDDING_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INDEX_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL
# ============================================================

def load_model():
    """
    Load the same dense embedding model family used by
    the benchmarked retrieval pipeline.
    """

    return SentenceTransformer(
        MODEL_NAME
    )


# ============================================================
# JSONL DISCOVERY
# ============================================================

def find_version_file(
    versions_dir: Path,
    version: str
) -> Path:
    """
    Supports either:

        data/versions/v1.jsonl

    or:

        data/versions/v1/snippets.jsonl

    or:

        data/versions/v1.json
    """

    candidates = [
        versions_dir / f"{version}.jsonl",
        versions_dir / f"{version}.json",
        versions_dir / version / "snippets.jsonl",
        versions_dir / version / "snippets.json",
    ]

    for candidate in candidates:

        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No snapshot found for version '{version}'. "
        f"Expected one of: {candidates}"
    )


# ============================================================
# LOAD VERSION
# ============================================================

def load_version(
    versions_dir: Path,
    version: str
):
    """
    Read one version snapshot.

    JSONL is preferred.
    """

    path = find_version_file(
        versions_dir,
        version
    )

    items = []


    if path.suffix.lower() == ".jsonl":

        with path.open(
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                try:

                    item = json.loads(
                        line
                    )

                except json.JSONDecodeError:

                    continue

                if isinstance(
                    item,
                    dict
                ):

                    items.append(item)


    elif path.suffix.lower() == ".json":

        with path.open(
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)


        if isinstance(
            data,
            list
        ):

            items = [
                item
                for item in data
                if isinstance(
                    item,
                    dict
                )
            ]


        elif isinstance(
            data,
            dict
        ):

            possible_items = (
                data.get("snippets")
                or data.get("items")
                or data.get("data")
                or []
            )

            if isinstance(
                possible_items,
                list
            ):

                items = [
                    item
                    for item in possible_items
                    if isinstance(
                        item,
                        dict
                    )
                ]


    if not items:

        raise ValueError(
            f"Version '{version}' contains no valid snippets."
        )


    return path, items


# ============================================================
# CODE EXTRACTION
# ============================================================

def get_code(item):
    """
    Support the common snippet field names used by the project.
    """

    fields = [
        "code",
        "snippet",
        "text",
        "content",
        "source",
    ]

    for field in fields:

        value = item.get(field)

        if isinstance(
            value,
            str
        ) and value.strip():

            return value.strip()


    return ""


# ============================================================
# STABLE SNIPPET ID
# ============================================================

def get_snippet_id(
    item,
    position: int
) -> str:
    """
    Prefer an existing stable identifier.

    If no identifier exists, fall back to a deterministic
    position-based identifier for the version snapshot.
    """

    fields = [
        "id",
        "docid",
        "name",
        "title",
        "function",
    ]

    for field in fields:

        value = item.get(field)

        if value is not None:

            value = str(
                value
            ).strip()

            if value:

                return value


    return f"snippet_{position:06d}"


# ============================================================
# CONTENT HASH
# ============================================================

def content_hash(
    code: str
) -> str:
    """
    SHA-256 hash of the code content.

    This is the key used to determine whether an embedding
    can be reused across versions.
    """

    normalized = code.replace(
        "\r\n",
        "\n"
    ).strip()

    return hashlib.sha256(
        normalized.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# EMBEDDING CACHE PATH
# ============================================================

def embedding_cache_path(
    code_hash: str
) -> Path:

    return (
        EMBEDDING_CACHE_DIR
        / f"{code_hash}.npy"
    )


# ============================================================
# QUERY VIEWS
# ============================================================

def build_query_views(
    query: str
):
    """
    Problem-focused Core / Input / Output views.

    The same fixed weights are used:

        Core   = 0.50
        Input  = 0.25
        Output = 0.25
    """

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
# QUERY EMBEDDING
# ============================================================

def encode_query(
    model,
    query: str
) -> np.ndarray:
    """
    Encode Core, Input and Output separately and fuse them.
    """

    core, input_view, output_view = (
        build_query_views(query)
    )


    vectors = model.encode(
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


    vectors = np.asarray(
        vectors,
        dtype=np.float32
    )


    final_query = (
        CORE_WEIGHT * vectors[0]
        + INPUT_WEIGHT * vectors[1]
        + OUTPUT_WEIGHT * vectors[2]
    )


    norm = np.linalg.norm(
        final_query
    )


    if norm > 0:

        final_query = (
            final_query / norm
        )


    return final_query.astype(
        np.float32
    )


# ============================================================
# BUILD VERSION INDEX
# ============================================================

def build_version_index(
    model,
    versions_dir: Path,
    version: str
):
    """
    Build or rebuild a version-specific index.

    IMPORTANT:

    Unchanged snippets reuse their previously cached embedding.

    Changed or new snippets are embedded only once.

    This directly supports the P1 requirement for rebuilding
    indexes/caches across code versions.
    """

    version_path, items = load_version(
        versions_dir,
        version
    )


    records = []

    for position, item in enumerate(
        items
    ):

        code = get_code(item)

        if not code:
            continue

        snippet_id = get_snippet_id(
            item,
            position
        )

        code_hash = content_hash(
            code
        )

        records.append(
            {
                "snippet_id": snippet_id,
                "content_hash": code_hash,
                "code": code,
                "item": item,
            }
        )


    if not records:

        raise ValueError(
            f"Version '{version}' has no usable code snippets."
        )


    embeddings = []

    reused = 0
    newly_encoded = 0

    missing_codes = []
    missing_indexes = []


    # --------------------------------------------------------
    # REUSE EXISTING EMBEDDINGS
    # --------------------------------------------------------

    for index, record in enumerate(
        records
    ):

        cache_path = (
            embedding_cache_path(
                record["content_hash"]
            )
        )


        if cache_path.exists():

            try:

                vector = np.load(
                    cache_path,
                    allow_pickle=False
                )

                vector = np.asarray(
                    vector,
                    dtype=np.float32
                )


                if (
                    vector.ndim == 1
                    and vector.shape[0]
                    == EMBEDDING_DIMENSION
                ):

                    embeddings.append(
                        vector
                    )

                    reused += 1

                    continue


            except Exception:

                pass


        embeddings.append(
            None
        )

        missing_codes.append(
            record["code"]
        )

        missing_indexes.append(
            index
        )


    # --------------------------------------------------------
    # ENCODE ONLY NEW / CHANGED CODE
    # --------------------------------------------------------

    if missing_codes:

        print(
            f"Encoding {len(missing_codes)} "
            f"new/changed snippet(s)...",
            flush=True
        )


        new_vectors = model.encode(
            missing_codes,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )


        new_vectors = np.asarray(
            new_vectors,
            dtype=np.float32
        )


        for vector_index, record_index in enumerate(
            missing_indexes
        ):

            vector = new_vectors[
                vector_index
            ]


            vector = np.asarray(
                vector,
                dtype=np.float32
            )


            embeddings[
                record_index
            ] = vector


            cache_path = (
                embedding_cache_path(
                    records[
                        record_index
                    ]["content_hash"]
                )
            )


            try:

                np.save(
                    cache_path,
                    vector
                )

            except Exception:

                pass


            newly_encoded += 1


    # --------------------------------------------------------
    # FINAL MATRIX
    # --------------------------------------------------------

    matrix = np.vstack(
        embeddings
    ).astype(
        np.float32,
        copy=False
    )


    # --------------------------------------------------------
    # VERSION INDEX
    # --------------------------------------------------------

    version_cache_file = (
        INDEX_CACHE_DIR
        / f"{version}.npz"
    )


    np.savez(
        version_cache_file,
        embeddings=matrix
    )


    # --------------------------------------------------------
    # MANIFEST
    # --------------------------------------------------------

    manifest = {
        "version": version,
        "source_file": str(
            version_path
        ),
        "model": MODEL_NAME,
        "dimension": EMBEDDING_DIMENSION,
        "weights": {
            "core": CORE_WEIGHT,
            "input": INPUT_WEIGHT,
            "output": OUTPUT_WEIGHT,
        },
        "snippet_count": len(records),
        "reused_embeddings": reused,
        "new_or_changed_embeddings": newly_encoded,
        "records": [
            {
                "snippet_id": record[
                    "snippet_id"
                ],
                "content_hash": record[
                    "content_hash"
                ],
            }
            for record in records
        ],
        "created_at": time.time(),
    }


    manifest_file = (
        INDEX_CACHE_DIR
        / f"{version}.json"
    )


    with manifest_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2
        )


    return {
        "version": version,
        "source_file": str(
            version_path
        ),
        "snippet_count": len(records),
        "reused_embeddings": reused,
        "new_or_changed_embeddings": newly_encoded,
        "index_file": str(
            version_cache_file
        ),
        "manifest_file": str(
            manifest_file
        ),
        "matrix": matrix,
        "records": records,
    }


# ============================================================
# LOAD VERSION INDEX
# ============================================================

def load_version_index(
    versions_dir: Path,
    version: str,
    model
):
    """
    Build the index if necessary.

    Then return its current records and embeddings.
    """

    result = build_version_index(
        model,
        versions_dir,
        version
    )

    return (
        result["records"],
        result["matrix"],
        result,
    )


# ============================================================
# SEARCH A VERSION
# ============================================================

def search_version(
    model,
    versions_dir: Path,
    version: str,
    query: str,
    top_k: int = DEFAULT_TOP_K
):
    """
    Retrieve only from the selected code version.
    """

    start = time.perf_counter()


    records, matrix, build_info = (
        load_version_index(
            versions_dir,
            version,
            model
        )
    )


    query_vector = encode_query(
        model,
        query
    )


    scores = (
        matrix @ query_vector
    )


    k = min(
        top_k,
        len(scores)
    )


    if k == 0:

        return {
            "version": version,
            "query": query,
            "results": [],
            "latency_ms": 0.0,
            "build_info": build_info,
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


    results = []


    for rank, index in enumerate(
        indexes,
        start=1
    ):

        index = int(index)

        record = records[index]

        results.append(
            {
                "rank": rank,
                "snippet_id": record[
                    "snippet_id"
                ],
                "score": float(
                    scores[index]
                ),
                "code": record[
                    "code"
                ],
            }
        )


    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000.0


    return {
        "version": version,
        "query": query,
        "results": results,
        "latency_ms": latency_ms,
        "build_info": {
            "snippet_count":
                build_info[
                    "snippet_count"
                ],
            "reused_embeddings":
                build_info[
                    "reused_embeddings"
                ],
            "new_or_changed_embeddings":
                build_info[
                    "new_or_changed_embeddings"
                ],
        },
    }


# ============================================================
# LIST AVAILABLE VERSIONS
# ============================================================

def list_versions(
    versions_dir: Path
):

    versions = set()


    if not versions_dir.exists():

        return []


    # v1.jsonl / v1.json
    for path in versions_dir.glob(
        "*.jsonl"
    ):

        versions.add(
            path.stem
        )


    for path in versions_dir.glob(
        "*.json"
    ):

        versions.add(
            path.stem
        )


    # v1/snippets.jsonl / v1/snippets.json
    for directory in versions_dir.iterdir():

        if not directory.is_dir():
            continue


        if (
            (directory / "snippets.jsonl").exists()
            or
            (directory / "snippets.json").exists()
        ):

            versions.add(
                directory.name
            )


    return sorted(
        versions
    )


# ============================================================
# PRINT SEARCH RESULTS
# ============================================================

def print_results(
    result
):

    print()
    print("=" * 72)
    print("P1 VERSION-AWARE CODE RETRIEVAL")
    print("=" * 72)

    print(
        f"Version : {result['version']}"
    )

    print(
        f"Query   : {result['query']}"
    )

    print()


    info = result.get(
        "build_info",
        {}
    )


    print(
        f"Snippets              : "
        f"{info.get('snippet_count', '-')}"
    )

    print(
        f"Reused embeddings     : "
        f"{info.get('reused_embeddings', '-')}"
    )

    print(
        f"New/changed embeddings: "
        f"{info.get('new_or_changed_embeddings', '-')}"
    )

    print(
        f"Retrieval time        : "
        f"{result['latency_ms']:.2f} ms"
    )

    print()


    for item in result[
        "results"
    ]:

        print(
            f"Rank #{item['rank']} | "
            f"{item['snippet_id']} | "
            f"score={item['score']:.4f}"
        )

        print(
            item["code"]
        )

        print("-" * 72)


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "P1 version-aware code retrieval"
        )
    )


    parser.add_argument(
        "--versions-dir",
        type=str,
        default=str(
            DEFAULT_VERSIONS_DIR
        ),
        help=(
            "Directory containing version snapshots."
        )
    )


    parser.add_argument(
        "--list-versions",
        action="store_true",
        help=(
            "List available code versions."
        )
    )


    parser.add_argument(
        "--version",
        type=str,
        help=(
            "Version to retrieve from."
        )
    )


    parser.add_argument(
        "--query",
        type=str,
        help=(
            "Natural-language programming query."
        )
    )


    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "Number of snippets to return."
        )
    )


    parser.add_argument(
        "--build",
        action="store_true",
        help=(
            "Build/rebuild the selected version index "
            "without searching."
        )
    )


    args = parser.parse_args()


    versions_dir = Path(
        args.versions_dir
    )


    # --------------------------------------------------------
    # LIST
    # --------------------------------------------------------

    if args.list_versions:

        versions = list_versions(
            versions_dir
        )


        print(
            "Available versions:"
        )


        if not versions:

            print(
                "  No versions found."
            )

        else:

            for version in versions:

                print(
                    f"  - {version}"
                )


        return 0


    # --------------------------------------------------------
    # VERSION REQUIRED
    # --------------------------------------------------------

    if not args.version:

        parser.error(
            "--version is required unless "
            "--list-versions is used."
        )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print(
        "Loading embedding model...",
        flush=True
    )

    model_start = time.perf_counter()


    model = load_model()


    model_time = (
        time.perf_counter()
        - model_start
    )


    print(
        f"Model ready in "
        f"{model_time:.2f} seconds.",
        flush=True
    )


    # --------------------------------------------------------
    # BUILD
    # --------------------------------------------------------

    if args.build:

        print(
            f"Building version '{args.version}'...",
            flush=True
        )


        start = time.perf_counter()


        result = build_version_index(
            model,
            versions_dir,
            args.version
        )


        elapsed = (
            time.perf_counter()
            - start
        )


        print()
        print(
            "=" * 72
        )

        print(
            "VERSION INDEX READY"
        )

        print(
            "=" * 72
        )

        print(
            f"Version              : "
            f"{result['version']}"
        )

        print(
            f"Snippets             : "
            f"{result['snippet_count']}"
        )

        print(
            f"Reused embeddings    : "
            f"{result['reused_embeddings']}"
        )

        print(
            f"New/changed encoded  : "
            f"{result['new_or_changed_embeddings']}"
        )

        print(
            f"Index build time     : "
            f"{elapsed:.2f} seconds"
        )

        print(
            f"Index file           : "
            f"{result['index_file']}"
        )

        print(
            f"Manifest             : "
            f"{result['manifest_file']}"
        )

        print(
            "=" * 72
        )

        return 0


    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    if not args.query:

        parser.error(
            "--query is required when "
            "--build is not used."
        )


    result = search_version(
        model=model,
        versions_dir=versions_dir,
        version=args.version,
        query=args.query,
        top_k=args.top_k,
    )


    print_results(
        result
    )


    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )