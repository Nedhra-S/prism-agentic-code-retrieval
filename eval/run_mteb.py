"""
Official MTEB evaluation for the final Multi-View Code Retrieval method.

Method:
    Core   = 0.50
    Input  = 0.25
    Output = 0.25

The weights were selected using training data and are frozen for test
evaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

# Make both project root and src available to Python.
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# ============================================================
# IMPORTS
# ============================================================

import mteb

from src.encoder import MultiViewCodeEncoder


# ============================================================
# SETTINGS
# ============================================================

TASK_NAME = "AppsRetrieval"

BATCH_SIZE = 64

OUTPUT_FOLDER = PROJECT_ROOT / "results" / "multiview_apps"


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 80)
    print("OFFICIAL MTEB EVALUATION")
    print("=" * 80)

    print("Task          :", TASK_NAME)
    print("Method        : Multi-View Code Retrieval")
    print("Core weight   : 0.50")
    print("Input weight  : 0.25")
    print("Output weight : 0.25")
    print("Batch size    :", BATCH_SIZE)
    print("Output folder :", OUTPUT_FOLDER)

    print("=" * 80)
    print()

    # --------------------------------------------------------
    # Load our final encoder
    # --------------------------------------------------------

    print("Loading Multi-View encoder...")
    model = MultiViewCodeEncoder()

    # --------------------------------------------------------
    # Load official MTEB task
    # --------------------------------------------------------

    print()
    print("Loading official AppsRetrieval task...")

    tasks = mteb.get_tasks(
        tasks=[TASK_NAME]
    )

    print("Tasks loaded:", len(tasks))

    for task in tasks:
        print("Task:", task)

    # --------------------------------------------------------
    # Create evaluator
    # --------------------------------------------------------

    print()
    print("Creating MTEB evaluator...")

    evaluation = mteb.MTEB(
        tasks=tasks
    )

    # --------------------------------------------------------
    # Run official test evaluation
    # --------------------------------------------------------

    print()
    print("Starting official MTEB test evaluation...")
    print()

    results = evaluation.run(
        model=model,
        eval_splits=["test"],
        output_folder=str(OUTPUT_FOLDER),
        encode_kwargs={
            "batch_size": BATCH_SIZE
        },
        overwrite_results=True,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("OFFICIAL MTEB EVALUATION COMPLETE")
    print("=" * 80)

    print(results)

    print()
    print("Results saved to:")
    print(OUTPUT_FOLDER)

    print("=" * 80)


if __name__ == "__main__":
    main()