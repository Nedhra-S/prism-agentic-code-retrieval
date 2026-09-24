"""
Final Multi-View Code Retrieval Encoder
Compatible with MTEB 2.21.x.

Query:
    Full programming problem
        -> Core View
        -> Input View
        -> Output View
        -> weighted embedding fusion

Frozen weights selected on training data:
    Core   = 0.50
    Input  = 0.25
    Output = 0.25

Corpus:
    Original full code snippet
        -> single embedding
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

try:
    from src.query_views import (
        extract_core,
        extract_input,
        extract_output,
    )
except ModuleNotFoundError:
    from query_views import (
        extract_core,
        extract_input,
        extract_output,
    )


class MultiViewCodeEncoder(AbsEncoder):
    """
    MTEB-compatible multi-view encoder.

    Query:
        Core   = 0.50
        Input  = 0.25
        Output = 0.25

    Corpus:
        Full code = 1.00
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    CORE_WEIGHT = 0.50
    INPUT_WEIGHT = 0.25
    OUTPUT_WEIGHT = 0.25

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        revision: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:

        super().__init__()

        self.model_name = model_name
        self.revision = revision

        print("=" * 72)
        print("LOADING MULTI-VIEW CODE RETRIEVAL ENCODER")
        print("=" * 72)

        print("Embedding model:", self.model_name)

        print(
            "Frozen weights:",
            f"CORE={self.CORE_WEIGHT:.2f},",
            f"INPUT={self.INPUT_WEIGHT:.2f},",
            f"OUTPUT={self.OUTPUT_WEIGHT:.2f}",
        )

        self.model = SentenceTransformer(
            self.model_name,
            revision=self.revision,
            device=device,
        )

        self.model.max_seq_length = 512

        embedding_dim = (
            self.model.get_sentence_embedding_dimension()
        )

        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites={
                "name": self.model_name,
                "revision": self.revision,
                "loader": type(self),
                "embed_dim": embedding_dim,
                "max_tokens": 512,
                "similarity_fn_name": ScoringFunction.COSINE,
                "framework": [
                    "Sentence Transformers",
                    "PyTorch",
                ],
                "model_type": ["dense"],
            }
        )

        print("Encoder ready.")
        print("=" * 72)

    # ============================================================
    # NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize_embeddings(
        embeddings: np.ndarray,
    ) -> np.ndarray:

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        norms = np.maximum(
            norms,
            1e-12,
        )

        return embeddings / norms

    # ============================================================
    # BASIC SENTENCE TRANSFORMER ENCODING
    # ============================================================

    def _encode_texts(
        self,
        texts: list[str],
        batch_size: int = 64,
        **kwargs: Any,
    ) -> np.ndarray:

        if not texts:
            dimension = (
                self.model.get_sentence_embedding_dimension()
            )

            return np.empty(
                (0, dimension),
                dtype=np.float32,
            )

        # Avoid passing MTEB-only arguments to SentenceTransformer.
        allowed_kwargs = {}

        for key in [
            "show_progress_bar",
            "convert_to_numpy",
            "normalize_embeddings",
            "precision",
            "truncate_dim",
        ]:
            if key in kwargs:
                allowed_kwargs[key] = kwargs[key]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            **allowed_kwargs,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    # ============================================================
    # MTEB ENCODE
    # ============================================================

    def encode(
        self,
        inputs,
        *,
        task_metadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        MTEB 2.21.x encoder interface.

        MTEB supplies a DataLoader whose batches contain a "text" field.

        Query path:
            text -> Core/Input/Output -> weighted fusion

        Document path:
            text -> direct embedding
        """

        # --------------------------------------------------------
        # Collect text from MTEB DataLoader
        # --------------------------------------------------------

        texts: list[str] = []

        for batch in inputs:

            if "text" not in batch:
                raise ValueError(
                    "MTEB input batch does not contain a 'text' field."
                )

            for text in batch["text"]:
                texts.append(str(text))

        # --------------------------------------------------------
        # QUERY PATH
        # --------------------------------------------------------

        if prompt_type == PromptType.query:

            print(
                f"Encoding {len(texts)} queries using "
                "Core/Input/Output views..."
            )

            # ----------------------------------------------------
            # Create three focused representations
            # ----------------------------------------------------

            core_views = [
                extract_core(text)
                for text in texts
            ]

            input_views = [
                extract_input(text)
                for text in texts
            ]

            output_views = [
                extract_output(text)
                for text in texts
            ]

            # ----------------------------------------------------
            # Encode Core
            # ----------------------------------------------------

            print("Encoding CORE views...")

            core_embeddings = self._encode_texts(
                core_views,
                batch_size=kwargs.get("batch_size", 64),
            )

            # ----------------------------------------------------
            # Encode Input
            # ----------------------------------------------------

            print("Encoding INPUT views...")

            input_embeddings = self._encode_texts(
                input_views,
                batch_size=kwargs.get("batch_size", 64),
            )

            # ----------------------------------------------------
            # Encode Output
            # ----------------------------------------------------

            print("Encoding OUTPUT views...")

            output_embeddings = self._encode_texts(
                output_views,
                batch_size=kwargs.get("batch_size", 64),
            )

            # ----------------------------------------------------
            # Weighted fusion
            # ----------------------------------------------------

            print("Applying frozen evidence weights...")

            fused_embeddings = (
                self.CORE_WEIGHT * core_embeddings
                + self.INPUT_WEIGHT * input_embeddings
                + self.OUTPUT_WEIGHT * output_embeddings
            )

            # ----------------------------------------------------
            # Normalize after fusion
            # ----------------------------------------------------

            fused_embeddings = self._normalize_embeddings(
                fused_embeddings
            )

            return fused_embeddings

        # --------------------------------------------------------
        # DOCUMENT / CORPUS PATH
        # --------------------------------------------------------

        print(
            f"Encoding {len(texts)} code snippets..."
        )

        corpus_embeddings = self._encode_texts(
            texts,
            batch_size=kwargs.get("batch_size", 64),
        )

        return corpus_embeddings

    # ============================================================
    # OPTIONAL DIRECT QUERY ENCODER
    # ============================================================

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int = 64,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Convenience method for local experiments.

        This is not the primary MTEB path.
        """

        core_views = [
            extract_core(query)
            for query in queries
        ]

        input_views = [
            extract_input(query)
            for query in queries
        ]

        output_views = [
            extract_output(query)
            for query in queries
        ]

        core_embeddings = self._encode_texts(
            core_views,
            batch_size=batch_size,
        )

        input_embeddings = self._encode_texts(
            input_views,
            batch_size=batch_size,
        )

        output_embeddings = self._encode_texts(
            output_views,
            batch_size=batch_size,
        )

        fused_embeddings = (
            self.CORE_WEIGHT * core_embeddings
            + self.INPUT_WEIGHT * input_embeddings
            + self.OUTPUT_WEIGHT * output_embeddings
        )

        return self._normalize_embeddings(
            fused_embeddings
        )

    # ============================================================
    # OPTIONAL DIRECT CORPUS ENCODER
    # ============================================================

    def encode_corpus(
        self,
        corpus: list[Any],
        batch_size: int = 64,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Convenience method for local experiments.
        """

        texts: list[str] = []

        for item in corpus:

            if isinstance(item, str):
                texts.append(item)
                continue

            if isinstance(item, dict):

                title = item.get(
                    "title",
                    "",
                )

                text = item.get(
                    "text",
                    "",
                )

                if not text:
                    text = item.get(
                        "code",
                        "",
                    )

                if not text:
                    text = item.get(
                        "content",
                        "",
                    )

                if title and text:
                    texts.append(
                        f"{title}\n{text}"
                    )

                elif text:
                    texts.append(
                        str(text)
                    )

                elif title:
                    texts.append(
                        str(title)
                    )

                else:
                    texts.append(
                        str(item)
                    )

            else:
                texts.append(
                    str(item)
                )

        return self._encode_texts(
            texts,
            batch_size=batch_size,
        )


# ================================================================
# SMOKE TEST
# ================================================================

if __name__ == "__main__":

    print()
    print("Running Multi-View encoder smoke test...")

    encoder = MultiViewCodeEncoder()

    sample_queries = [
        "Write a Python function to check whether a number is prime.",
        "Read an integer and print the sum of its digits.",
    ]

    embeddings = encoder.encode_queries(
        sample_queries,
        batch_size=2,
    )

    print()
    print("Smoke test successful.")
    print(
        "Query embedding shape:",
        embeddings.shape,
    )