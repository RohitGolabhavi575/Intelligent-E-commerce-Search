import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"

class SemanticSearch:

    def __init__(
        self,
        products,
        embedding_path,
        model_name = MODEL_NAME
    ):

        self.products = products

        self.model = SentenceTransformer(
            model_name
        )

        self.product_embeddings = np.load(
            embedding_path
        )

    def search(self, query, top_k=5):
        
        if not query or not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        # Because embeddings are normalized,
        # dot product = cosine similarity.
        scores = np.dot(
            self.product_embeddings,
            query_embedding
        )

        top_indices = np.argsort(
            scores
        )[::-1][:top_k]

        return self.products.iloc[
            top_indices
        ][["pid", "title"]].assign(
            score=scores[top_indices]
        ).reset_index(drop=True)