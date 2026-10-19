import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFSearch:

    def __init__(self, products):
        self.products = products

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )

        self.product_matrix = self.vectorizer.fit_transform(
            products["search_text_rich"]
        )

    def search(self, query, top_k=10):
        
        if not query or not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        query_vector = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            self.product_matrix
        ).flatten()

        top_indices = scores.argsort()[::-1][:top_k]

        return self.products.iloc[
            top_indices
        ][["pid", "title"]].assign(
            score=scores[top_indices]
        ).reset_index(drop=True)