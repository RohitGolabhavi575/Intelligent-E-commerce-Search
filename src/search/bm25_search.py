import pandas as pd
import re

from rank_bm25 import BM25Okapi

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

class BM25Search:

    def __init__(self, products):
        self.products = products

        # Tokenize every product
        self.tokenized_products = [
            tokenize(text)
            for text in products["search_text_clean"]
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(
            self.tokenized_products
        )

    def search(self, query, top_k=5):
        
        if not query or not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        query_tokens = tokenize(query)

        scores = self.bm25.get_scores(
            query_tokens
        )

        top_indices = scores.argsort()[::-1][:top_k]

        return self.products.iloc[
            top_indices
        ][["pid", "title"]].assign(
            score=scores[top_indices]
        ).reset_index(drop=True)