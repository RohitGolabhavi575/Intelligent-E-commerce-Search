import pandas as pd

from src.search.tfidf_search import TFIDFSearch
from src.search.bm25_search import BM25Search
from src.search.semantic_search import SemanticSearch
from src.search.rrf_search import RRFSearch

class SearchService:
    def __init__(self, data_path, embedding_path):
        # Load products
        self.products = pd.read_csv(data_path)
        
        tfidf = TFIDFSearch(
            self.products
        )

        bm25 = BM25Search(
            self.products
        )

        semantic = SemanticSearch(
            self.products,
            embedding_path
        )
        
        self.search_engine = RRFSearch(
            tfidf_search=tfidf,
            bm25_search=bm25,
            semantic_search=semantic,
            k=60,
            candidate_k=20
        )
    

    def search(self, query, top_k=10):

        if not query or not query.strip():
            raise ValueError(
                 "Search query cannot be empty."
            )

        return self.search_engine.search(
             query=query,
             top_k=top_k
        )
    
    
    def get_product(self, pid: str):
        matches = self.products[
            self.products["pid"].astype(str) == pid
        ]

        if matches.empty:
            return None

        product = matches.iloc[0].to_dict()

        return {
            key: (
                None if value is None or (
                    isinstance(value, float) and pd.isna(value)
                ) else value
            )
            for key, value in product.items()
        }
    

search_service = SearchService(
    data_path="data/processed/products_clean2.csv",
    embedding_path="data/embeddings/product_embeddings_bge_small.npy"
)
