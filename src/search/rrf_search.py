import pandas as pd

class RRFSearch:

    def __init__(
        self,
        tfidf_search,
        bm25_search,
        semantic_search,
        k=60,
        candidate_k=20
    ):
        self.retrievers = {
            "tfidf": tfidf_search,
            "bm25": bm25_search,
            "semantic": semantic_search
        }

        self.k = k
        self.candidate_k = candidate_k

    def search(self, query, top_k=10):

        rrf_scores = {}
        product_info = {}

        for retriever in self.retrievers.values():

            results = retriever.search(
                query,
                top_k=self.candidate_k
            )

            for rank, (_, row) in enumerate(results.iterrows(), start=1):

                pid = row["pid"]

                # RRF contribution
                contribution = 1 / (self.k + rank)

                rrf_scores[pid] = (
                    rrf_scores.get(pid, 0) + contribution
                )

                # Store product information
                if pid not in product_info:
                    product_info[pid] = {
                        "pid": row["pid"],
                        "title": row["title"],
                    }

        results = [
            {
                "pid": pid,
                "title": product_info[pid]["title"],
                "score": score
            }
            for pid, score in rrf_scores.items()
        ]

        results = pd.DataFrame(results)

        results = results.sort_values(
            "score",
            ascending=False
        )

        return results.head(top_k).reset_index(drop=True)