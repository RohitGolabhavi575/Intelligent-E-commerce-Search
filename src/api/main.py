from fastapi import FastAPI, Query, HTTPException
    
from src.services.search_service import SearchService


app = FastAPI(
    title="Intelligent E-commerce Search API",
    description="ML-powered e-commerce product search",
    version="1.0.0"
)


search_service = SearchService(
    data_path="data/processed/products_clean2.csv",
    embedding_path="data/embeddings/product_embeddings_bge_small.npy"
)


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.get("/search")
def search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(10, ge=1, le=50)
):
    results = search_service.search(
        query=q,
        top_k=top_k
    )

    return {
        "query": q,
        "count": len(results),
        "results": results.to_dict(
            orient="records"
        )
    }

@app.get("/products/{pid}")
def get_product(pid: str):
    product = search_service.get_product(pid)

    if product is None:
        raise HTTPException(
            status_code=404,
            detail=f"Product with pid '{pid}' not found"
        )

    return product