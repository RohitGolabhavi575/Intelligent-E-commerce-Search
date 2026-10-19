# Intelligent E-commerce Search & Recommendation Engine

A hybrid information retrieval system for fashion e-commerce that combines **keyword-based search, probabilistic ranking, and semantic similarity** to retrieve relevant products from a product catalog.

The project explores how different retrieval strategies can complement one another to improve product discovery, particularly when a customer's search query does not exactly match the wording of a product title or description.

The system exposes its search functionality through a REST API built with FastAPI and provides a Docker-based environment for reproducible deployment.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Search Methodology](#search-methodology)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Running with Docker](#running-with-docker)
- [API Documentation](#api-documentation)
- [Example Search Queries](#example-search-queries)
- [Design Decisions](#design-decisions)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Acknowledgements](#acknowledgements)

## Overview

Traditional product search systems often rely heavily on lexical matching. While effective for exact terms, they may struggle when users express their intent differently from the language used in product listings.

For example, a user searching for *"comfortable red summer shorts"* may benefit from products whose titles or descriptions contain related terms, even when the exact query phrase is absent.

This project implements a hybrid retrieval pipeline that combines three complementary approaches:

1. **TF-IDF** — identifies lexical relevance using weighted term frequencies.
2. **BM25** — ranks documents using a probabilistic relevance scoring function.
3. **Semantic Search** — uses dense vector embeddings to identify products with semantically related text.

The ranked results from these methods are combined using **Reciprocal Rank Fusion (RRF)** to produce a unified list of candidate products.

The retrieval system is exposed through a FastAPI application, making it accessible to other applications and suitable for containerized deployment.

## Key Features

- **Hybrid retrieval:** Combines lexical, probabilistic, and semantic search.
- **Semantic matching:** Uses pretrained sentence embeddings to capture textual similarity beyond exact keyword overlap.
- **Rank fusion:** Combines rankings from multiple retrieval systems using Reciprocal Rank Fusion.
- **Product catalog preprocessing:** Cleans product data and prepares a consistent text representation for indexing.
- **REST API:** Provides endpoints for product search, individual product lookup, and service health checks.
- **Docker support:** Packages the application runtime and dependencies into a portable container image.
- **Interactive API documentation:** Supports endpoint exploration and testing through FastAPI's Swagger UI.

## System Architecture

```text
                  Client / User
                       |
                       v
                 FastAPI REST API
                       |
                       v
                  SearchService
                       |
                       v
              Hybrid Search Pipeline
                       |
          +------------+------------+
          |            |            |
          v            v            v
        TF-IDF        BM25      Semantic Search
          |            |            |
          +------------+------------+
                       |
                       v
             Reciprocal Rank Fusion
                       |
                       v
              Ranked Product Results
                       |
                       v
                    Response
```

The application also supports retrieving an individual product by its product identifier.

### Main Components

| Component | Responsibility |
|---|---|
| Data preprocessing | Cleans product records and prepares searchable text |
| TF-IDF retrieval | Finds products based on weighted lexical similarity |
| BM25 retrieval | Ranks products using a probabilistic term-matching approach |
| Semantic retrieval | Compares query and product embeddings |
| Reciprocal Rank Fusion | Combines rankings from the individual retrieval systems |
| Search service | Coordinates retrieval and product lookup operations |
| FastAPI | Exposes the search functionality through HTTP endpoints |
| Docker | Provides a consistent application runtime |

## Search Methodology

### 1. Product Text Preparation

Product information is transformed into a searchable text representation using fields such as:

- Product title
- Brand
- Category and subcategory
- Description
- Relevant product attributes

The processed catalog contains a `search_text_clean` field used by the lexical retrieval components.

Preparing a consistent text representation allows different retrieval methods to operate on the same underlying product information.

### 2. TF-IDF Retrieval

TF-IDF (Term Frequency–Inverse Document Frequency) assigns importance to terms based on their frequency within a document and how common they are across the collection.

The implementation uses scikit-learn's `TfidfVectorizer` with:

- Lowercased text
- English stop-word removal
- Unigrams and bigrams
- Minimum document frequency filtering
- Sublinear term-frequency scaling

This component provides a lightweight lexical retrieval baseline.

### 3. BM25 Retrieval

BM25 is a probabilistic information retrieval function widely used for ranking documents against text queries.

The implementation uses the `rank_bm25` library with tokenized product text.

BM25 complements TF-IDF by using term-frequency saturation and document-length normalization when scoring candidate products.

### 4. Semantic Search

Semantic retrieval represents text as dense numerical vectors, allowing the system to compare the meaning-related patterns learned by a pretrained embedding model.

The project uses:

**Embedding model:** `BAAI/bge-small-en-v1.5`

The model produces 384-dimensional embeddings. Product embeddings are precomputed and stored in a NumPy array, while incoming queries are encoded at search time.

The implementation compares normalized embeddings using dot-product similarity, equivalent to cosine similarity for unit-normalized vectors.

This approach can retrieve semantically related products even when the query and product text do not share all their exact terms.

### 5. Reciprocal Rank Fusion

The individual retrieval methods produce separate ranked lists. Their raw scores are not directly interchangeable because each method uses a different scoring mechanism.

Instead of combining those raw scores directly, the system uses Reciprocal Rank Fusion.

For a product \(d\), the fused score is:

\[
\operatorname{RRF}(d)=
\sum_{r \in R}
\frac{1}{k+\operatorname{rank}_r(d)}
\]

Where:

- \(R\) is the set of retrieval rankings.
- \(\operatorname{rank}_r(d)\) is the rank of product \(d\) in ranking \(r\).
- \(k\) is a rank-smoothing constant, configured as 60 in the current implementation.

Products ranked highly by multiple retrieval methods can receive stronger combined scores.

The current pipeline uses a candidate depth of 20 for the retrieval components before rank fusion.

## Technology Stack

| Area | Technologies |
|---|---|
| Language | Python |
| Data processing | Pandas, NumPy |
| Lexical retrieval | scikit-learn, TF-IDF |
| Probabilistic retrieval | rank-bm25 |
| Semantic retrieval | Sentence Transformers |
| Embedding model | BAAI/bge-small-en-v1.5 |
| API framework | FastAPI |
| API server | Uvicorn |
| Containerization | Docker, Docker Compose |
| API exploration | Swagger UI |

## Dataset

The project uses the [Flipkart Fashion Products Dataset](https://www.kaggle.com/datasets/aaditshukla/flipkart-fasion-products-dataset) available on Kaggle.

The dataset contains product information such as titles, brands, categories, descriptions, prices, ratings, product identifiers, and image references.

The original dataset contained approximately 30,000 records. After preprocessing and duplicate handling, the working processed catalog contains **24,835 product records**.

Preprocessing included:

- Handling missing and inconsistent values.
- Correcting the identified column-mapping issue between the description and discount fields.
- Removing repeated product identifiers.
- Conservatively identifying and removing exact image-fingerprint duplicates.
- Preparing searchable product text.

The processed catalog and precomputed embeddings are external application assets rather than files copied into the Docker image.

## Project Structure

The following represents the expected high-level structure. Some supporting files may differ depending on the final repository contents.

```text
Intelligent-E-commerce-Search/
│
├── src/
│   ├── api/
│   │   └── main.py
│   ├── search/
│   │   ├── tfidf_search.py
│   │   ├── bm25_search.py
│   │   ├── semantic_search.py
│   │   └── rrf_search.py
│   └── services/
│       └── search_service.py
│
├── data/
│   └── processed/
│   │   └── products_clean.csv
│   └── embeddings/
│       └── product_embeddings_bge_small.npy
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── requirements.txt
└── README.md
```

The search module filenames above describe the intended component organization; use the actual filenames present in the repository if they differ.

## Getting Started

### Prerequisites

For local execution, install:

- Python 3.11
- Git
- The required Python dependencies
- The processed product catalog
- The precomputed product embeddings

For containerized execution, install Docker Desktop or another compatible Docker environment with Docker Compose support.

### 1. Clone the repository

```bash
git lfs clone https://github.com/RohitGolabhavi575/Intelligent-E-commerce-Search
cd Intelligent E-commerce Search
```
LFS-Large file storage is used to compress large data files in order to store in git; make sure to install, if not  done already: _git lfs install_.

### 2. Prepare the required assets

Ensure the following files exist in the project root:

```text
data/processed/products_clean.csv
data/embeddings//product_embeddings_bge_small.npy
```

The filenames and locations must match the paths configured by the application.

If the assets are not included in the repository, obtain them through the documented download or preprocessing procedure.

### 3. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Start the API

From the project root:

```bash
uvicorn src.api.main:app --reload
```

The API should be available at:

- API base URL: `http://127.0.0.1:8000`
- Interactive API documentation: `http://127.0.0.1:8000/docs`
- Health endpoint: `http://127.0.0.1:8000/health`

The semantic embedding model may need to be downloaded on its first use, depending on the local cache and model-loading configuration.

## Running with Docker

Docker provides a repeatable way to build and run the API without manually installing the application's Python dependencies on the host.

### 1. Build and start the application

From the project root:

```bash
docker compose up --build
```

Docker builds the image according to the `Dockerfile` and starts the service defined in `docker-compose.yml`.

The current configuration exposes port 8000 and mounts the local `data/` and `models/` directories into the container as read-only directories.

### 2. Access the API

Open:

- `http://localhost:8000/health`
- `http://localhost:8000/docs`

### 3. Stop the application

```bash
docker compose down
```

### How Docker is used in this project

The Docker image contains the application source code and its Python runtime dependencies.

The product dataset and precomputed embeddings are supplied through bind mounts:

```yaml
volumes:
  - ./data:/app/data:ro
  - ./models:/app/models:ro
```

This keeps the data and embedding files outside the image while making them accessible to the application inside the container.

Consequently, anyone running the project needs to obtain these assets and place them in the expected locations. Docker does not automatically transfer files from the original developer's machine.

The pretrained embedding model may also be downloaded into the container's Hugging Face cache if it is not already available. Its availability and persistence depend on the model-loading configuration.

## API Documentation

FastAPI automatically generates interactive documentation for the application.

Visit:

```text
http://localhost:8000/docs
```

### Available Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Checks API health |
| GET | `/search` | Searches the product catalog |
| GET | `/products/{pid}` | Retrieves an individual product by product identifier |

### Search Products

Example request:

```http
GET /search?q=red%20shorts&top_k=5
```

Equivalent URL:

```text
http://localhost:8000/search?q=red%20shorts&top_k=5
```

Parameters:

| Parameter | Description |
|---|---|
| `q` | Search query |
| `top_k` | Maximum number of results requested |

The response contains ranked product results from the hybrid retrieval pipeline. Consult `/docs` for the exact response schema and parameter validation implemented by the current API.

### Retrieve a Product

```http
GET /products/{pid}
```

Replace `{pid}` with a product identifier from the processed catalog.

### Health Check

```http
GET /health
```

This endpoint can be used to check whether the API is responding. It should not be interpreted as a complete validation that every model and data asset is loaded successfully unless the implementation explicitly checks those components.

## Example Search Queries

Try queries such as:

- `red shorts`
- `black printed t-shirt`
- `women floral dress`
- `casual running shoes`
- `blue denim jeans`

These examples illustrate different types of product-search intent, including color, category, style, and use case.

Actual retrieval quality depends on the product catalog, text preprocessing, embedding model, and ranking configuration.

## Design Decisions

### Why hybrid retrieval?

Lexical and semantic retrieval solve related but different problems.

- TF-IDF and BM25 provide strong signals when query terms overlap with product text.
- Semantic search helps identify products based on learned textual similarity.
- Reciprocal Rank Fusion combines ranked lists without requiring the raw scores from different retrieval methods to be calibrated to the same scale.

The hybrid approach offers a practical foundation for product discovery while keeping each retrieval component independently understandable.

### Why precompute product embeddings?

Product text does not need to be encoded again for every search request when the catalog remains unchanged.

Precomputing embeddings reduces repeated work during inference. The query is encoded at search time and compared against the stored product vectors.

When the catalog or embedding model changes, the corresponding product embeddings must be regenerated to maintain consistency.

### Why FastAPI?

FastAPI provides a lightweight interface for exposing Python-based retrieval functionality through HTTP endpoints. It also generates interactive API documentation, making it straightforward to test the application and integrate it with other clients.

### Why Docker?

Docker packages the application runtime and dependencies into a consistent environment. This makes the application easier to reproduce on another machine and provides a foundation for deployment.

The current setup deliberately keeps the catalog and embeddings outside the image to avoid unnecessarily packaging these assets into every image build.

## Limitations and Future Improvements

Potential next steps include:

- **Search evaluation:** Expand relevance judgments and evaluate retrieval quality using metrics such as Recall@K, Precision@K, and NDCG@K.
- **Filtering:** Support filters for price, brand, category, rating, and availability.
- **Ranking improvements:** Explore learned ranking approaches using query-product relevance features.
- **Recommendation functionality:** Extend search into related-product recommendations or personalized discovery.
- **Operational improvements:** Add structured logging, automated tests, request validation, and monitoring.
- **Deployment:** Publish a container image and deploy the API to a cloud environment.
- **Data and model management:** Version the catalog, embedding artifacts, and embedding model configuration.

These are possible extensions rather than features claimed to be implemented in the current version.

## Acknowledgements

- **Dataset:** [Flipkart Fashion Products Dataset — Kaggle](https://www.kaggle.com/datasets/aaditshukla/flipkart-fasion-products-dataset)
- **Embedding model:** [BAAI/bge-small-en-v1.5 — Hugging Face](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **API framework:** [FastAPI](https://fastapi.tiangolo.com/)
- **Machine learning library:** [scikit-learn](https://scikit-learn.org/)
- **Containerization:** [Docker](https://www.docker.com/)

## Snapshots
<img width="960" height="540" alt="image" src="https://github.com/user-attachments/assets/fff0e1cd-c740-4d78-96f8-effb47988c73" />
<img width="960" height="540" alt="image" src="https://github.com/user-attachments/assets/4544d1b3-f763-405a-a3b8-a06ec228a427" />

