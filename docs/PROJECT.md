# Project Documentation: FastText Vector Search Engine (E-commerce)

This repo builds a simple vector search engine for Japanese e-commerce product search using a FastText embedding model (trained locally) and exact cosine-similarity retrieval.

## Data Structure

### Key directories and files

- `data/raw/`
- `data/processed/`
- `data/processed/vse.sqlite`
- `data/processed/metadata.json`
- `models/fasttext/`
- `models/fasttext/fasttext.model`
- `models/fasttext/product_embeddings.npy`
- `models/fasttext/product_index.sqlite`
- `models/fasttext/engine_metadata.json`
- `reports/train.json`
- `reports/test.json`


### Processed dataset (SQLite)

`src/vse/data_pipeline.py::prepare_product_database` loads `data/processed/vse.sqlite`.

Tables:

- `products`

Null/NaN/NULL cleaning requirement:

- Before writing to SQLite, every column is converted to strings and any nil/NaN/NULL-like values are replaced with `""` (empty string). This ensures the SQLite dataset has no NULL values.

`products` table fields (subset):

- `product_id`
- `product_title`, `product_brand`, `product_color`
- `product_bullet_point`, `product_description`
- `text_for_training` (concatenation of the fields above)

Train/test split:

- The pipeline assigns a deterministic `split` (`train` / `test`) per `product_id` (hash-based), unless the `products` table already has a valid `split` column.

## Train Method

Entry point: `scripts/train_model.py`

High-level flow:

1. Builds or loads the processed SQLite dataset via `prepare_product_database`.
2. Tokenizes `text_for_training` using `janome` (`JapaneseTokenizer`).
3. Trains a FastText model (gensim) with epoch logs (`EpochLogger`).
4. Builds product embeddings and saves the engine artifacts to `models/fasttext/`.

Training logs include:

- `[1/5] ... [5/5]` step logs
- `[train] epoch N start/end`

## Test Method

Entry point: `scripts/test_search.py`

Modes:

- Ad-hoc search: `--query "サングラス"`
- Evaluation: `--evaluate` (uses held-out `products` rows with `split == "test"`)

Metrics:

- `hit_rate@k`
- `mrr@k`

Output:

- `reports/test.json`

## Vector Search Algorithm

Representation:

- Tokenize text.
- Embed tokens with FastText word vectors.
- Mean-pool token vectors into a single vector.

Index:

- Create a matrix `V` of product vectors.
- L2-normalize each row of `V`.

Query scoring:

- Embed and L2-normalize the query vector `q`.
- Compute cosine similarity with dot product: `scores = V @ q`.

Retrieval:

- Exact top-k via sorting: `argsort(-scores)[:k]`.

## How To Run

Train (also prepares SQLite for training):

```powershell
.\.venv\Scripts\python.exe .\scripts\train_model.py --max-products 20000 --test-fraction 0.2 --epochs 10
```

Evaluate:

```powershell
.\.venv\Scripts\python.exe .\scripts\test_search.py --top-k 10 --evaluate --eval-field product_title --max-eval-queries 500
```

Search:

```powershell
.\.venv\Scripts\python.exe .\scripts\test_search.py --top-k 5 --query "サングラス"
```

## Notes

- This is an unsupervised embedding baseline (trained on product text); it does not directly optimize the labeled relevance data.
- Retrieval is brute-force exact cosine similarity; for large catalogs, replace with an ANN index (HNSW/FAISS) for latency.


## Python Version: 3.12