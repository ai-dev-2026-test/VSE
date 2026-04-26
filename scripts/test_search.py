from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vse.config import ProjectPaths
from vse.data_pipeline import assign_product_split
from vse.search_engine import VectorSearchEngine, embed_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test/search a saved vector search engine (local-only).")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--query", default="")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval-field", default="product_title")
    parser.add_argument("--max-eval-queries", type=int, default=500)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    return parser.parse_args()


def _pick_eval_text(row: pd.Series, field: str) -> str:
    value = row.get(field, "")
    if isinstance(value, str) and value.strip():
        return value.strip()
    fallback = row.get("text_for_training", "")
    return fallback.strip() if isinstance(fallback, str) else ""


def evaluate(
    engine: VectorSearchEngine,
    products: pd.DataFrame,
    top_k: int,
    eval_field: str,
    max_queries: int,
    test_fraction: float,
) -> dict[str, float]:
    if "split" not in products.columns:
        products = assign_product_split(products, test_fraction=test_fraction)

    eval_products = products.loc[products["split"] == "test"].copy().reset_index(drop=True)
    if max_queries > 0 and len(eval_products) > max_queries:
        eval_products = eval_products.head(max_queries).copy()

    id_to_index = {product_id: idx for idx, product_id in enumerate(engine.product_ids)}

    hits = 0
    reciprocal_ranks: list[float] = []
    tested = 0

    batch_size = 64
    for start in range(0, len(eval_products), batch_size):
        batch = eval_products.iloc[start : start + batch_size]
        queries: list[str] = []
        expected_indices: list[int] = []
        for _, row in batch.iterrows():
            product_id = str(row.get("product_id", ""))
            expected_idx = id_to_index.get(product_id)
            if expected_idx is None:
                continue
            query = _pick_eval_text(row, eval_field)
            if not query:
                continue
            queries.append(query)
            expected_indices.append(int(expected_idx))

        if not queries:
            continue

        query_vectors = []
        for query in queries:
            tokens = engine.tokenizer.tokenize(query)
            vec = embed_text(engine.model, tokens).astype(np.float32)
            norm = float(np.linalg.norm(vec))
            if norm == 0.0:
                vec = np.zeros(engine.model.vector_size, dtype=np.float32)
            else:
                vec = vec / norm
            query_vectors.append(vec)

        q = np.vstack(query_vectors).astype(np.float32)  # (B, D)
        scores = engine.embeddings @ q.T  # (N, B)

        for col, expected_idx in enumerate(expected_indices):
            tested += 1
            col_scores = scores[:, col]
            if top_k <= 0:
                reciprocal_ranks.append(0.0)
                continue
            k = min(top_k, len(col_scores))
            top_indices = np.argpartition(-col_scores, kth=k - 1)[:k]
            top_indices = top_indices[np.argsort(-col_scores[top_indices])]

            rank = 0
            for index, candidate_idx in enumerate(top_indices, start=1):
                if int(candidate_idx) == int(expected_idx):
                    rank = index
                    break

            if rank > 0:
                hits += 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

    return {
        "queries_tested": int(tested),
        f"hit_rate@{top_k}": round(hits / tested, 4) if tested else 0.0,
        f"mrr@{top_k}": round(float(np.mean(reciprocal_ranks)), 4) if reciprocal_ranks else 0.0,
    }


def main() -> None:
    args = parse_args()
    paths = ProjectPaths(ROOT)

    model_dir = paths.models_dir / f"fasttext"
    engine = VectorSearchEngine.load(model_dir)

    report: dict[str, object] = {
        "model_dir": str(model_dir),
        "top_k": int(args.top_k),
    }

    if args.query.strip():
        results = engine.search(args.query, top_k=args.top_k)
        report["query"] = args.query
        report["results"] = [
            {"product_id": r.product_id, "score": round(r.score, 4), "title": r.product_title, "brand": r.product_brand}
            for r in results
        ]

    if args.evaluate:
        products = engine.products.copy()
        report["eval_field"] = args.eval_field
        report["max_eval_queries"] = int(args.max_eval_queries)
        report["metrics"] = evaluate(
            engine,
            products,
            top_k=args.top_k,
            eval_field=args.eval_field,
            max_queries=args.max_eval_queries,
            test_fraction=args.test_fraction,
        )

    out_path = paths.reports_dir / f"test.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
