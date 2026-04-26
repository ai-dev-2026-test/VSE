from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vse.config import ProjectPaths
from vse.data_pipeline import prepare_product_database
from vse.search_engine import JapaneseTokenizer, VectorSearchEngine, train_fasttext_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/build a Japanese FastText vector search engine (local dataset).")
    parser.add_argument("--max-products", type=int, default=20000)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--vector-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def build_corpus(products: pd.DataFrame, tokenizer: JapaneseTokenizer) -> list[list[str]]:
    return [tokenizer.tokenize(text) for text in products["text_for_training"].tolist()]


def main() -> None:
    args = parse_args()
    paths = ProjectPaths(ROOT)
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print("[1/5] Prepare local product DB")
    prepared = prepare_product_database(
        paths=paths,
        max_products=args.max_products,
        test_fraction=args.test_fraction,
    )
    print(
        f"[data] products={len(prepared.products)} train={len(prepared.train_products)} test={len(prepared.test_products)} "
        f"=max_products={args.max_products} test_fraction={args.test_fraction}"
    )
    print(f"[data] sqlite={prepared.sqlite_path}")

    print("[2/5] Tokenize + build training corpus")
    tokenizer = JapaneseTokenizer()
    corpus = build_corpus(prepared.train_products, tokenizer)
    corpus = [tokens for tokens in corpus if tokens]
    print(f"[data] training_sentences={len(corpus)}")

    print("[3/5] Train FastText model (epoch logs enabled)")
    t_train0 = time.perf_counter()
    model = train_fasttext_model(
        corpus=corpus,
        vector_size=args.vector_size,
        epochs=args.epochs,
        workers=args.workers,
        log_epochs=True,
    )
    print(f"[train] done in {time.perf_counter() - t_train0:.2f}s")

    print("[4/5] Build vector index (product embeddings)")
    t_index0 = time.perf_counter()
    engine = VectorSearchEngine(model=model, tokenizer=tokenizer, products=prepared.products)
    print(f"[index] done in {time.perf_counter() - t_index0:.2f}s")

    print("[5/5] Save model + index")
    model_dir = paths.models_dir / f"fasttext"
    engine.save(model_dir)
    print(f"[save] {model_dir}")

    report = {
        "num_products_indexed": int(len(prepared.products)),
        "num_products_train": int(len(prepared.train_products)),
        "num_products_test": int(len(prepared.test_products)),
        "test_fraction": float(args.test_fraction),
        "training_sentences": int(len(corpus)),
        "vector_size": int(args.vector_size),
        "epochs": int(args.epochs),
        "workers": int(args.workers),
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "model_dir": str(model_dir),
        "dataset_mode": "local_files_only",
    }
    report_path = paths.reports_dir / f"train.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
