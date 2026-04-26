from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec
from janome.tokenizer import Tokenizer


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


class JapaneseTokenizer:
    def __init__(self) -> None:
        self._tokenizer = Tokenizer()

    def tokenize(self, text: str) -> list[str]:
        text = _safe_text(text)
        if not text:
            return []
        tokens = [token.surface for token in self._tokenizer.tokenize(text) if token.surface.strip()]
        return tokens or list(text)


def train_fasttext_model(
    corpus: Iterable[list[str]],
    vector_size: int = 128,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 10,
    workers: int = 1,
    log_epochs: bool = True,
) -> FastText:
    sentences = list(corpus)
    model = FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,
        min_n=2,
        max_n=5,
    )
    model.build_vocab(sentences)
    callbacks = [EpochLogger()] if log_epochs else []
    model.train(sentences, total_examples=len(sentences), epochs=epochs, callbacks=callbacks)
    return model


def embed_text(model: FastText, tokens: list[str]) -> np.ndarray:
    if not tokens:
        return np.zeros(model.vector_size, dtype=np.float32)

    vectors = []
    for token in tokens:
        if token:
            vectors.append(model.wv[token])
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(np.asarray(vectors, dtype=np.float32), axis=0)


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


@dataclass
class SearchResult:
    product_id: str
    score: float
    product_title: str
    product_brand: str
    product_description: str


class EpochLogger(CallbackAny2Vec):
    def __init__(self) -> None:
        self.epoch = 0

    def on_epoch_begin(self, model) -> None:  # type: ignore[override]
        self.epoch += 1
        print(f"[train] epoch {self.epoch} start")

    def on_epoch_end(self, model) -> None:  # type: ignore[override]
        print(f"[train] epoch {self.epoch} end")


class VectorSearchEngine:
    def __init__(
        self,
        model: FastText,
        tokenizer: JapaneseTokenizer,
        products: pd.DataFrame,
        build_embeddings: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.products = products.reset_index(drop=True).copy()
        self.embeddings = self._build_product_embeddings() if build_embeddings else np.empty((0, 0), dtype=np.float32)
        self.product_ids = self.products["product_id"].astype(str).tolist()

    def _build_product_embeddings(self) -> np.ndarray:
        vectors = []
        for text in self.products["text_for_training"].tolist():
            tokens = self.tokenizer.tokenize(text)
            vectors.append(embed_text(self.model, tokens))
        return normalize_matrix(np.vstack(vectors).astype(np.float32))

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_tokens = self.tokenizer.tokenize(query)
        query_vector = embed_text(self.model, query_tokens).astype(np.float32)
        norm = np.linalg.norm(query_vector)
        if norm == 0:
            return []
        query_vector = query_vector / norm

        scores = self.embeddings @ query_vector
        top_indices = np.argsort(-scores)[:top_k]
        results: list[SearchResult] = []
        for idx in top_indices:
            row = self.products.iloc[int(idx)]
            results.append(
                SearchResult(
                    product_id=str(row["product_id"]),
                    score=float(scores[idx]),
                    product_title=_safe_text(row.get("product_title")),
                    product_brand=_safe_text(row.get("product_brand")),
                    product_description=_safe_text(row.get("product_description")),
                )
            )
        return results

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(output_dir / "fasttext.model"))
        np.save(output_dir / "product_embeddings.npy", self.embeddings)
        index_db = output_dir / "product_index.sqlite"
        products = self.products.copy()
        for col in products.columns:
            products[col] = products[col].astype("string").fillna("")
        with sqlite3.connect(str(index_db)) as conn:
            conn.execute("PRAGMA journal_mode=MEMORY")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute("PRAGMA temp_store=MEMORY")
            products.to_sql("products", conn, if_exists="replace", index=False, chunksize=5000)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_products_product_id ON products(product_id)")
            conn.commit()
        metadata = {
            "vector_size": self.model.vector_size,
            "num_products": int(len(self.products)),
        }
        (output_dir / "engine_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, output_dir: Path) -> "VectorSearchEngine":
        model_path = output_dir / "fasttext.model"
        if not model_path.exists():
            # Backward compatibility with earlier runs.
            model_path = output_dir / "fasttext.model"
        model = FastText.load(str(model_path))
        index_db = output_dir / "product_index.sqlite"
        if not index_db.exists():
            raise FileNotFoundError(f"Missing product index SQLite: {index_db}")
        with sqlite3.connect(str(index_db)) as conn:
            products = pd.read_sql_query("SELECT * FROM products", conn)
        engine = cls(model=model, tokenizer=JapaneseTokenizer(), products=products, build_embeddings=False)
        saved_embeddings = np.load(output_dir / "product_embeddings.npy")
        engine.embeddings = saved_embeddings.astype(np.float32)
        engine.product_ids = engine.products["product_id"].astype(str).tolist()
        return engine
