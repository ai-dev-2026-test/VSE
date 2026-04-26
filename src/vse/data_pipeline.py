from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import (
    PROCESSED_SQLITE_TEMPLATE,
    ProjectPaths,
)


@dataclass
class ProcessedSQLite:
    products: pd.DataFrame
    sqlite_path: Path
    metadata_path: Path


def _sqlite_paths(paths: ProjectPaths) -> tuple[Path, Path, Path]:
    primary = paths.processed_dir / PROCESSED_SQLITE_TEMPLATE
    rebuilt = paths.processed_dir / f"vse_rebuilt.sqlite"
    metadata = paths.processed_dir / f"metadata.json"
    return primary, rebuilt, metadata


def _connect_sqlite(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    # Avoid disk IO errors caused by journal/temp files in restricted environments.
    conn.execute("PRAGMA journal_mode=MEMORY")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def _read_products_table(conn: sqlite3.Connection) -> pd.DataFrame:
    try:
        return pd.read_sql_query("SELECT * FROM products", conn)
    except Exception:
        return pd.read_sql_query("SELECT * FROM product", conn)


def ensure_processed_sqlite(paths: ProjectPaths) -> ProcessedSQLite | None:
    primary, rebuilt, metadata_path = _sqlite_paths(paths)
    for path in (primary, rebuilt):
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            with _connect_sqlite(path) as conn:
                products = _read_products_table(conn)
            return ProcessedSQLite(products=products, sqlite_path=path, metadata_path=metadata_path)
        except Exception:
            continue
    return None


def ensure_raw_sqlite(paths: ProjectPaths) -> Path:
    raise FileNotFoundError(
        "Raw dataset import has been removed from the project. "
        "Provide a processed SQLite dataset at data/processed/vse.sqlite."
    )


@dataclass
class PreparedData:
    products: pd.DataFrame
    train_products: pd.DataFrame
    test_products: pd.DataFrame
    sqlite_path: Path
    metadata_path: Path


def _ensure_text_for_training(products: pd.DataFrame) -> pd.DataFrame:
    if "text_for_training" in products.columns:
        return products

    parts = []
    for col in (
        "product_title",
        "product_brand",
        "product_color",
        "product_bullet_point",
        "product_description",
    ):
        if col in products.columns:
            parts.append(products[col].astype("string").fillna(""))

    out = products.copy()
    out["text_for_training"] = ""
    if parts:
        text = parts[0]
        for series in parts[1:]:
            text = text.str.cat(series, sep=" ", na_rep="")
        out["text_for_training"] = text.str.replace(r"\s+", " ", regex=True).str.strip()
    return out


def assign_product_split(products: pd.DataFrame, test_fraction: float = 0.2) -> pd.DataFrame:
    if "split" in products.columns:
        normalized = products["split"].astype("string").str.lower().fillna("")
        if normalized.isin(["train", "test"]).any():
            out = products.copy()
            out["split"] = normalized.where(normalized.isin(["train", "test"]), "train")
            return out

    threshold = int(max(0.0, min(1.0, test_fraction)) * 10000)

    def split_for_id(product_id: object) -> str:
        text = "" if product_id is None else str(product_id)
        if not text:
            return "train"
        digest = hashlib.sha1(text.encode("utf-8")).digest()
        value = int.from_bytes(digest[:4], "big") % 10000
        return "test" if value < threshold else "train"

    out = products.copy()
    out["split"] = out.get("product_id", pd.Series([""] * len(out))).apply(split_for_id)
    return out


def prepare_product_database(
    paths: ProjectPaths,
    max_products: int | None = 30000,
    test_fraction: float = 0.2,
) -> PreparedData:
    paths.processed_dir.mkdir(parents=True, exist_ok=True)

    processed = ensure_processed_sqlite(paths)
    if processed is not None:
        products = processed.products
        sqlite_path = processed.sqlite_path
        metadata_path = processed.metadata_path

        if max_products is not None and len(products) > max_products:
            products = products.head(max_products).copy()

        products = _ensure_text_for_training(products)
        products = assign_product_split(products, test_fraction=test_fraction)
        train_products = products.loc[products["split"] == "train"].copy().reset_index(drop=True)
        test_products = products.loc[products["split"] == "test"].copy().reset_index(drop=True)
        return PreparedData(
            products=products,
            train_products=train_products,
            test_products=test_products,
            sqlite_path=sqlite_path,
            metadata_path=metadata_path,
        )

    primary_sqlite_path, _rebuilt_sqlite_path, _metadata_path = _sqlite_paths(paths)
    raise FileNotFoundError(
        f"Missing processed SQLite dataset for locale={primary_sqlite_path}. "
        "Create this file (table: products) then rerun."
    )
