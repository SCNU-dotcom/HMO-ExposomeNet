"""Shared input settings and data loading for the PFAS analysis scripts.

This module is the single source of truth for:

* the location of the input workbook (``WORKBOOK``),
* the PFAS cluster ordering and the short-/long-chain split,
* building the specimen x pollutant-cluster concentration matrix.

It is imported by all three ``analyze_*.py`` scripts.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

# Repository root (the folder that contains this file).
PROJECT_DIR = Path(__file__).resolve().parent

# Input data: long-format table, one row per (Sample, Organ, compound) detection.
# Required columns: Sample, Organ, System, Name, Category, Clusters, Concentration.
WORKBOOK = PROJECT_DIR / "Sample-resolved_human_chemical_exposome.xlsx"
SHEET = 0  # first worksheet

PFAS_ORDER = ["PFAS-1", "PFAS-2", "PFAS-3", "PFAS-4", "PFAS-5"]
# Cluster IDs follow the current workbook; never exchange PFAS-2 and PFAS-4.
SHORT_CHAIN = ["PFAS-1", "PFAS-2"]
LONG_CHAIN = ["PFAS-3", "PFAS-4", "PFAS-5"]


def find_workbook() -> Path:
    """Return the input workbook path, failing early if it is missing."""
    if not WORKBOOK.is_file():
        raise FileNotFoundError(f"analysis workbook not found: {WORKBOOK}")
    return WORKBOOK


def natural_key(text: str) -> tuple:
    """Sort key that orders 'Pest.-10' after 'Pest.-9' (numeric suffix)."""
    head, _, tail = str(text).rpartition("-")
    try:
        return (head.lower(), int(tail))
    except ValueError:
        return (str(text).lower(), 10 ** 9)


def pfas_group(cluster: str, category: str) -> str:
    """Map a cluster to its reporting group (short-chain focus, long-chain, ...)."""
    if category != "PFAS":
        return "Non-PFAS cluster"
    if cluster in SHORT_CHAIN:
        return "Short-chain PFAS focus"
    return "Long-chain PFAS cluster" if cluster in LONG_CHAIN else "Other PFAS cluster"


def load_specimens() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (wide concentration matrix, node metadata, specimen metadata).

    One row per specimen = individual x organ; one column per pollutant cluster.
    """
    df = pd.read_excel(find_workbook(), sheet_name=SHEET)
    df["Concentration"] = pd.to_numeric(df["Concentration"],
                                        errors="coerce").fillna(0.0)
    for col in ("Sample", "Organ", "Clusters", "Category", "System"):
        df[col] = df[col].astype(str).str.strip()
    df["Category"] = df["Category"].replace({"others": "Others"})
    df["specimen"] = df["Sample"] + " | " + df["Organ"]

    wide = (df.groupby(["specimen", "Clusters"])["Concentration"].sum()
              .unstack(fill_value=0.0))
    cats = df.drop_duplicates("Clusters").set_index("Clusters")["Category"]
    cols = sorted(wide.columns,
                  key=lambda c: (0 if c in PFAS_ORDER else 1, natural_key(c)))
    wide = wide[cols]

    spec = df.drop_duplicates("specimen").set_index("specimen").reindex(wide.index)
    meta_spec = pd.DataFrame({
        "specimen": wide.index,
        "individual": spec["Sample"].values,
        "organ": spec["Organ"].values,
        "system": spec["System"].values,
    })

    nodes = pd.DataFrame({
        "node": cols,
        "category": [cats.get(c, "") for c in cols],
        "group": [pfas_group(c, cats.get(c, "")) for c in cols],
        "n_detected": [int((wide[c] > 0).sum()) for c in cols],
        "detect_rate": [float((wide[c] > 0).mean()) for c in cols],
    })
    return wide, nodes, meta_spec
