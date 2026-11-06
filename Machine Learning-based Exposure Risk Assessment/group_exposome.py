"""Aggregate chemical measurements by sample, organ, and cluster.

The input workbook is read-only. The grouped workbook is written to a separate
path so that the source data is never overwritten.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path(__file__).resolve().parent / "Sample-resolved_human_chemical_exposome.xlsx"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "Grouped sample-resolved human chemical exposome.xlsx"
KEY_COLUMNS = ["Sample", "Organ", "Clusters"]
REQUIRED_COLUMNS = KEY_COLUMNS + ["Concentration"]


def group_exposome(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Sum concentration within each (sample, organ, cluster) group."""
    frame = pd.read_excel(input_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Input workbook is missing required columns: {missing}")

    frame["Concentration"] = pd.to_numeric(frame["Concentration"], errors="raise")
    aggregation = {"Concentration": "sum"}

    # Keep metadata from the first row; validate it is consistent within groups.
    metadata = [
        column for column in frame.columns
        if column not in KEY_COLUMNS + ["Concentration"]
    ]
    metadata_aggregation = {}
    for column in metadata:
        inconsistent = frame.groupby(KEY_COLUMNS, dropna=False)[column].nunique(dropna=False)
        if (inconsistent > 1).any():
            raise ValueError(
                f"Metadata column {column!r} varies within one sample/organ/cluster group."
            )
        metadata_aggregation[column] = "first"

    grouped = (
        frame.groupby(KEY_COLUMNS, as_index=False, sort=False, dropna=False)
        .agg({**aggregation, **metadata_aggregation})
        .rename(columns={"Clusters": "Cluster"})
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_excel(output_path, index=False, sheet_name="Grouped data")
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = group_exposome(args.input, args.output)
    print(f"Wrote {len(result):,} grouped rows to {args.output}")


if __name__ == "__main__":
    main()
