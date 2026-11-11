from pathlib import Path

import polars as pl

from src.tables.common import save_table_csv
from src.tables.schema import (
    CANONICAL_MODEL_COLUMNS,
    IR_DATASETS,
    LADDER_K_ORDERED,
    METHODS_COMP_ORDERED,
    METHODS_FULL_ORDERED,
    STS_DATASETS,
)

DATASET_SHORT: dict[str, str] = {
    "FiQA2018": "fiqa",
    "ArguAna": "arguana",
    "SCIDOCS": "scidocs",
    "STSBenchmark": "stsb",
    "SICK-R": "sickr",
    "STS22.v2": "sts22",
}

GEOMETRY_METRICS: list[str] = [
    "centroid_norm",
    "average_cosine",
    "mev_top1",
    "nid_spectral_entropy",
    "isoscore",
]


def build_suite_4_geometry(
    master_lf: pl.LazyFrame, tables_dir: Path
) -> dict[str, pl.DataFrame]:
    """Build the 12 non-redundant geometry tables (6 full-dimension 54x7 + 6 compression 120x8) covering all 6 models."""
    geom_dir = tables_dir / "geometry"
    geom_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, pl.DataFrame] = {}

    model_order_map = {m: i for i, m in enumerate(CANONICAL_MODEL_COLUMNS)}
    full_method_order_map = {m: i for i, m in enumerate(METHODS_FULL_ORDERED)}
    comp_method_order_map = {m: i for i, m in enumerate(METHODS_COMP_ORDERED)}
    k_order_map = {k: i for i, k in enumerate(LADDER_K_ORDERED)}

    all_datasets = IR_DATASETS + STS_DATASETS

    for d in all_datasets:
        d_short = DATASET_SHORT[d]

        # 1. Full-dimension geometry table (54 rows = 6 models x 9 full methods)
        full_df = (
            master_lf.filter(
                (pl.col("track") == "geometry")
                & (pl.col("dataset") == d)
                & (pl.col("regime") == "full")
                & pl.col("method").is_in(METHODS_FULL_ORDERED)
            )
            .collect()
            .pivot(
                on="metric",
                index=["model", "method"],
                values="value",
            )
            .with_columns(
                pl.col("model")
                .replace_strict(model_order_map, default=999, return_dtype=pl.Int32)
                .alias("__m_ord"),
                pl.col("method")
                .replace_strict(
                    full_method_order_map, default=999, return_dtype=pl.Int32
                )
                .alias("__meth_ord"),
            )
            .sort(["__m_ord", "__meth_ord"])
            .drop(["__m_ord", "__meth_ord"])
        )

        full_cols = ["model", "method"] + [
            c for c in GEOMETRY_METRICS if c in full_df.columns
        ]
        full_result = full_df.select(full_cols)
        full_out = geom_dir / f"{d_short}_geometry_full.csv"
        save_table_csv(full_result, full_out, precision=4)
        generated[f"{d_short}_geometry_full"] = full_result

        # 2. Compression geometry table (120 rows = 4 k x 6 models x 5 compression methods)
        comp_df = (
            master_lf.filter(
                (pl.col("track") == "geometry")
                & (pl.col("dataset") == d)
                & (pl.col("regime") == "compression")
                & pl.col("method").is_in(METHODS_COMP_ORDERED)
            )
            .collect()
            .pivot(
                on="metric",
                index=["k", "model", "method"],
                values="value",
            )
            .with_columns(
                pl.col("k")
                .replace_strict(k_order_map, default=999, return_dtype=pl.Int32)
                .alias("__k_ord"),
                pl.col("model")
                .replace_strict(model_order_map, default=999, return_dtype=pl.Int32)
                .alias("__m_ord"),
                pl.col("method")
                .replace_strict(
                    comp_method_order_map, default=999, return_dtype=pl.Int32
                )
                .alias("__meth_ord"),
            )
            .sort(["__k_ord", "__m_ord", "__meth_ord"])
            .drop(["__k_ord", "__m_ord", "__meth_ord"])
        )

        comp_cols = ["k", "model", "method"] + [
            c for c in GEOMETRY_METRICS if c in comp_df.columns
        ]
        comp_result = comp_df.select(comp_cols)
        comp_out = geom_dir / f"{d_short}_geometry_compression.csv"
        save_table_csv(comp_result, comp_out, precision=4)
        generated[f"{d_short}_geometry_compression"] = comp_result

    return generated
