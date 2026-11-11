from pathlib import Path
from typing import Any

import polars as pl

from src.tables.common import add_tier_summary_columns, save_table_csv
from src.tables.schema import (
    CANONICAL_MODEL_COLUMNS,
    IR_DATASETS,
    LADDER_K_ORDERED,
    METHODS_ABLATION_COMP_ORDERED,
    METHODS_ABLATION_FULL_ORDERED,
    MODEL_EMBGEMMA,
    MODEL_GEMMA_LAST,
    MODEL_GEMMA_MEAN,
    MODEL_QWEN_BASE_LAST,
    MODEL_QWEN_BASE_MEAN,
    MODEL_QWEN_EMB,
    STS_DATASETS,
)

IR_METRIC_SHORT: dict[str, str] = {
    "ndcg_at_10": "ndcg",
    "recall_at_100": "recall",
    "mrr_at_10": "mrr",
}

STS_METRIC_SHORT: dict[str, str] = {
    "spearman_rho": "spearman",
    "pearson_r": "pearson",
}

GEOMETRY_METRICS: list[str] = [
    "centroid_norm",
    "average_cosine",
    "mev_top1",
    "nid_spectral_entropy",
    "isoscore",
]

FULL_ABLATION_LABELS: dict[str, str] = {
    "ablation_o0_c0_e2": "Unregularized ZCA (Sigma, gamma=1, eps=1e-6)",
    "ablation_o0_c0_e1": "Centered + Adaptive eps* (Sigma, gamma=1, eps*)",
    "ablation_o0_c1_e0": "Centered + Crowding gamma (Sigma, gamma_crowd, 1e-4)",
    "ablation_o0_c1_e1": "w/o Orthogonal Projection (Sigma, gamma_crowd, eps*)",
    "ablation_o1_c0_e0": "Orthogonal Soft-ZCA (Sigma_perp, gamma=1, 1e-4)",
    "ablation_o1_c0_e1": "w/o Crowding Exponent (Sigma_perp, gamma=1, eps*)",
    "ablation_o1_c2_e0": "Fixed Exponent gamma=0.5 (Sigma_perp, 0.5, 1e-4)",
    "ablation_o1_c1_e0": "w/o Adaptive Regularizer (Sigma_perp, gamma_crowd, 1e-4)",
}

COMP_ABLATION_LABELS: dict[str, str] = {
    "ablation_o1_pca": "Orthogonal PCA (Sigma_perp, gamma=0, eps=0)",
    "ablation_o1_c0_e0": "Orthogonal SpecTemp (Sigma_perp, gamma_SNR, 1e-6)",
    "ablation_o1_c0_e2": "w/o Crowding Floor (Sigma_perp, gamma_SNR, eps*(k))",
    "ablation_o1_c4_e2": "Fixed Exponent gamma=0.5 (Sigma_perp, 0.5, eps*(k))",
    "ablation_o1_c2_e2": "w/o SNR Schedule (Sigma_perp, gamma_crowd, eps*(k))",
    "ablation_o1_c1_e0": "w/o Adaptive Regularizer (Sigma_perp, gamma*(k), 1e-6)",
    "ablation_o1_c1_e1": "w/o Rank Scaling tau(k)^2 (Sigma_perp, gamma*(k), eps*(d))",
    "ablation_o0_c1_e2": "w/o Orthogonal Projection (Sigma, gamma*(k), eps*(k))",
}


def _build_full_ablation_tables(
    master_lf: pl.LazyFrame,
    ablation_dir: Path,
    generated: dict[str, pl.DataFrame],
) -> None:
    """Build 3 retrieval and 2 similarity full-dimension ablation tables (24 rows x 12 cols each)."""
    method_order = {m: i for i, m in enumerate(METHODS_ABLATION_FULL_ORDERED)}

    track_configs = [
        ("retrieval", IR_DATASETS, IR_METRIC_SHORT, 4),
        ("similarity", STS_DATASETS, STS_METRIC_SHORT, 2),
    ]

    for track, datasets, metric_map, prec in track_configs:
        ds_order = {d: i for i, d in enumerate(datasets)}
        for metric_name, m_short in metric_map.items():
            sub_df = (
                master_lf.filter(
                    (pl.col("track") == track)
                    & pl.col("dataset").is_in(datasets)
                    & (pl.col("regime") == "full")
                    & pl.col("method").is_in(METHODS_ABLATION_FULL_ORDERED)
                    & (pl.col("metric") == metric_name)
                )
                .collect()
                .pivot(
                    on="model",
                    index=["dataset", "method"],
                    values="value",
                    aggregate_function="mean",
                )
            )

            with_summaries = add_tier_summary_columns(sub_df)
            enriched = (
                with_summaries.with_columns(
                    pl.col("method")
                    .replace_strict(FULL_ABLATION_LABELS, return_dtype=pl.Utf8)
                    .alias("variant"),
                    pl.col("dataset")
                    .replace_strict(ds_order, default=999, return_dtype=pl.Int32)
                    .alias("__d_ord"),
                    pl.col("method")
                    .replace_strict(method_order, default=999, return_dtype=pl.Int32)
                    .alias("__m_ord"),
                )
                .sort(["__d_ord", "__m_ord"])
                .drop(["__d_ord", "__m_ord"])
            )

            cols = (
                ["dataset", "method", "variant"]
                + [c for c in CANONICAL_MODEL_COLUMNS if c in enriched.columns]
                + [
                    c
                    for c in ["mean_dedicated", "mean_base_mean", "mean_base_last"]
                    if c in enriched.columns
                ]
            )
            final_df = enriched.select(cols)
            table_key = f"{track}_ablation_{m_short}"
            save_table_csv(final_df, ablation_dir / f"{table_key}.csv", precision=prec)
            generated[table_key] = final_df


def _build_compression_ablation_tables(
    master_df: pl.DataFrame,
    ablation_dir: Path,
    generated: dict[str, pl.DataFrame],
) -> None:
    """Build 5 compression ablation tables (96 rows = 3 datasets x 4 k x 8 variants, covering all 6 models)."""
    track_configs = [
        ("retrieval", IR_DATASETS, IR_METRIC_SHORT, 4),
        ("similarity", STS_DATASETS, STS_METRIC_SHORT, 2),
    ]

    for track, datasets, metric_map, prec in track_configs:
        for metric_name, m_short in metric_map.items():
            rows: list[dict[str, Any]] = []
            for dataset in datasets:
                native_sub = master_df.filter(
                    (pl.col("track") == track)
                    & (pl.col("dataset") == dataset)
                    & (pl.col("regime") == "full")
                    & (pl.col("method") == "baseline")
                    & (pl.col("metric") == metric_name)
                )
                native_map = dict(zip(native_sub["model"], native_sub["value"]))

                comp_sub = master_df.filter(
                    (pl.col("track") == track)
                    & (pl.col("dataset") == dataset)
                    & (pl.col("regime") == "compression")
                    & pl.col("method").is_in(METHODS_ABLATION_COMP_ORDERED)
                    & (pl.col("metric") == metric_name)
                )

                for k in LADDER_K_ORDERED:
                    for m in METHODS_ABLATION_COMP_ORDERED:
                        row: dict[str, Any] = {
                            "dataset": dataset,
                            "k": k,
                            "method": m,
                            "variant": COMP_ABLATION_LABELS[m],
                        }
                        sub = comp_sub.filter(
                            (pl.col("k") == k) & (pl.col("method") == m)
                        )
                        scores = dict(zip(sub["model"], sub["value"]))
                        ret_map: dict[str, float] = {}

                        for mod_col in CANONICAL_MODEL_COLUMNS:
                            val = scores.get(mod_col, 0.0)
                            nat = native_map.get(mod_col, 1.0)
                            ret_pct = (val / nat * 100.0) if abs(nat) > 1e-9 else 0.0
                            ret_map[mod_col] = ret_pct
                            row[mod_col] = round(val, prec)
                            row[f"{mod_col} (retention %)"] = round(ret_pct, 2)

                        row["mean_dedicated_retention_pct"] = round(
                            (ret_map[MODEL_EMBGEMMA] + ret_map[MODEL_QWEN_EMB]) / 2.0,
                            2,
                        )
                        row["mean_base_mean_retention_pct"] = round(
                            (ret_map[MODEL_GEMMA_MEAN] + ret_map[MODEL_QWEN_BASE_MEAN])
                            / 2.0,
                            2,
                        )
                        row["mean_base_last_retention_pct"] = round(
                            (ret_map[MODEL_GEMMA_LAST] + ret_map[MODEL_QWEN_BASE_LAST])
                            / 2.0,
                            2,
                        )
                        rows.append(row)

            final_df = pl.DataFrame(rows)
            table_key = f"compression_ablation_{m_short}"
            save_table_csv(final_df, ablation_dir / f"{table_key}.csv", precision=prec)
            generated[table_key] = final_df


def _build_geometry_ablation_tables(
    master_lf: pl.LazyFrame,
    ablation_dir: Path,
    generated: dict[str, pl.DataFrame],
) -> None:
    """Build 2 geometry ablation tables covering all 6 models, 6 datasets, and 8 variants."""
    all_datasets = IR_DATASETS + STS_DATASETS
    ds_order = {d: i for i, d in enumerate(all_datasets)}
    model_order = {m: i for i, m in enumerate(CANONICAL_MODEL_COLUMNS)}
    full_order = {m: i for i, m in enumerate(METHODS_ABLATION_FULL_ORDERED)}
    comp_order = {m: i for i, m in enumerate(METHODS_ABLATION_COMP_ORDERED)}
    k_order = {k: i for i, k in enumerate(LADDER_K_ORDERED)}

    # 1. Full-dimension geometry ablation table (288 rows = 6 datasets x 6 models x 8 variants)
    full_df = (
        master_lf.filter(
            (pl.col("track") == "geometry")
            & (pl.col("regime") == "full")
            & pl.col("method").is_in(METHODS_ABLATION_FULL_ORDERED)
        )
        .collect()
        .pivot(
            on="metric",
            index=["dataset", "model", "method"],
            values="value",
        )
        .with_columns(
            pl.col("method")
            .replace_strict(FULL_ABLATION_LABELS, return_dtype=pl.Utf8)
            .alias("variant"),
            pl.col("dataset")
            .replace_strict(ds_order, default=999, return_dtype=pl.Int32)
            .alias("__d_ord"),
            pl.col("model")
            .replace_strict(model_order, default=999, return_dtype=pl.Int32)
            .alias("__m_ord"),
            pl.col("method")
            .replace_strict(full_order, default=999, return_dtype=pl.Int32)
            .alias("__meth_ord"),
        )
        .sort(["__d_ord", "__m_ord", "__meth_ord"])
        .drop(["__d_ord", "__m_ord", "__meth_ord"])
    )
    full_cols = ["dataset", "model", "method", "variant"] + [
        c for c in GEOMETRY_METRICS if c in full_df.columns
    ]
    full_df = full_df.select(full_cols)
    save_table_csv(full_df, ablation_dir / "geometry_ablation_full.csv", precision=4)
    generated["geometry_ablation_full"] = full_df

    # 2. Compression geometry ablation table (1,152 rows = 6 datasets x 4 k x 6 models x 8 variants)
    comp_df = (
        master_lf.filter(
            (pl.col("track") == "geometry")
            & (pl.col("regime") == "compression")
            & pl.col("method").is_in(METHODS_ABLATION_COMP_ORDERED)
        )
        .collect()
        .pivot(
            on="metric",
            index=["dataset", "k", "model", "method"],
            values="value",
        )
        .with_columns(
            pl.col("method")
            .replace_strict(COMP_ABLATION_LABELS, return_dtype=pl.Utf8)
            .alias("variant"),
            pl.col("dataset")
            .replace_strict(ds_order, default=999, return_dtype=pl.Int32)
            .alias("__d_ord"),
            pl.col("k")
            .replace_strict(k_order, default=999, return_dtype=pl.Int32)
            .alias("__k_ord"),
            pl.col("model")
            .replace_strict(model_order, default=999, return_dtype=pl.Int32)
            .alias("__m_ord"),
            pl.col("method")
            .replace_strict(comp_order, default=999, return_dtype=pl.Int32)
            .alias("__meth_ord"),
        )
        .sort(["__d_ord", "__k_ord", "__m_ord", "__meth_ord"])
        .drop(["__d_ord", "__k_ord", "__m_ord", "__meth_ord"])
    )
    comp_cols = ["dataset", "k", "model", "method", "variant"] + [
        c for c in GEOMETRY_METRICS if c in comp_df.columns
    ]
    comp_df = comp_df.select(comp_cols)
    save_table_csv(
        comp_df, ablation_dir / "geometry_ablation_compression.csv", precision=4
    )
    generated["geometry_ablation_compression"] = comp_df


def build_suite_6_ablation(
    master_lf: pl.LazyFrame, tables_dir: Path
) -> dict[str, pl.DataFrame]:
    """Build the 12 non-redundant component ablation tables in tables/ablation/."""
    ablation_dir = tables_dir / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, pl.DataFrame] = {}

    master_df = master_lf.collect()
    _build_full_ablation_tables(master_lf, ablation_dir, generated)
    _build_compression_ablation_tables(master_df, ablation_dir, generated)
    _build_geometry_ablation_tables(master_lf, ablation_dir, generated)
    return generated
