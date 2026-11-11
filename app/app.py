"""Interactive Gradio results explorer for COAST embedding post-processing and compression."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("coast_app")

DATA_DIR = Path(__file__).resolve().parent / "data"

IR_DATASETS = ["FiQA-2018", "ArguAna", "SCIDOCS"]
STS_DATASETS = ["STSBenchmark", "SICK-R", "STS22.v2"]
ALL_DATASETS = IR_DATASETS + STS_DATASETS

DATASET_TO_SLUG = {
    "FiQA-2018": "fiqa",
    "ArguAna": "arguana",
    "SCIDOCS": "scidocs",
    "STSBenchmark": "stsb",
    "SICK-R": "sickr",
    "STS22.v2": "sts22",
}

DATASET_TO_RAW = {
    "FiQA-2018": "FiQA2018",
    "ArguAna": "ArguAna",
    "SCIDOCS": "SCIDOCS",
    "STSBenchmark": "STSBenchmark",
    "SICK-R": "SICK-R",
    "STS22.v2": "STS22.v2",
}

RAW_TO_DATASET = {v: k for k, v in DATASET_TO_RAW.items()}

IR_METRICS = {
    "nDCG@10": ("ndcg", 4),
    "MRR@10": ("mrr", 4),
    "Recall@100": ("recall", 4),
}

STS_METRICS = {
    "Spearman (ρ × 100)": ("spearman", 2),
    "Pearson (r × 100)": ("pearson", 2),
}

GEOM_METRICS = {
    "Centroid Norm ||μ||₂": "centroid_norm",
    "Average Cosine": "average_cosine",
    "MEV (Top-1)": "mev_top1",
    "NID (Spectral Entropy)": "nid_spectral_entropy",
    "IsoScore": "isoscore",
}

GEOM_OPTIONS = ["All 5 Geometry Metrics"] + list(GEOM_METRICS.keys())

PARAM_OPTIONS = [
    "Variance Partition & Knee Parameters",
    "Tempering Schedules (γ_SNR, ε*, γ*)",
]

MODEL_COL_MAP = {
    "EmbGemma": "google/embeddinggemma-300m",
    "Qwen3-Emb": "Qwen/Qwen3-Embedding-0.6B",
    "Gemma3-Mean": "google/gemma-3-1b-pt (mean_pooling)",
    "Gemma3-Last": "google/gemma-3-1b-pt (last_token_pooling)",
    "Qwen3.5-Mean": "Qwen/Qwen3.5-0.8B-Base (mean_pooling)",
    "Qwen3.5-Last": "Qwen/Qwen3.5-0.8B-Base (last_token_pooling)",
}

RAW_TO_SHORT_MODEL = {v: k for k, v in MODEL_COL_MAP.items()}

CATEGORY_COL_MAP = {
    "Dedicated Mean": ("mean_dedicated", "mean_dedicated_retention_pct"),
    "Base-Mean": ("mean_base_mean", "mean_base_mean_retention_pct"),
    "Base-Last": ("mean_base_last", "mean_base_last_retention_pct"),
}

ALL_DISPLAY_COLS = list(MODEL_COL_MAP.keys()) + list(CATEGORY_COL_MAP.keys())

FULL_MAIN_METHODS = {
    "baseline": "Baseline",
    "standardization": "Standardization",
    "r1": "R1",
    "r2": "R2",
    "soft_zca": "Soft-ZCA",
    "coast": "COAST (Ours)",
    "abtt_1": "ABTT (D=1)",
    "abtt_2": "ABTT (D=2)",
    "abtt_3": "ABTT (D=3)",
}

FULL_ABLATION_METHODS = {
    "ablation_o0_c0_e2": "Unreg. ZCA (Σ, γ=1, ε=1e-6)",
    "ablation_o0_c0_e1": "Centered + ε* (Σ, γ=1, ε*)",
    "ablation_o0_c1_e0": "Centered + γ_crowd (Σ, γ_crowd, 1e-4)",
    "ablation_o0_c1_e1": "w/o Orth. Proj. (Σ, γ_crowd, ε*)",
    "ablation_o1_c0_e0": "Orth. Soft-ZCA (Σ_⊥, γ=1, 1e-4)",
    "ablation_o1_c0_e1": "w/o Crowd. Exp. (Σ_⊥, γ=1, ε*)",
    "ablation_o1_c2_e0": "Fixed γ=0.5 (Σ_⊥, γ=0.5, 1e-4)",
    "ablation_o1_c1_e0": "w/o Adaptive ε* (Σ_⊥, γ_crowd, 1e-4)",
}

COMP_MAIN_METHODS = {
    "prefix": "Prefix (MRL)",
    "pca": "PCA-k",
    "whitening": "Whitening-k",
    "spectemp": "SpecTemp",
    "coast": "COAST (Ours)",
}

COMP_ABLATION_METHODS = {
    "ablation_o1_pca": "Orth. PCA (Σ_⊥, γ=0, ε=0)",
    "ablation_o1_c0_e0": "Orth. SpecTemp (Σ_⊥, γ_SNR, 1e-6)",
    "ablation_o1_c0_e2": "w/o Crowd. Floor (Σ_⊥, γ_SNR, ε*)",
    "ablation_o1_c2_e2": "Fixed γ=0.5 (Σ_⊥, γ=0.5, ε*)",
    "ablation_o1_c4_e2": "w/o SNR Sched. (Σ_⊥, γ_crowd, ε*)",
    "ablation_o1_c1_e0": "w/o Adaptive ε* (Σ_⊥, γ*, 1e-6)",
    "ablation_o1_c1_e1": "w/o Rank Scale τ² (Σ_⊥, γ*, ε*(d))",
    "ablation_o0_c1_e2": "w/o Orth. Proj. (Σ, γ*, ε*)",
}


def get_method_map(dim_mode: str, method_group: str) -> dict[str, str]:
    """Return mapping from raw method code to display label for the selected mode and group."""
    is_full = dim_mode.startswith("Full")
    if is_full:
        if method_group == "Main Methods":
            return dict(FULL_MAIN_METHODS)
        if method_group == "Component Ablations":
            return dict(FULL_ABLATION_METHODS)
        return {**FULL_MAIN_METHODS, **FULL_ABLATION_METHODS}
    if method_group == "Main Methods":
        return dict(COMP_MAIN_METHODS)
    if method_group == "Component Ablations":
        return dict(COMP_ABLATION_METHODS)
    return {**COMP_MAIN_METHODS, **COMP_ABLATION_METHODS}


def build_task_table(
    track: str,
    metric_label: str,
    dim_mode: str,
    selected_k: list[str],
    comp_display: str,
    datasets: list[str],
    method_group: str,
    selected_methods: list[str],
    selected_cols: list[str],
) -> tuple[pd.DataFrame, str]:
    """Build the filtered task evaluation DataFrame."""
    is_ir = track.startswith("Retrieval")
    metric_map = IR_METRICS if is_ir else STS_METRICS
    if metric_label not in metric_map:
        metric_label = next(iter(metric_map.keys()))
    metric_slug, dec = metric_map[metric_label]
    subdir = "retrieval" if is_ir else "similarity"
    valid_datasets = [
        d for d in datasets if d in (IR_DATASETS if is_ir else STS_DATASETS)
    ]
    if not valid_datasets:
        valid_datasets = IR_DATASETS if is_ir else STS_DATASETS

    method_map = get_method_map(dim_mode, method_group)
    label_to_code = {v: k for k, v in method_map.items()}
    active_codes = [label_to_code[m] for m in selected_methods if m in label_to_code]
    if not active_codes:
        active_codes = list(method_map.keys())

    active_cols = [c for c in ALL_DISPLAY_COLS if c in selected_cols]
    if not active_cols:
        active_cols = ALL_DISPLAY_COLS

    is_full = dim_mode.startswith("Full")
    rows_out: list[dict[str, Any]] = []

    if is_full:
        ablation_file = (
            DATA_DIR
            / "ablation"
            / f"{'retrieval' if is_ir else 'similarity'}_ablation_{metric_slug}.csv"
        )
        df_abl = pl.read_csv(ablation_file) if ablation_file.exists() else None

        for ds in valid_datasets:
            ds_slug = DATASET_TO_SLUG[ds]
            ds_raw = DATASET_TO_RAW[ds]
            main_df = pl.read_csv(DATA_DIR / subdir / f"{ds_slug}_{metric_slug}.csv")
            lookup: dict[str, dict[str, Any]] = {
                str(r["method"]): r for r in main_df.iter_rows(named=True)
            }
            if df_abl is not None:
                for r in df_abl.filter(pl.col("dataset") == ds_raw).iter_rows(
                    named=True
                ):
                    lookup[str(r["method"])] = r

            for code in active_codes:
                if code not in lookup:
                    continue
                r = lookup[code]
                row_dict: dict[str, Any] = {
                    "Dataset": ds,
                    "k": "d",
                    "Method": method_map[code],
                }
                for col_label in active_cols:
                    if col_label in MODEL_COL_MAP:
                        raw_col = MODEL_COL_MAP[col_label]
                    else:
                        raw_col = CATEGORY_COL_MAP[col_label][0]
                    row_dict[col_label] = f"{float(r[raw_col]):.{dec}f}"
                rows_out.append(row_dict)
    else:
        k_ints = [int(k) for k in selected_k if k in {"512", "256", "128", "64"}]
        if not k_ints:
            k_ints = [512, 256, 128, 64]
        abl_comp_file = (
            DATA_DIR / "ablation" / f"compression_ablation_{metric_slug}.csv"
        )
        df_abl_comp = pl.read_csv(abl_comp_file) if abl_comp_file.exists() else None

        for ds in valid_datasets:
            ds_slug = DATASET_TO_SLUG[ds]
            ds_raw = DATASET_TO_RAW[ds]
            main_comp_df = pl.read_csv(
                DATA_DIR / "compression" / f"{ds_slug}_compression_{metric_slug}.csv"
            )
            for k_val in k_ints:
                lookup_k: dict[str, dict[str, Any]] = {
                    str(r["method"]): r
                    for r in main_comp_df.filter(pl.col("k") == k_val).iter_rows(
                        named=True
                    )
                }
                if df_abl_comp is not None:
                    for r in df_abl_comp.filter(
                        (pl.col("dataset") == ds_raw) & (pl.col("k") == k_val)
                    ).iter_rows(named=True):
                        lookup_k[str(r["method"])] = r

                for code in active_codes:
                    if code not in lookup_k:
                        continue
                    r = lookup_k[code]
                    row_dict = {
                        "Dataset": ds,
                        "k": str(k_val),
                        "Method": method_map[code],
                    }
                    for col_label in active_cols:
                        if col_label in MODEL_COL_MAP:
                            raw_col = MODEL_COL_MAP[col_label]
                            score_val = float(r[raw_col])
                            ret_val = float(r[f"{raw_col} (retention %)"])
                        else:
                            model_pair = (
                                ["EmbGemma", "Qwen3-Emb"]
                                if col_label == "Dedicated Mean"
                                else (
                                    ["Gemma3-Mean", "Qwen3.5-Mean"]
                                    if col_label == "Base-Mean"
                                    else ["Gemma3-Last", "Qwen3.5-Last"]
                                )
                            )
                            score_val = (
                                sum(float(r[MODEL_COL_MAP[m]]) for m in model_pair)
                                / 2.0
                            )
                            ret_val = float(r[CATEGORY_COL_MAP[col_label][1]])

                        if comp_display == "Absolute Score":
                            row_dict[col_label] = f"{score_val:.{dec}f}"
                        elif comp_display == "Retention %":
                            row_dict[col_label] = f"{ret_val:.1f}%"
                        else:
                            row_dict[col_label] = (
                                f"{score_val:.{dec}f} ({ret_val:.1f}%)"
                            )
                    rows_out.append(row_dict)

    pdf = pd.DataFrame(rows_out)
    summary = (
        f"**View:** Task Performance | **Track:** {track} | **Metric:** {metric_label} | "
        f"**Mode:** {dim_mode} | **Rows:** {len(pdf)}"
    )
    return pdf, summary


def build_geometry_table(
    metric_label: str,
    dim_mode: str,
    selected_k: list[str],
    datasets: list[str],
    method_group: str,
    selected_methods: list[str],
    selected_cols: list[str],
) -> tuple[pd.DataFrame, str]:
    """Build the filtered intrinsic geometry DataFrame."""
    valid_datasets = [d for d in datasets if d in ALL_DATASETS] or ALL_DATASETS
    method_map = get_method_map(dim_mode, method_group)
    label_to_code = {v: k for k, v in method_map.items()}
    active_codes = [label_to_code[m] for m in selected_methods if m in label_to_code]
    if not active_codes:
        active_codes = list(method_map.keys())

    is_full = dim_mode.startswith("Full")
    k_ints = [int(k) for k in selected_k if k in {"512", "256", "128", "64"}] or [
        512,
        256,
        128,
        64,
    ]

    df_abl_gf = pl.read_csv(DATA_DIR / "ablation" / "geometry_ablation_full.csv")
    df_abl_gc = pl.read_csv(DATA_DIR / "ablation" / "geometry_ablation_compression.csv")

    if metric_label not in GEOM_OPTIONS:
        metric_label = "All 5 Geometry Metrics"

    active_models = [c for c in MODEL_COL_MAP if c in selected_cols] or list(
        MODEL_COL_MAP.keys()
    )
    rows_out: list[dict[str, Any]] = []

    if metric_label == "All 5 Geometry Metrics":
        for ds in valid_datasets:
            ds_slug = DATASET_TO_SLUG[ds]
            ds_raw = DATASET_TO_RAW[ds]
            if is_full:
                main_g = pl.read_csv(
                    DATA_DIR / "geometry" / f"{ds_slug}_geometry_full.csv"
                )
                abl_g = df_abl_gf.filter(pl.col("dataset") == ds_raw)
                lookup: dict[tuple[str, str], dict[str, Any]] = {}
                for r in main_g.iter_rows(named=True):
                    lookup[(str(r["model"]), str(r["method"]))] = r
                for r in abl_g.iter_rows(named=True):
                    lookup[(str(r["model"]), str(r["method"]))] = r

                for m_short in active_models:
                    m_raw = MODEL_COL_MAP[m_short]
                    for code in active_codes:
                        if (m_raw, code) not in lookup:
                            continue
                        r = lookup[(m_raw, code)]
                        rows_out.append(
                            {
                                "Dataset": ds,
                                "k": "d",
                                "Model": m_short,
                                "Method": method_map[code],
                                "||μ||₂": f"{float(r['centroid_norm']):.4f}",
                                "AvgCos": f"{float(r['average_cosine']):.4f}",
                                "MEV": f"{float(r['mev_top1']):.4f}",
                                "NID": f"{float(r['nid_spectral_entropy']):.4f}",
                                "IsoScore": f"{float(r['isoscore']):.4f}",
                            }
                        )
            else:
                main_gc = pl.read_csv(
                    DATA_DIR / "geometry" / f"{ds_slug}_geometry_compression.csv"
                )
                abl_gc = df_abl_gc.filter(pl.col("dataset") == ds_raw)
                lookup_c: dict[tuple[int, str, str], dict[str, Any]] = {}
                for r in main_gc.iter_rows(named=True):
                    lookup_c[(int(r["k"]), str(r["model"]), str(r["method"]))] = r
                for r in abl_gc.iter_rows(named=True):
                    lookup_c[(int(r["k"]), str(r["model"]), str(r["method"]))] = r

                for k_val in k_ints:
                    for m_short in active_models:
                        m_raw = MODEL_COL_MAP[m_short]
                        for code in active_codes:
                            if (k_val, m_raw, code) not in lookup_c:
                                continue
                            r = lookup_c[(k_val, m_raw, code)]
                            rows_out.append(
                                {
                                    "Dataset": ds,
                                    "k": str(k_val),
                                    "Model": m_short,
                                    "Method": method_map[code],
                                    "||μ||₂": f"{float(r['centroid_norm']):.4f}",
                                    "AvgCos": f"{float(r['average_cosine']):.4f}",
                                    "MEV": f"{float(r['mev_top1']):.4f}",
                                    "NID": f"{float(r['nid_spectral_entropy']):.4f}",
                                    "IsoScore": f"{float(r['isoscore']):.4f}",
                                }
                            )
    else:
        g_col = GEOM_METRICS[metric_label]
        active_cols = [
            c for c in ALL_DISPLAY_COLS if c in selected_cols
        ] or ALL_DISPLAY_COLS
        for ds in valid_datasets:
            ds_slug = DATASET_TO_SLUG[ds]
            ds_raw = DATASET_TO_RAW[ds]
            if is_full:
                main_g = pl.read_csv(
                    DATA_DIR / "geometry" / f"{ds_slug}_geometry_full.csv"
                )
                abl_g = df_abl_gf.filter(pl.col("dataset") == ds_raw)
                lookup_single: dict[tuple[str, str], float] = {}
                for r in main_g.iter_rows(named=True):
                    lookup_single[(str(r["model"]), str(r["method"]))] = float(r[g_col])
                for r in abl_g.iter_rows(named=True):
                    lookup_single[(str(r["model"]), str(r["method"]))] = float(r[g_col])

                for code in active_codes:
                    row_dict: dict[str, Any] = {
                        "Dataset": ds,
                        "k": "d",
                        "Method": method_map[code],
                    }
                    for col_label in active_cols:
                        if col_label in MODEL_COL_MAP:
                            val = lookup_single.get(
                                (MODEL_COL_MAP[col_label], code), 0.0
                            )
                        else:
                            pair = (
                                ["EmbGemma", "Qwen3-Emb"]
                                if col_label == "Dedicated Mean"
                                else (
                                    ["Gemma3-Mean", "Qwen3.5-Mean"]
                                    if col_label == "Base-Mean"
                                    else ["Gemma3-Last", "Qwen3.5-Last"]
                                )
                            )
                            val = (
                                sum(
                                    lookup_single.get((MODEL_COL_MAP[m], code), 0.0)
                                    for m in pair
                                )
                                / 2.0
                            )
                        row_dict[col_label] = f"{val:.4f}"
                    rows_out.append(row_dict)
            else:
                main_gc = pl.read_csv(
                    DATA_DIR / "geometry" / f"{ds_slug}_geometry_compression.csv"
                )
                abl_gc = df_abl_gc.filter(pl.col("dataset") == ds_raw)
                lookup_cs: dict[tuple[int, str, str], float] = {}
                for r in main_gc.iter_rows(named=True):
                    lookup_cs[(int(r["k"]), str(r["model"]), str(r["method"]))] = float(
                        r[g_col]
                    )
                for r in abl_gc.iter_rows(named=True):
                    lookup_cs[(int(r["k"]), str(r["model"]), str(r["method"]))] = float(
                        r[g_col]
                    )

                for k_val in k_ints:
                    for code in active_codes:
                        row_dict = {
                            "Dataset": ds,
                            "k": str(k_val),
                            "Method": method_map[code],
                        }
                        for col_label in active_cols:
                            if col_label in MODEL_COL_MAP:
                                val = lookup_cs.get(
                                    (k_val, MODEL_COL_MAP[col_label], code), 0.0
                                )
                            else:
                                pair = (
                                    ["EmbGemma", "Qwen3-Emb"]
                                    if col_label == "Dedicated Mean"
                                    else (
                                        ["Gemma3-Mean", "Qwen3.5-Mean"]
                                        if col_label == "Base-Mean"
                                        else ["Gemma3-Last", "Qwen3.5-Last"]
                                    )
                                )
                                val = (
                                    sum(
                                        lookup_cs.get(
                                            (k_val, MODEL_COL_MAP[m], code), 0.0
                                        )
                                        for m in pair
                                    )
                                    / 2.0
                                )
                            row_dict[col_label] = f"{val:.4f}"
                        rows_out.append(row_dict)

    pdf = pd.DataFrame(rows_out)
    summary = (
        f"**View:** Intrinsic Geometry | **Metric:** {metric_label} | "
        f"**Mode:** {dim_mode} | **Rows:** {len(pdf)}"
    )
    return pdf, summary


def build_parameters_table(
    metric_label: str,
    datasets: list[str],
    selected_cols: list[str],
) -> tuple[pd.DataFrame, str]:
    """Build the empirical spectral parameters DataFrame."""
    valid_datasets = [d for d in datasets if d in ALL_DATASETS] or ALL_DATASETS
    active_models = [c for c in MODEL_COL_MAP if c in selected_cols] or list(
        MODEL_COL_MAP.keys()
    )
    raw_datasets = {DATASET_TO_RAW[d] for d in valid_datasets}
    raw_models = {MODEL_COL_MAP[m] for m in active_models}

    rows_out: list[dict[str, Any]] = []
    if metric_label == "Tempering Schedules (γ_SNR, ε*, γ*)":
        df = pl.read_csv(DATA_DIR / "parameters" / "adaptive_tempering_parameters.csv")
        for r in df.iter_rows(named=True):
            if (
                str(r["dataset"]) not in raw_datasets
                or str(r["model"]) not in raw_models
            ):
                continue
            rows_out.append(
                {
                    "Dataset": RAW_TO_DATASET[str(r["dataset"])],
                    "Model": RAW_TO_SHORT_MODEL[str(r["model"])],
                    "d": int(r["native_d"]),
                    "k_knee (Σ)": int(r["spectemp_knee_k"]),
                    "γ_SNR(512)": f"{float(r['spectemp_gamma_k512']):.2f}",
                    "γ_SNR(256)": f"{float(r['spectemp_gamma_k256']):.2f}",
                    "γ_SNR(128)": f"{float(r['spectemp_gamma_k128']):.2f}",
                    "γ_SNR(64)": f"{float(r['spectemp_gamma_k64']):.2f}",
                    "ε*(d) [×1e-3]": f"{float(r['epsilon_star_full']) * 1e3:.3f}",
                    "γ*(d)": f"{float(r['coast_gamma_full']):.2f}",
                    "ε*(512) [×1e-3]": f"{float(r['epsilon_star_k512']) * 1e3:.3f}",
                    "γ*(512)": f"{float(r['coast_gamma_k512']):.2f}",
                    "ε*(64) [×1e-3]": f"{float(r['epsilon_star_k64']) * 1e3:.3f}",
                    "γ*(64)": f"{float(r['coast_gamma_k64']):.2f}",
                }
            )
    else:
        df = pl.read_csv(DATA_DIR / "parameters" / "energy_and_knee_parameters.csv")
        for r in df.iter_rows(named=True):
            if (
                str(r["dataset"]) not in raw_datasets
                or str(r["model"]) not in raw_models
            ):
                continue
            rows_out.append(
                {
                    "Dataset": RAW_TO_DATASET[str(r["dataset"])],
                    "Model": RAW_TO_SHORT_MODEL[str(r["model"])],
                    "d": int(r["native_d"]),
                    "E_∥ (||μ||₂²)": f"{float(r['crowding_energy_mu_sq']):.4f}",
                    "σ²_∥": f"{float(r['axial_variance_sigma_par_sq']):.4f}",
                    "E_⊥ (Tr(Σ_⊥))": f"{float(r['dispersion_trace']):.4f}",
                    "E_∥ / E_⊥": f"{float(r['crowding_to_dispersion_ratio']):.2f}",
                    "γ_crowd": f"{float(r['gamma_crowd']):.2f}",
                    "cos_unc": f"{float(r['cos_mu_pc1_uncentered']):.4f}",
                    "cos_cen": f"{float(r['cos_mu_pc1_centered']):.4f}",
                    "||U_tail^T μ̂||₂": f"{float(r['centered_tail_leakage_norm']):.4f}",
                    "k_knee (Σ_⊥)": int(r["kneedle_knee_k"]),
                    "λ_knee [×1e-3]": f"{float(r['lambda_knee']) * 1e3:.3f}",
                }
            )

    pdf = pd.DataFrame(rows_out)
    summary = f"**View:** Spectral Parameters | **Table:** {metric_label} | **Rows:** {len(pdf)}"
    return pdf, summary


def on_scope_change(
    eval_view: str,
    track: str,
    dim_mode: str,
    method_group: str,
) -> tuple[Any, ...]:
    """Update dependent selector choices and visibility whenever primary scope controls change."""
    is_task = eval_view == "Task Performance"
    is_geom = eval_view == "Intrinsic Geometry"
    is_param = eval_view == "Spectral Parameters"
    is_comp = dim_mode.startswith("Compression")

    if is_task:
        is_ir = track.startswith("Retrieval")
        metric_choices = list(IR_METRICS.keys()) if is_ir else list(STS_METRICS.keys())
        ds_choices = IR_DATASETS if is_ir else STS_DATASETS
    elif is_geom:
        metric_choices = GEOM_OPTIONS
        ds_choices = ALL_DATASETS
    else:
        metric_choices = PARAM_OPTIONS
        ds_choices = ALL_DATASETS

    method_map = get_method_map(dim_mode, method_group)
    method_choices = list(method_map.values())

    return (
        gr.update(visible=is_task),
        gr.update(choices=metric_choices, value=metric_choices[0]),
        gr.update(visible=not is_param),
        gr.update(visible=(not is_param) and is_comp),
        gr.update(visible=is_task and is_comp),
        gr.update(choices=ds_choices, value=ds_choices),
        gr.update(visible=not is_param),
        gr.update(choices=method_choices, value=method_choices, visible=not is_param),
    )


def render_table(
    eval_view: str,
    track: str,
    metric_label: str,
    dim_mode: str,
    selected_k: list[str],
    comp_display: str,
    datasets: list[str],
    method_group: str,
    selected_methods: list[str],
    selected_cols: list[str],
) -> tuple[pd.DataFrame, str]:
    """Master dispatcher that renders the table for any selector state."""
    if eval_view == "Task Performance":
        return build_task_table(
            track=track,
            metric_label=metric_label,
            dim_mode=dim_mode,
            selected_k=selected_k,
            comp_display=comp_display,
            datasets=datasets,
            method_group=method_group,
            selected_methods=selected_methods,
            selected_cols=selected_cols,
        )
    if eval_view == "Intrinsic Geometry":
        return build_geometry_table(
            metric_label=metric_label,
            dim_mode=dim_mode,
            selected_k=selected_k,
            datasets=datasets,
            method_group=method_group,
            selected_methods=selected_methods,
            selected_cols=selected_cols,
        )
    return build_parameters_table(
        metric_label=metric_label,
        datasets=datasets,
        selected_cols=selected_cols,
    )


def create_app() -> gr.Blocks:
    """Create the single-screen Gradio Blocks interface."""
    default_methods = list(FULL_MAIN_METHODS.values())
    initial_df, initial_summary = render_table(
        eval_view="Task Performance",
        track="Retrieval (IR)",
        metric_label="nDCG@10",
        dim_mode="Full Dimension (k = d)",
        selected_k=["512", "256", "128", "64"],
        comp_display="Score (Retention %)",
        datasets=IR_DATASETS,
        method_group="Main Methods",
        selected_methods=default_methods,
        selected_cols=ALL_DISPLAY_COLS,
    )

    with gr.Blocks(title="COAST Results Explorer") as demo:
        gr.Markdown(
            "## COAST: Text Embedding Post-Processing & Compression Explorer\n"
            "Interactive evaluation across 4 transformer backbones, 2 pooling strategies, "
            "6 MTEB v2 benchmarks, 14 main transformations, and 16 component ablation variants."
        )

        with gr.Row():
            eval_view = gr.Radio(
                choices=[
                    "Task Performance",
                    "Intrinsic Geometry",
                    "Spectral Parameters",
                ],
                value="Task Performance",
                label="Evaluation View",
                scale=3,
            )
            track = gr.Radio(
                choices=["Retrieval (IR)", "Similarity (STS)"],
                value="Retrieval (IR)",
                label="Task Track",
                scale=2,
            )
            metric_label = gr.Dropdown(
                choices=list(IR_METRICS.keys()),
                value="nDCG@10",
                label="Metric / Parameter Table",
                scale=2,
            )
            dim_mode = gr.Radio(
                choices=[
                    "Full Dimension (k = d)",
                    "Compression (k ∈ {512, 256, 128, 64})",
                ],
                value="Full Dimension (k = d)",
                label="Dimension Mode",
                scale=3,
            )

        with gr.Row():
            selected_k = gr.CheckboxGroup(
                choices=["512", "256", "128", "64"],
                value=["512", "256", "128", "64"],
                label="Target Dimensions (k)",
                visible=False,
                scale=2,
            )
            comp_display = gr.Radio(
                choices=["Score (Retention %)", "Absolute Score", "Retention %"],
                value="Score (Retention %)",
                label="Compression Display Format",
                visible=False,
                scale=2,
            )

        with gr.Row():
            datasets = gr.CheckboxGroup(
                choices=IR_DATASETS,
                value=IR_DATASETS,
                label="Benchmarks",
                scale=2,
            )
            method_group = gr.Radio(
                choices=[
                    "Main Methods",
                    "Component Ablations",
                    "All (Main + Ablations)",
                ],
                value="Main Methods",
                label="Method Group",
                scale=2,
            )
            selected_cols = gr.CheckboxGroup(
                choices=ALL_DISPLAY_COLS,
                value=ALL_DISPLAY_COLS,
                label="Displayed Models & Category Means",
                scale=4,
            )

        selected_methods = gr.CheckboxGroup(
            choices=default_methods,
            value=default_methods,
            label="Methods / Operators",
        )

        summary_md = gr.Markdown(value=initial_summary)
        results_table = gr.DataFrame(
            value=initial_df,
            interactive=False,
            wrap=False,
        )

        scope_inputs = [eval_view, track, dim_mode, method_group]
        scope_outputs = [
            track,
            metric_label,
            dim_mode,
            selected_k,
            comp_display,
            datasets,
            method_group,
            selected_methods,
        ]
        all_inputs = [
            eval_view,
            track,
            metric_label,
            dim_mode,
            selected_k,
            comp_display,
            datasets,
            method_group,
            selected_methods,
            selected_cols,
        ]

        for ctrl in scope_inputs:
            ctrl.change(
                fn=on_scope_change,
                inputs=scope_inputs,
                outputs=scope_outputs,
            ).then(
                fn=render_table,
                inputs=all_inputs,
                outputs=[results_table, summary_md],
            )

        for ctrl in [
            metric_label,
            selected_k,
            comp_display,
            datasets,
            selected_methods,
            selected_cols,
        ]:
            ctrl.change(
                fn=render_table,
                inputs=all_inputs,
                outputs=[results_table, summary_md],
            )

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch()
