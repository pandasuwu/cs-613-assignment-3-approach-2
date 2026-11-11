"""Programmatic LaTeX table generator for the ACL 2027 paper (100% orthogonal coverage, zero negative controls)."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger("export_latex")

PAPER_DIR = Path("/Users/devanshlodha/orca/spectral-postprocessing")
CSV_DIR = Path("/Users/devanshlodha/orca/cs-613-assignment-2-approach-2/tables")

MODEL_COLS = [
    ("google/embeddinggemma-300m", "EmbGemma"),
    ("Qwen/Qwen3-Embedding-0.6B", "Qwen3-Emb"),
    ("google/gemma-3-1b-pt (mean_pooling)", "Gemma3-M"),
    ("google/gemma-3-1b-pt (last_token_pooling)", "Gemma3-L"),
    ("Qwen/Qwen3.5-0.8B-Base (mean_pooling)", "Qwen3.5-M"),
    ("Qwen/Qwen3.5-0.8B-Base (last_token_pooling)", "Qwen3.5-L"),
]

METHOD_LATEX = {
    "baseline": "Baseline",
    "standardization": "Standardization",
    "r1": "R1",
    "r2": "R2",
    "abtt_1": "ABTT ($D{=}1$)",
    "abtt_2": "ABTT ($D{=}2$)",
    "abtt_3": "ABTT ($D{=}3$)",
    "soft_zca": "Soft-ZCA",
    "coast": "\\textbf{COAST (Ours)}",
    "prefix": "Prefix (MRL)",
    "pca": "PCA-$k$",
    "whitening": "Whitening-$k$",
    "spectemp": "SpecTemp",
    # Full-dimension ablation variants (human-readable)
    "ablation_o0_c0_e2": "Unreg. ZCA ($\\Sigma, 1, 10^{-6}$)",
    "ablation_o0_c0_e1": "Centered + $\\epsilon^*$ ($\\Sigma, 1, \\epsilon^*$)",
    "ablation_o0_c1_e0": "Centered + $\\gamma_{\\text{crowd}}$ ($\\Sigma$)",
    "ablation_o0_c1_e1": "w/o Orth. Proj. ($\\Sigma, \\gamma_{\\text{crowd}}$)",
    "ablation_o1_c0_e0": "Orth. Soft-ZCA ($\\Sigma_\\perp, 10^{-4}$)",
    "ablation_o1_c0_e1": "w/o Crowd. Exp. ($\\Sigma_\\perp, \\epsilon^*$)",
    "ablation_o1_c2_e0": "Fixed $\\gamma{=}0.5$ ($\\Sigma_\\perp, 0.5$)",
    "ablation_o1_c1_e0": "w/o Adaptive $\\epsilon^*$ ($\\Sigma_\\perp, 10^{-4}$)",
}

COMP_ABLATION_LATEX = {
    "ablation_o1_pca": "Orth. PCA ($\\Sigma_\\perp, \\gamma{=}0$)",
    "ablation_o1_c0_e0": "Orth. SpecTemp ($\\Sigma_\\perp, 10^{-6}$)",
    "ablation_o1_c0_e2": "w/o Crowd. Floor ($\\gamma_{\\text{SNR}}$)",
    "ablation_o1_c4_e2": "Fixed $\\gamma{=}0.5$ ($\\Sigma_\\perp, 0.5$)",
    "ablation_o1_c2_e2": "w/o SNR Sched. ($\\gamma_{\\text{crowd}}$)",
    "ablation_o1_c1_e0": "w/o Adaptive $\\epsilon^*$ ($10^{-6}$)",
    "ablation_o1_c1_e1": "w/o Rank Scale $\\tau^2$ ($\\epsilon^*(d)$)",
    "ablation_o0_c1_e2": "w/o Orth. Proj. ($\\Sigma, \\gamma^*$)",
}


def _f(val: object) -> float:
    return float(str(val))


def fmt_method(m: str, is_comp_ablation: bool = False) -> str:
    if is_comp_ablation and m in COMP_ABLATION_LATEX:
        return COMP_ABLATION_LATEX[m]
    if m in METHOD_LATEX:
        return METHOD_LATEX[m]
    return m.replace("_", "\\_")


def fmt_dataset(ds: str) -> str:
    mapping = {
        "FiQA2018": "FiQA-2018",
        "ArguAna": "ArguAna",
        "SCIDOCS": "SCIDOCS",
        "STSBenchmark": "STS-B",
        "SICK-R": "SICK-R",
        "STS22.v2": "STS22.v2",
    }
    return mapping.get(ds, ds)


def fmt_model_short(m: str) -> str:
    for full, short in MODEL_COLS:
        if m == full:
            return short
    return m.replace("_", "\\_")


def generate_appendix_retrieval_and_similarity() -> list[str]:
    """Generate Appendix B (Tables A1-A9) and Appendix C (Tables A10-A15) in two-column table* mode."""
    out_dir_1 = PAPER_DIR / "tables" / "appendix" / "retrieval"
    out_dir_2 = PAPER_DIR / "tables" / "appendix" / "similarity"
    out_dir_1.mkdir(parents=True, exist_ok=True)
    out_dir_2.mkdir(parents=True, exist_ok=True)

    suite_1_specs = [
        ("fiqa_ndcg.csv", "FiQA-2018", "nDCG@10", 4),
        ("fiqa_mrr.csv", "FiQA-2018", "MRR@10", 4),
        ("fiqa_recall.csv", "FiQA-2018", "Recall@100", 4),
        ("arguana_ndcg.csv", "ArguAna", "nDCG@10", 4),
        ("arguana_mrr.csv", "ArguAna", "MRR@10", 4),
        ("arguana_recall.csv", "ArguAna", "Recall@100", 4),
        ("scidocs_ndcg.csv", "SCIDOCS", "nDCG@10", 4),
        ("scidocs_mrr.csv", "SCIDOCS", "MRR@10", 4),
        ("scidocs_recall.csv", "SCIDOCS", "Recall@100", 4),
    ]
    suite_2_specs = [
        ("stsb_spearman.csv", "STSBenchmark", "Spearman $\\rho \\times 100$", 2),
        ("stsb_pearson.csv", "STSBenchmark", "Pearson $r \\times 100$", 2),
        ("sickr_spearman.csv", "SICK-R", "Spearman $\\rho \\times 100$", 2),
        ("sickr_pearson.csv", "SICK-R", "Pearson $r \\times 100$", 2),
        ("sts22_spearman.csv", "STS22.v2", "Spearman $\\rho \\times 100$", 2),
        ("sts22_pearson.csv", "STS22.v2", "Pearson $r \\times 100$", 2),
    ]

    all_inputs: list[str] = []
    idx = 1
    for specs, subdir, folder in [
        (suite_1_specs, "retrieval", out_dir_1),
        (suite_2_specs, "similarity", out_dir_2),
    ]:
        for fname, ds_label, metric_label, dec in specs:
            df = pl.read_csv(CSV_DIR / subdir / fname)
            num_cols = [c for c, _ in MODEL_COLS] + [
                "mean_dedicated",
                "mean_base_mean",
                "mean_base_last",
            ]
            max_vals = {c: _f(df[c].max()) for c in num_cols}

            lines = [
                "\\begin{table*}[t]",
                "\\centering",
                "\\footnotesize",
                "\\setlength{\\tabcolsep}{5.0pt}",
                "\\renewcommand{\\arraystretch}{1.02}",
                f"\\caption{{Full-dimension ($k=d$) evaluation on \\textbf{{{ds_label}}} measured by {metric_label}. Bold indicates the best score per column.}}",
                f"\\label{{tab:app_a{idx:02d}_{fname.replace('.csv', '')}}}",
                "\\begin{tabular}{lccccccccc}",
                "\\toprule",
                " & \\multicolumn{2}{c}{\\textbf{Dedicated Encoders}} & \\multicolumn{2}{c}{\\textbf{Gemma-3-1B-pt}} & \\multicolumn{2}{c}{\\textbf{Qwen3.5-0.8B-Base}} & \\multicolumn{3}{c}{\\textbf{Category Means}} \\\\",
                "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-10}",
                "\\textbf{Method} & \\textbf{EmbGemma} & \\textbf{Qwen3-Emb} & \\textbf{Mean} & \\textbf{Last} & \\textbf{Mean} & \\textbf{Last} & \\textbf{Ded.} & \\textbf{Base-M} & \\textbf{Base-L} \\\\",
                "\\midrule",
            ]
            for row in df.iter_rows(named=True):
                m_str = fmt_method(str(row["method"]))
                cells = [m_str]
                for c in num_cols:
                    val = float(row[c])
                    s = f"{val:.{dec}f}"
                    if abs(val - max_vals[c]) < 10 ** (-dec):
                        s = f"\\textbf{{{s}}}"
                    cells.append(s)
                lines.append(" & ".join(cells) + " \\\\")
            lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
            tex_name = f"a{idx:02d}_{fname.replace('.csv', '.tex')}"
            (folder / tex_name).write_text("\n".join(lines) + "\n")
            all_inputs.append(f"tables/appendix/{folder.name}/{tex_name}")
            idx += 1
    return all_inputs


def generate_appendix_compression() -> list[str]:
    """Generate Appendix D (Tables A16-A30): Dimensional compression ladders (20 rows each, 2 per page)."""
    out_dir = PAPER_DIR / "tables" / "appendix" / "compression"
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("fiqa_compression_ndcg.csv", "FiQA-2018", "nDCG@10", 4),
        ("fiqa_compression_mrr.csv", "FiQA-2018", "MRR@10", 4),
        ("fiqa_compression_recall.csv", "FiQA-2018", "Recall@100", 4),
        ("arguana_compression_ndcg.csv", "ArguAna", "nDCG@10", 4),
        ("arguana_compression_mrr.csv", "ArguAna", "MRR@10", 4),
        ("arguana_compression_recall.csv", "ArguAna", "Recall@100", 4),
        ("scidocs_compression_ndcg.csv", "SCIDOCS", "nDCG@10", 4),
        ("scidocs_compression_mrr.csv", "SCIDOCS", "MRR@10", 4),
        ("scidocs_compression_recall.csv", "SCIDOCS", "Recall@100", 4),
        (
            "stsb_compression_spearman.csv",
            "STSBenchmark",
            "Spearman $\\rho \\times 100$",
            2,
        ),
        ("stsb_compression_pearson.csv", "STSBenchmark", "Pearson $r \\times 100$", 2),
        ("sickr_compression_spearman.csv", "SICK-R", "Spearman $\\rho \\times 100$", 2),
        ("sickr_compression_pearson.csv", "SICK-R", "Pearson $r \\times 100$", 2),
        (
            "sts22_compression_spearman.csv",
            "STS22.v2",
            "Spearman $\\rho \\times 100$",
            2,
        ),
        ("sts22_compression_pearson.csv", "STS22.v2", "Pearson $r \\times 100$", 2),
    ]
    inputs: list[str] = []
    idx = 16
    for fname, ds_label, metric_label, dec in specs:
        df = pl.read_csv(CSV_DIR / "compression" / fname)
        lines = [
            "\\begin{table*}[t]",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{1.3pt}",
            "\\renewcommand{\\arraystretch}{0.94}",
            f"\\caption{{Dimensional compression ladder ($k \\in \\{{512, 256, 128, 64\\}}$) on \\textbf{{{ds_label}}} measured by {metric_label} and full-dimension retention percentage.}}",
            f"\\label{{tab:app_a{idx:02d}_{fname.replace('.csv', '')}}}",
            "\\begin{tabular}{clccccccccc}",
            "\\toprule",
            " & & \\multicolumn{6}{c}{\\textbf{Score (Full-Dimension Retention \\%)}} & \\multicolumn{3}{c}{\\textbf{Category Mean Retention (\\%)}} \\\\",
            "\\cmidrule(lr){3-8} \\cmidrule(lr){9-11}",
            "$k$ & \\textbf{Method} & \\textbf{EmbGemma} & \\textbf{Qwen3-Emb} & \\textbf{Gemma3-M} & \\textbf{Gemma3-L} & \\textbf{Qwen3.5-M} & \\textbf{Qwen3.5-L} & \\textbf{Ded.} & \\textbf{Base-M} & \\textbf{Base-L} \\\\",
            "\\midrule",
        ]
        for k_idx, k_val in enumerate([512, 256, 128, 64]):
            sub = df.filter(pl.col("k") == k_val)
            max_scores = {c: _f(sub[c].max()) for c, _ in MODEL_COLS}
            max_tiers = {
                c: _f(sub[c].max())
                for c in [
                    "mean_dedicated_retention_pct",
                    "mean_base_mean_retention_pct",
                    "mean_base_last_retention_pct",
                ]
            }
            if k_idx > 0:
                lines.append("\\midrule")
            for r_idx, row in enumerate(sub.iter_rows(named=True)):
                k_cell = f"\\multirow{{5}}{{*}}{{{k_val}}}" if r_idx == 0 else ""
                m_cell = fmt_method(str(row["method"]))
                cells = [k_cell, m_cell]
                for c, _ in MODEL_COLS:
                    sc = float(row[c])
                    ret = float(row[f"{c} (retention %)"])
                    sc_s = f"{sc:.{dec}f}"
                    ret_s = f"{ret:.1f}\\%"
                    if abs(sc - max_scores[c]) < 10 ** (-dec):
                        cells.append(f"\\textbf{{{sc_s}}} ({ret_s})")
                    else:
                        cells.append(f"{sc_s} ({ret_s})")
                for tc in [
                    "mean_dedicated_retention_pct",
                    "mean_base_mean_retention_pct",
                    "mean_base_last_retention_pct",
                ]:
                    tv = float(row[tc])
                    ts = f"{tv:.1f}\\%"
                    if abs(tv - max_tiers[tc]) < 0.05:
                        ts = f"\\textbf{{{ts}}}"
                    cells.append(ts)
                lines.append(" & ".join(cells) + " \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
        tex_name = f"a{idx:02d}_{fname.replace('.csv', '.tex')}"
        (out_dir / tex_name).write_text("\n".join(lines) + "\n")
        inputs.append(f"tables/appendix/compression/{tex_name}")
        idx += 1
    return inputs


def generate_appendix_geometry() -> list[str]:
    """Generate Appendix E (Tables A31-A42): 6 full-dimension (27x2 side-by-side) + 6 compression (30x2 side-by-side)."""
    out_dir = PAPER_DIR / "tables" / "appendix" / "geometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_list = [
        ("fiqa", "FiQA-2018"),
        ("arguana", "ArguAna"),
        ("scidocs", "SCIDOCS"),
        ("stsb", "STSBenchmark"),
        ("sickr", "SICK-R"),
        ("sts22", "STS22.v2"),
    ]
    inputs: list[str] = []
    idx = 31

    models_left = [m for m, _ in MODEL_COLS[:3]]
    models_right = [m for m, _ in MODEL_COLS[3:]]

    # 1. Full-dimension geometry (Tables A31-A36, 54 rows -> two 27-row panels per table*)
    for prefix, ds_label in ds_list:
        fname = f"{prefix}_geometry_full.csv"
        df = pl.read_csv(CSV_DIR / "geometry" / fname)
        df_l = df.filter(pl.col("model").is_in(models_left))
        df_r = df.filter(pl.col("model").is_in(models_right))
        rows_l = list(df_l.iter_rows(named=True))
        rows_r = list(df_r.iter_rows(named=True))

        lines = [
            "\\begin{table*}[t]",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{2.4pt}",
            "\\renewcommand{\\arraystretch}{0.96}",
            f"\\caption{{Intrinsic geometry diagnostics at full dimensionality ($k=d$) on \\textbf{{{ds_label}}} across all 6 model configurations and 9 methods.}}",
            f"\\label{{tab:app_a{idx:02d}_{prefix}_geom_full}}",
            "\\begin{tabular}{llccccc@{\\hspace{8pt}}llccccc}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{Method} & $\\|\\mu\\|_2$ & \\textbf{Cos} & \\textbf{MEV} & \\textbf{NID} & \\textbf{Iso} & \\textbf{Model} & \\textbf{Method} & $\\|\\mu\\|_2$ & \\textbf{Cos} & \\textbf{MEV} & \\textbf{NID} & \\textbf{Iso} \\\\",
            "\\midrule",
        ]
        for r_i in range(len(rows_l)):
            if r_i > 0 and r_i % 9 == 0:
                lines.append("\\midrule")
            rl = rows_l[r_i]
            rr = rows_r[r_i]
            ml_cell = fmt_model_short(str(rl["model"])) if r_i % 9 == 0 else ""
            mr_cell = fmt_model_short(str(rr["model"])) if r_i % 9 == 0 else ""
            ls = f"{ml_cell} & {fmt_method(str(rl['method']))} & {float(rl['centroid_norm']):.3f} & {float(rl['average_cosine']):.3f} & {float(rl['mev_top1']):.3f} & {float(rl['nid_spectral_entropy']):.3f} & {float(rl['isoscore']):.3f}"
            rs = f"{mr_cell} & {fmt_method(str(rr['method']))} & {float(rr['centroid_norm']):.3f} & {float(rr['average_cosine']):.3f} & {float(rr['mev_top1']):.3f} & {float(rr['nid_spectral_entropy']):.3f} & {float(rr['isoscore']):.3f}"
            lines.append(f"{ls} & {rs} \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
        tex_name = f"a{idx:02d}_{prefix}_geometry_full.tex"
        (out_dir / tex_name).write_text("\n".join(lines) + "\n")
        inputs.append(f"tables/appendix/geometry/{tex_name}")
        idx += 1

    # 2. Compression geometry (Tables A37-A42, 120 rows -> two 30-row panels per k-pair)
    for prefix, ds_label in ds_list:
        fname = f"{prefix}_geometry_compression.csv"
        df = pl.read_csv(CSV_DIR / "geometry" / fname)
        part_lines: list[str] = []
        for part_idx, k_pair in [(1, [512, 256]), (2, [128, 64])]:
            sub = df.filter(pl.col("k").is_in(k_pair))
            df_l = sub.filter(pl.col("model").is_in(models_left))
            df_r = sub.filter(pl.col("model").is_in(models_right))
            rows_l = list(df_l.iter_rows(named=True))
            rows_r = list(df_r.iter_rows(named=True))
            cnt_cmd = "" if part_idx == 1 else "\\addtocounter{table}{-1}"
            lbl_cmd = (
                f"\\label{{tab:app_a{idx:02d}_{prefix}_geom_comp}}"
                if part_idx == 1
                else ""
            )
            part_lines.extend(
                [
                    "\\begin{table*}[t]",
                    "\\centering",
                    "\\scriptsize",
                    "\\setlength{\\tabcolsep}{2.0pt}",
                    "\\renewcommand{\\arraystretch}{0.92}",
                    f"{cnt_cmd}\\caption{{Intrinsic geometry diagnostics under dimensional compression ($k \\in \\{{{k_pair[0]}, {k_pair[1]}\\}}$) on \\textbf{{{ds_label}}} across all 6 model configurations (Part {part_idx} of 2).}}{lbl_cmd}",
                    "\\begin{tabular}{cllccccc@{\\hspace{6pt}}cllccccc}",
                    "\\toprule",
                    "$k$ & \\textbf{Model} & \\textbf{Method} & $\\|\\mu\\|_2$ & \\textbf{Cos} & \\textbf{MEV} & \\textbf{NID} & \\textbf{Iso} & $k$ & \\textbf{Model} & \\textbf{Method} & $\\|\\mu\\|_2$ & \\textbf{Cos} & \\textbf{MEV} & \\textbf{NID} & \\textbf{Iso} \\\\",
                    "\\midrule",
                ]
            )
            for r_i in range(len(rows_l)):
                if r_i > 0 and r_i % 5 == 0:
                    part_lines.append("\\midrule")
                rl = rows_l[r_i]
                rr = rows_r[r_i]
                kl_cell = str(rl["k"]) if r_i % 15 == 0 else ""
                kr_cell = str(rr["k"]) if r_i % 15 == 0 else ""
                ml_cell = fmt_model_short(str(rl["model"])) if r_i % 5 == 0 else ""
                mr_cell = fmt_model_short(str(rr["model"])) if r_i % 5 == 0 else ""
                ls = f"{kl_cell} & {ml_cell} & {fmt_method(str(rl['method']))} & {float(rl['centroid_norm']):.3f} & {float(rl['average_cosine']):.3f} & {float(rl['mev_top1']):.3f} & {float(rl['nid_spectral_entropy']):.3f} & {float(rl['isoscore']):.3f}"
                rs = f"{kr_cell} & {mr_cell} & {fmt_method(str(rr['method']))} & {float(rr['centroid_norm']):.3f} & {float(rr['average_cosine']):.3f} & {float(rr['mev_top1']):.3f} & {float(rr['nid_spectral_entropy']):.3f} & {float(rr['isoscore']):.3f}"
                part_lines.append(f"{ls} & {rs} \\\\")
            part_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
        tex_name = f"a{idx:02d}_{prefix}_geometry_compression.tex"
        (out_dir / tex_name).write_text("\n".join(part_lines) + "\n")
        inputs.append(f"tables/appendix/geometry/{tex_name}")
        idx += 1

    return inputs


def generate_appendix_parameters() -> list[str]:
    """Generate Appendix F (Tables A43 and A44): Energy/knee parameters and adaptive tempering parameters."""
    out_dir = PAPER_DIR / "tables" / "appendix" / "parameters"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_e = pl.read_csv(CSV_DIR / "parameters" / "energy_and_knee_parameters.csv")
    lines_43 = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{1.4pt}",
        "\\renewcommand{\\arraystretch}{0.92}",
        "\\caption{Empirical unit-sphere energy partition ($E_\\parallel, \\sigma_\\parallel^2, E_\\perp$), centered covariance mean-leakage diagnostics, and orthogonal spectral knee statistics across all 36 model-dataset-pooling configurations.}",
        "\\label{tab:app_a43_energy_parameters}",
        "\\begin{tabular}{llcccccccccccccc}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{Model} & $d$ & $E_\\parallel$ & $\\sigma_\\parallel^2$ & $E_\\perp$ & $\\bar{\\lambda}_\\perp {\\times} 10^3$ & $E_\\parallel/E_\\perp$ & $\\gamma_{\\text{crowd}}$ & $\\cos_{\\text{unc}}$ & $\\cos_{\\text{cen}}$ & $\\|U_{\\text{tail}}^\\top\\hat{\\mu}\\|_2$ & $\\lambda_{\\perp,1}$ & $\\sigma_{\\text{noise}}^2 {\\times} 10^4$ & $k_{\\text{knee}}$ & $\\lambda_{\\text{knee}}{\\times}10^3$ \\\\",
        "\\midrule",
    ]
    for r_i, row in enumerate(df_e.iter_rows(named=True)):
        if r_i > 0 and r_i % 6 == 0:
            lines_43.append("\\midrule")
        ds_cell = fmt_dataset(str(row["dataset"])) if r_i % 6 == 0 else ""
        m_cell = fmt_model_short(str(row["model"]))
        lines_43.append(
            f"{ds_cell} & {m_cell} & {int(row['native_d'])} & {float(row['crowding_energy_mu_sq']):.4f} & {float(row['axial_variance_sigma_par_sq']):.4f} & {float(row['dispersion_trace']):.4f} & {float(row['mean_eigenvalue']) * 1e3:.3f} & {float(row['crowding_to_dispersion_ratio']):.2f} & {float(row['gamma_crowd']):.4f} & {float(row['cos_mu_pc1_uncentered']):.4f} & {float(row['cos_mu_pc1_centered']):.4f} & {float(row['centered_tail_leakage_norm']):.4f} & {float(row['lambda_perp_1']):.4f} & {float(row['noise_floor_variance']) * 1e4:.3f} & {int(row['kneedle_knee_k'])} & {float(row['lambda_knee']) * 1e3:.2f} \\\\"
        )
    lines_43.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    (out_dir / "a43_energy_and_knee_parameters.tex").write_text(
        "\n".join(lines_43) + "\n"
    )

    df_t = pl.read_csv(CSV_DIR / "parameters" / "adaptive_tempering_parameters.csv")
    lines_44 = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.4pt}",
        "\\renewcommand{\\arraystretch}{0.92}",
        "\\caption{Derived SpecTemp exponents $\\gamma_{\\text{SNR}}(k)$, COAST adaptive regularizers $\\epsilon^*(k) \\times 10^3$, and COAST tempering exponents $\\gamma^*(k)$ across all 36 model-dataset-pooling configurations.}",
        "\\label{tab:app_a44_tempering_parameters}",
        "\\begin{tabular}{llccccccccccccccccc}",
        "\\toprule",
        " & & & & \\multicolumn{4}{c}{\\textbf{SpecTemp} $\\gamma_{\\text{SNR}}(k)$} & \\multicolumn{5}{c}{\\textbf{COAST} $\\epsilon^*(k) \\times 10^3$} & \\multicolumn{5}{c}{\\textbf{COAST} $\\gamma^*(k)$} \\\\",
        "\\cmidrule(lr){5-8} \\cmidrule(lr){9-13} \\cmidrule(lr){14-18}",
        "\\textbf{Dataset} & \\textbf{Model} & $d$ & $k_{\\text{knee}}^{\\text{raw}}$ & $512$ & $256$ & $128$ & $64$ & $d$ & $512$ & $256$ & $128$ & $64$ & $d$ & $512$ & $256$ & $128$ & $64$ \\\\",
        "\\midrule",
    ]
    for r_i, row in enumerate(df_t.iter_rows(named=True)):
        if r_i > 0 and r_i % 6 == 0:
            lines_44.append("\\midrule")
        ds_cell = fmt_dataset(str(row["dataset"])) if r_i % 6 == 0 else ""
        m_cell = fmt_model_short(str(row["model"]))
        lines_44.append(
            f"{ds_cell} & {m_cell} & {int(row['native_d'])} & {int(row['spectemp_knee_k'])} & {float(row['spectemp_gamma_k512']):.2f} & {float(row['spectemp_gamma_k256']):.2f} & {float(row['spectemp_gamma_k128']):.2f} & {float(row['spectemp_gamma_k64']):.2f} & {float(row['epsilon_star_full']) * 1e3:.3f} & {float(row['epsilon_star_k512']) * 1e3:.3f} & {float(row['epsilon_star_k256']) * 1e3:.3f} & {float(row['epsilon_star_k128']) * 1e3:.3f} & {float(row['epsilon_star_k64']) * 1e3:.3f} & {float(row['coast_gamma_full']):.2f} & {float(row['coast_gamma_k512']):.2f} & {float(row['coast_gamma_k256']):.2f} & {float(row['coast_gamma_k128']):.2f} & {float(row['coast_gamma_k64']):.2f} \\\\"
        )
    lines_44.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    (out_dir / "a44_adaptive_tempering_parameters.tex").write_text(
        "\n".join(lines_44) + "\n"
    )
    return [
        "tables/appendix/parameters/a43_energy_and_knee_parameters.tex",
        "tables/appendix/parameters/a44_adaptive_tempering_parameters.tex",
    ]


def generate_appendix_ablations() -> list[str]:
    """Generate Appendix G (Tables A45-A56): 12 component ablation tables covering all 6 models with human names."""
    out_dir = PAPER_DIR / "tables" / "appendix" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_f in out_dir.glob("*.tex"):
        old_f.unlink()
    inputs: list[str] = []

    # 1. Full-dimension retrieval & similarity ablations (Tables A45-A49, 24 rows x 11 cols each)
    full_specs = [
        (
            45,
            "retrieval_ablation_ndcg.csv",
            "Full-Dimension Retrieval Component Ablation (nDCG@10)",
            4,
        ),
        (
            46,
            "retrieval_ablation_mrr.csv",
            "Full-Dimension Retrieval Component Ablation (MRR@10)",
            4,
        ),
        (
            47,
            "retrieval_ablation_recall.csv",
            "Full-Dimension Retrieval Component Ablation (Recall@100)",
            4,
        ),
        (
            48,
            "similarity_ablation_spearman.csv",
            "Full-Dimension Similarity Component Ablation (Spearman $\\rho \\times 100$)",
            2,
        ),
        (
            49,
            "similarity_ablation_pearson.csv",
            "Full-Dimension Similarity Component Ablation (Pearson $r \\times 100$)",
            2,
        ),
    ]
    for idx, fname, title, dec in full_specs:
        df = pl.read_csv(CSV_DIR / "ablation" / fname)
        lines = [
            "\\begin{table*}[t]",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{2.4pt}",
            "\\renewcommand{\\arraystretch}{0.95}",
            f"\\caption{{{title} across the 8 component ablation variants per benchmark.}}",
            f"\\label{{tab:app_a{idx:02d}_{fname.replace('.csv', '')}}}",
            "\\begin{tabular}{llccccccccc}",
            "\\toprule",
            "\\textbf{Dataset} & \\textbf{Ablation Variant} & \\textbf{EmbGemma} & \\textbf{Qwen3-Emb} & \\textbf{Gemma3-M} & \\textbf{Gemma3-L} & \\textbf{Qwen3.5-M} & \\textbf{Qwen3.5-L} & \\textbf{Ded.} & \\textbf{Base-M} & \\textbf{Base-L} \\\\",
            "\\midrule",
        ]
        for r_i, row in enumerate(df.iter_rows(named=True)):
            if r_i > 0 and r_i % 8 == 0:
                lines.append("\\midrule")
            ds_cell = fmt_dataset(str(row["dataset"])) if r_i % 8 == 0 else ""
            cells = [ds_cell, fmt_method(str(row["method"]))]
            for c, _ in MODEL_COLS:
                cells.append(f"{float(row[c]):.{dec}f}")
            for tc in ["mean_dedicated", "mean_base_mean", "mean_base_last"]:
                cells.append(f"{float(row[tc]):.{dec}f}")
            lines.append(" & ".join(cells) + " \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
        tex_name = f"a{idx:02d}_{fname.replace('.csv', '.tex')}"
        (out_dir / tex_name).write_text("\n".join(lines) + "\n")
        inputs.append(f"tables/appendix/ablation/{tex_name}")

    # 2. Compression component ablations (Tables A50-A54, 96 rows -> 3 dataset sub-tables of 32 rows x 11 cols each!)
    comp_specs = [
        (
            50,
            "compression_ablation_ndcg.csv",
            "Retrieval Compression Component Ablation (nDCG@10)",
            4,
        ),
        (
            51,
            "compression_ablation_mrr.csv",
            "Retrieval Compression Component Ablation (MRR@10)",
            4,
        ),
        (
            52,
            "compression_ablation_recall.csv",
            "Retrieval Compression Component Ablation (Recall@100)",
            4,
        ),
        (
            53,
            "compression_ablation_spearman.csv",
            "Similarity Compression Component Ablation (Spearman $\\rho \\times 100$)",
            2,
        ),
        (
            54,
            "compression_ablation_pearson.csv",
            "Similarity Compression Component Ablation (Pearson $r \\times 100$)",
            2,
        ),
    ]
    for idx, fname, title, dec in comp_specs:
        df = pl.read_csv(CSV_DIR / "ablation" / fname)
        datasets = df["dataset"].unique(maintain_order=True).to_list()
        file_lines: list[str] = []
        for d_i, ds in enumerate(datasets):
            sub_ds = df.filter(pl.col("dataset") == ds)
            cnt_cmd = "" if d_i == 0 else "\\addtocounter{table}{-1}"
            lbl_cmd = (
                f"\\label{{tab:app_a{idx:02d}_{fname.replace('.csv', '')}}}"
                if d_i == 0
                else ""
            )
            file_lines.extend(
                [
                    "\\begin{table*}[t]",
                    "\\centering",
                    "\\scriptsize",
                    "\\setlength{\\tabcolsep}{1.3pt}",
                    "\\renewcommand{\\arraystretch}{0.94}",
                    f"{cnt_cmd}\\caption{{{title} on \\textbf{{{fmt_dataset(ds)}}} across $k \\in \\{{512, 256, 128, 64\\}}$ and the 8 component ablation variants (Part {d_i + 1} of 3).}}{lbl_cmd}",
                    "\\begin{tabular}{clccccccccc}",
                    "\\toprule",
                    " & & \\multicolumn{6}{c}{\\textbf{Score (Full-Dimension Retention \\%)}} & \\multicolumn{3}{c}{\\textbf{Category Mean Retention (\\%)}} \\\\",
                    "\\cmidrule(lr){3-8} \\cmidrule(lr){9-11}",
                    "$k$ & \\textbf{Ablation Variant} & \\textbf{EmbGemma} & \\textbf{Qwen3-Emb} & \\textbf{Gemma3-M} & \\textbf{Gemma3-L} & \\textbf{Qwen3.5-M} & \\textbf{Qwen3.5-L} & \\textbf{Ded.} & \\textbf{Base-M} & \\textbf{Base-L} \\\\",
                    "\\midrule",
                ]
            )
            for r_i, row in enumerate(sub_ds.iter_rows(named=True)):
                if r_i > 0 and r_i % 8 == 0:
                    file_lines.append("\\midrule")
                k_cell = f"\\multirow{{8}}{{*}}{{{row['k']}}}" if r_i % 8 == 0 else ""
                cells = [
                    k_cell,
                    fmt_method(str(row["method"]), is_comp_ablation=True),
                ]
                for c, _ in MODEL_COLS:
                    sc = float(row[c])
                    ret = float(row[f"{c} (retention %)"])
                    cells.append(f"{sc:.{dec}f} ({ret:.1f})")
                for tc in [
                    "mean_dedicated_retention_pct",
                    "mean_base_mean_retention_pct",
                    "mean_base_last_retention_pct",
                ]:
                    cells.append(f"{float(row[tc]):.1f}")
                file_lines.append(" & ".join(cells) + " \\\\")
            file_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
        tex_name = f"a{idx:02d}_{fname.replace('.csv', '.tex')}"
        (out_dir / tex_name).write_text("\n".join(file_lines) + "\n")
        inputs.append(f"tables/appendix/ablation/{tex_name}")

    # 3. Geometry component ablations (Table A55 full 288 rows -> 6 dataset tables of 24x2 panels; Table A56 compression 1152 rows)
    models_left = [m for m, _ in MODEL_COLS[:3]]
    models_right = [m for m, _ in MODEL_COLS[3:]]

    df_gf = pl.read_csv(CSV_DIR / "ablation" / "geometry_ablation_full.csv")
    datasets_gf = df_gf["dataset"].unique(maintain_order=True).to_list()
    lines_55: list[str] = []
    for d_i, ds in enumerate(datasets_gf):
        sub = df_gf.filter(pl.col("dataset") == ds)
        rows_l = list(
            sub.filter(pl.col("model").is_in(models_left)).iter_rows(named=True)
        )
        rows_r = list(
            sub.filter(pl.col("model").is_in(models_right)).iter_rows(named=True)
        )
        cnt_cmd = "" if d_i == 0 else "\\addtocounter{table}{-1}"
        lbl_cmd = "\\label{tab:app_a55_geometry_ablation_full}" if d_i == 0 else ""
        lines_55.extend(
            [
                "\\begin{table*}[t]",
                "\\centering",
                "\\scriptsize",
                "\\setlength{\\tabcolsep}{1.5pt}",
                "\\renewcommand{\\arraystretch}{0.94}",
                f"{cnt_cmd}\\caption{{Intrinsic geometry component ablation at full dimensionality ($k=d$) on \\textbf{{{fmt_dataset(ds)}}} across all 6 models and 8 variants (Part {d_i + 1} of 6).}}{lbl_cmd}",
                "\\begin{tabular}{llccccc@{\\hspace{5pt}}llccccc}",
                "\\toprule",
                "\\textbf{Model} & \\textbf{Ablation Variant} & $\\|\\mu\\|_2$ & \\textbf{Cos} & \\textbf{MEV} & \\textbf{NID} & \\textbf{Iso} & \\textbf{Model} & \\textbf{Ablation Variant} & $\\|\\mu\\|_2$ & \\textbf{Cos} & \\textbf{MEV} & \\textbf{NID} & \\textbf{Iso} \\\\",
                "\\midrule",
            ]
        )
        for r_i in range(len(rows_l)):
            if r_i > 0 and r_i % 8 == 0:
                lines_55.append("\\midrule")
            rl = rows_l[r_i]
            rr = rows_r[r_i]
            ml = fmt_model_short(str(rl["model"])) if r_i % 8 == 0 else ""
            mr = fmt_model_short(str(rr["model"])) if r_i % 8 == 0 else ""
            ls = f"{ml} & {fmt_method(str(rl['method']))} & {float(rl['centroid_norm']):.3f} & {float(rl['average_cosine']):.3f} & {float(rl['mev_top1']):.3f} & {float(rl['nid_spectral_entropy']):.3f} & {float(rl['isoscore']):.3f}"
            rs = f"{mr} & {fmt_method(str(rr['method']))} & {float(rr['centroid_norm']):.3f} & {float(rr['average_cosine']):.3f} & {float(rr['mev_top1']):.3f} & {float(rr['nid_spectral_entropy']):.3f} & {float(rr['isoscore']):.3f}"
            lines_55.append(f"{ls} & {rs} \\\\")
        lines_55.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    (out_dir / "a55_geometry_ablation_full.tex").write_text("\n".join(lines_55) + "\n")
    inputs.append("tables/appendix/ablation/a55_geometry_ablation_full.tex")

    df_gc = pl.read_csv(CSV_DIR / "ablation" / "geometry_ablation_compression.csv")
    datasets_gc = df_gc["dataset"].unique(maintain_order=True).to_list()
    lines_56: list[str] = []
    part_num = 1
    for ds in datasets_gc:
        for k_pair in [[512, 256], [128, 64]]:
            sub = df_gc.filter((pl.col("dataset") == ds) & pl.col("k").is_in(k_pair))
            rows_l = list(
                sub.filter(pl.col("model").is_in(models_left)).iter_rows(named=True)
            )
            rows_r = list(
                sub.filter(pl.col("model").is_in(models_right)).iter_rows(named=True)
            )
            cnt_cmd = "" if part_num == 1 else "\\addtocounter{table}{-1}"
            lbl_cmd = (
                "\\label{tab:app_a56_geometry_ablation_compression}"
                if part_num == 1
                else ""
            )
            lines_56.extend(
                [
                    "\\begin{table*}[t]",
                    "\\centering",
                    "\\scriptsize",
                    "\\setlength{\\tabcolsep}{0.75pt}",
                    "\\renewcommand{\\arraystretch}{0.84}",
                    f"{cnt_cmd}\\caption{{Intrinsic geometry component ablation under dimensional compression ($k \\in \\{{{k_pair[0]}, {k_pair[1]}\\}}$) on \\textbf{{{fmt_dataset(ds)}}} across all 6 models and 8 variants (Part {part_num} of 12).}}{lbl_cmd}",
                    "\\begin{tabular}{cllccccc@{\\hspace{4pt}}cllccccc}",
                    "\\toprule",
                    "$k$ & \\textbf{Model} & \\textbf{Ablation Variant} & $\\|\\mu\\|_2$ & \\textbf{Cos} & \\textbf{MEV} & \\textbf{NID} & \\textbf{Iso} & $k$ & \\textbf{Model} & \\textbf{Ablation Variant} & $\\|\\mu\\|_2$ & \\textbf{Cos} & \\textbf{MEV} & \\textbf{NID} & \\textbf{Iso} \\\\",
                    "\\midrule",
                ]
            )
            for r_i in range(len(rows_l)):
                if r_i > 0 and r_i % 8 == 0:
                    lines_56.append("\\midrule")
                rl = rows_l[r_i]
                rr = rows_r[r_i]
                kl = str(rl["k"]) if r_i % 24 == 0 else ""
                kr = str(rr["k"]) if r_i % 24 == 0 else ""
                ml = fmt_model_short(str(rl["model"])) if r_i % 8 == 0 else ""
                mr = fmt_model_short(str(rr["model"])) if r_i % 8 == 0 else ""
                ls = f"{kl} & {ml} & {fmt_method(str(rl['method']), is_comp_ablation=True)} & {float(rl['centroid_norm']):.3f} & {float(rl['average_cosine']):.3f} & {float(rl['mev_top1']):.3f} & {float(rl['nid_spectral_entropy']):.3f} & {float(rl['isoscore']):.3f}"
                rs = f"{kr} & {mr} & {fmt_method(str(rr['method']), is_comp_ablation=True)} & {float(rr['centroid_norm']):.3f} & {float(rr['average_cosine']):.3f} & {float(rr['mev_top1']):.3f} & {float(rr['nid_spectral_entropy']):.3f} & {float(rr['isoscore']):.3f}"
                lines_56.append(f"{ls} & {rs} \\\\")
            lines_56.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
            if part_num in (4, 8):
                lines_56.append("\\clearpage")
            part_num += 1
    (out_dir / "a56_geometry_ablation_compression.tex").write_text(
        "\n".join(lines_56) + "\n"
    )
    inputs.append("tables/appendix/ablation/a56_geometry_ablation_compression.tex")

    return inputs


def generate_main_tables() -> None:
    """Generate the 7 main-body tables in tables/main/ (zero negative vspace, zero negative controls)."""
    main_dir = PAPER_DIR / "tables" / "main"
    main_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = PAPER_DIR / "figures"
    for f in fig_dir.glob("*.tex"):
        f.unlink()

    # Table 1: Unified operator taxonomy
    t1 = """\\begin{table*}[t]
\\centering
\\scriptsize
\\setlength{\\tabcolsep}{4.8pt}
\\renewcommand{\\arraystretch}{0.96}
\\caption{Comparison of training-free post-processing and dimensional compression operators under the unified spectral transformation in Equation~\\eqref{eq:unified_operator}.}
\\label{tab:main_operators}
\\begin{tabular}{lllllll}
\\toprule
\\textbf{Method} & \\textbf{Centering $g(x)$} & \\textbf{Basis $U_k$} & \\textbf{Dim $k$} & \\textbf{Regularizer $\\epsilon(k)$} & \\textbf{Exponent $\\gamma(k)$} & \\textbf{Tuned Params} \\\\
\\midrule
Baseline & $x$ & $I_d$ & $d$ & $0$ & $0$ & None \\\\
Standardization & $\\operatorname{diag}(\\hat{\\sigma})^{-1}(x - \\mu)$ & $I_d$ & $d$ & $0$ & $0$ & None \\\\
R1 & $x - \\mu$ & $I_d$ & $d$ & $0$ & $0$ & None \\\\
R2 & $x - (x^\\top \\hat{\\mu})\\hat{\\mu}$ & $I_d$ & $d$ & $0$ & $0$ & None \\\\
ABTT-$D$ & $(I_d - U_D U_D^\\top)(x - \\mu)$ & $I_d$ & $d$ & $0$ & $0$ & $D \\in \\{1,2,3\\}$ \\\\
Prefix (MRL) & $x$ & $[I_k; 0]$ & $k \\le d$ & $0$ & $0$ & None \\\\
PCA-$k$ & $x - \\mu$ & $U_k(\\Sigma)$ & $k < d$ & $0$ & $0$ & None \\\\
Whitening-$k$ & $x - \\mu$ & $U_k(\\Sigma)$ & $k < d$ & $10^{-6}$ & $1$ & $\\epsilon = 10^{-6}$ \\\\
Soft-ZCA & $x - \\mu$ & $U_d(\\Sigma)$ & $d$ & $10^{-4}$ & $1$ & $\\epsilon = 10^{-4}$ \\\\
SpecTemp & $x - \\mu$ & $U_k(\\Sigma)$ & $k < d$ & $10^{-6}$ & $\\gamma_{\\text{SNR}}(k)$ & None \\\\
\\midrule
\\textbf{COAST (Ours)} & $x - (x^\\top \\hat{\\mu})\\hat{\\mu}$ & $U_{\\perp, k}(\\Sigma_\\perp)$ & $k \\le d$ & $\\lambda_{\\perp, k_{\\text{knee}}}^{1-\\gamma_{\\text{crowd}}}(\\tau(k)^2 E_\\perp \\bar{\\lambda}_\\perp)^{\\gamma_{\\text{crowd}}}$ & $\\max(\\gamma_{\\text{crowd}}, \\gamma_{\\text{SNR}}(k))$ & \\textbf{None} \\\\
\\bottomrule
\\end{tabular}
\\end{table*}
"""
    (main_dir / "table1_operators.tex").write_text(t1)

    # Table 2: Models and Datasets
    t2 = """\\begin{table*}[t]
\\centering
\\scriptsize
\\setlength{\\tabcolsep}{2.8pt}
\\renewcommand{\\arraystretch}{0.94}
\\caption{Evaluated model backbones (left) and MTEB v2 retrieval and semantic similarity benchmarks (right).}
\\label{tab:main_setup}
\\begin{tabular}{llccc@{\\hspace{8pt}}llccc}
\\toprule
\\multicolumn{5}{c}{\\textbf{(a) Model Backbones (Uniform 2,048-Token Evaluation Limit)}} & \\multicolumn{5}{c}{\\textbf{(b) MTEB v2 Evaluation Benchmarks}} \\\\
\\cmidrule(lr){1-5} \\cmidrule(lr){6-10}
\\textbf{Model Identifier} & \\textbf{Category} & \\textbf{Params} & $d$ & \\textbf{Pooling} & \\textbf{Dataset} & \\textbf{Track / Domain} & \\textbf{Corpus / Pairs} & \\textbf{Queries} & \\textbf{Avg Tok.} \\\\
\\midrule
\\texttt{embeddinggemma-300m} & Ded. (MRL $\\ge 128$) & 308M & 768 & Mean & FiQA-2018 & IR (Financial QA) & 57,638 & 648 & 174 \\\\
\\texttt{Qwen3-Embedding-0.6B} & Ded. (MRL $\\ge 32$) & 600M & 1024 & Last & ArguAna & IR (Argument) & 8,674 & 1,406 & 226 \\\\
\\texttt{gemma-3-1b-pt} & Base Causal LLM & 1.00B & 1152 & Mean, Last & SCIDOCS & IR (Scientific) & 25,657 & 1,000 & 215 \\\\
\\texttt{Qwen3.5-0.8B-Base} & Base Causal LLM & 800M & 1024 & Mean, Last & STSBenchmark & STS (General) & 1,379 pairs & -- & 12 \\\\
 & & & & & SICK-R & STS (Captions) & 9,927 pairs & -- & 10 \\\\
 & & & & & STS22.v2 & STS (Long News) & 197 pairs (394) & -- & 638 \\\\
\\bottomrule
\\end{tabular}
\\end{table*}
"""
    (main_dir / "table2_setup.tex").write_text(t2)

    # Table 3: Spectral energy partition and mean-leakage verification
    df_e = pl.read_csv(CSV_DIR / "parameters" / "energy_and_knee_parameters.csv")
    df_t = pl.read_csv(CSV_DIR / "parameters" / "adaptive_tempering_parameters.csv")
    sp = df_e.join(df_t, on=["dataset", "model", "native_d"]).with_columns(
        pl.when(
            pl.col("model").is_in(
                ["google/embeddinggemma-300m", "Qwen/Qwen3-Embedding-0.6B"]
            )
        )
        .then(pl.lit("Ded."))
        .when(pl.col("model").str.contains("mean_pooling"))
        .then(pl.lit("Base-M"))
        .otherwise(pl.lit("Base-L"))
        .alias("tier")
    )
    sp_agg = sp.group_by(["dataset", "tier"], maintain_order=True).agg(
        [
            pl.col("crowding_energy_mu_sq").mean().alias("e_par"),
            pl.col("dispersion_trace").mean().alias("e_perp"),
            pl.col("gamma_crowd").mean().alias("gc"),
            pl.col("centered_tail_leakage_norm").mean().alias("t_leak"),
            (pl.col("lambda_knee") * 1e3).mean().alias("lk"),
            (pl.col("epsilon_star_full") * 1e3).mean().alias("ed"),
            pl.col("coast_gamma_k64").mean().alias("g64"),
        ]
    )
    rows_sp = list(sp_agg.iter_rows(named=True))
    sp_l = rows_sp[:9]
    sp_r = rows_sp[9:]
    t3_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{1.8pt}",
        "\\renewcommand{\\arraystretch}{0.92}",
        "\\caption{Empirical unit-sphere energy partition ($E_\\parallel + \\frac{N-1}{N}(\\sigma_\\parallel^2 + E_\\perp) = 1$), centered covariance tail leakage $\\|U_{\\text{tail}}^\\top \\hat{\\mu}\\|_2$, and derived spectral parameters across all 6 benchmarks and 3 model categories ($\\lambda_{\\text{knee}}$ and $\\epsilon^*(d)$ in units of $10^{-3}$; all 36 configurations in Appendix Tables~\\ref{tab:app_a43_energy_parameters}--\\ref{tab:app_a44_tempering_parameters}).}",
        "\\label{tab:main_spectral_regimes}",
        "\\begin{tabular}{llccccccc@{\\hspace{8pt}}llccccccc}",
        "\\toprule",
        "\\multicolumn{9}{c}{\\textbf{(a) Information Retrieval Benchmarks}} & \\multicolumn{9}{c}{\\textbf{(b) Semantic Textual Similarity Benchmarks}} \\\\",
        "\\cmidrule(lr){1-9} \\cmidrule(lr){10-18}",
        "\\textbf{Dataset} & \\textbf{Cat.} & $E_\\parallel$ & $E_\\perp$ & $\\gamma_{\\text{crowd}}$ & $\\|U_{\\text{tail}}^\\top\\hat{\\mu}\\|_2$ & $\\lambda_{\\text{knee}}$ & $\\epsilon^*(d)$ & $\\gamma^*(64)$ & \\textbf{Dataset} & \\textbf{Cat.} & $E_\\parallel$ & $E_\\perp$ & $\\gamma_{\\text{crowd}}$ & $\\|U_{\\text{tail}}^\\top\\hat{\\mu}\\|_2$ & $\\lambda_{\\text{knee}}$ & $\\epsilon^*(d)$ & $\\gamma^*(64)$ \\\\",
        "\\midrule",
    ]
    for r_i in range(9):
        if r_i > 0 and r_i % 3 == 0:
            t3_lines.append("\\midrule")
        rl = sp_l[r_i]
        rr = sp_r[r_i]
        dsl = fmt_dataset(str(rl["dataset"])) if r_i % 3 == 0 else ""
        dsr = fmt_dataset(str(rr["dataset"])) if r_i % 3 == 0 else ""
        ls = f"{dsl} & {rl['tier']} & {float(rl['e_par']):.3f} & {float(rl['e_perp']):.3f} & {float(rl['gc']):.2f} & {float(rl['t_leak']):.3f} & {float(rl['lk']):.2f} & {float(rl['ed']):.3f} & {float(rl['g64']):.2f}"
        rs = f"{dsr} & {rr['tier']} & {float(rr['e_par']):.3f} & {float(rr['e_perp']):.3f} & {float(rr['gc']):.2f} & {float(rr['t_leak']):.3f} & {float(rr['lk']):.2f} & {float(rr['ed']):.3f} & {float(rr['g64']):.2f}"
        t3_lines.append(f"{ls} & {rs} \\\\")
    t3_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    (main_dir / "table3_spectral_regimes.tex").write_text("\n".join(t3_lines) + "\n")

    # Table 4: Full-dimension retrieval (nDCG@10) and similarity (Spearman) across all 6 datasets x 3 categories (9 methods)
    ds_full_specs = [
        ("retrieval/fiqa_ndcg.csv", 4),
        ("retrieval/arguana_ndcg.csv", 4),
        ("retrieval/scidocs_ndcg.csv", 4),
        ("similarity/stsb_spearman.csv", 2),
        ("similarity/sickr_spearman.csv", 2),
        ("similarity/sts22_spearman.csv", 2),
    ]
    dfs_t4 = [pl.read_csv(CSV_DIR / p) for p, _ in ds_full_specs]
    methods_t4 = dfs_t4[0]["method"].to_list()
    t4_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.2pt}",
        "\\renewcommand{\\arraystretch}{0.94}",
        "\\caption{Full-dimension ($k=d$) retrieval (nDCG@10) and semantic similarity (Spearman $\\rho \\times 100$) across all 6 benchmarks and 3 model categories: Dedicated Encoders (\\textbf{Ded.}), Base Mean Pooling (\\textbf{B-M}), and Base Last-Token Pooling (\\textbf{B-L}). Per-model breakdowns for all 5 extrinsic metrics appear in Appendix Tables~\\ref{tab:app_a01_fiqa_ndcg}--\\ref{tab:app_a15_sts22_pearson}.}",
        "\\label{tab:main_full_results}",
        "\\begin{tabular}{lccc@{\\hspace{5pt}}ccc@{\\hspace{5pt}}ccc@{\\hspace{6pt}}ccc@{\\hspace{5pt}}ccc@{\\hspace{5pt}}ccc}",
        "\\toprule",
        " & \\multicolumn{3}{c}{\\textbf{FiQA-2018 (nDCG)}} & \\multicolumn{3}{c}{\\textbf{ArguAna (nDCG)}} & \\multicolumn{3}{c}{\\textbf{SCIDOCS (nDCG)}} & \\multicolumn{3}{c}{\\textbf{STS-B ($\\rho$)}} & \\multicolumn{3}{c}{\\textbf{SICK-R ($\\rho$)}} & \\multicolumn{3}{c}{\\textbf{STS22.v2 ($\\rho$)}} \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-13} \\cmidrule(lr){14-16} \\cmidrule(lr){17-19}",
        "\\textbf{Method} & \\textbf{Ded.} & \\textbf{B-M} & \\textbf{B-L} & \\textbf{Ded.} & \\textbf{B-M} & \\textbf{B-L} & \\textbf{Ded.} & \\textbf{B-M} & \\textbf{B-L} & \\textbf{Ded.} & \\textbf{B-M} & \\textbf{B-L} & \\textbf{Ded.} & \\textbf{B-M} & \\textbf{B-L} & \\textbf{Ded.} & \\textbf{B-M} & \\textbf{B-L} \\\\",
        "\\midrule",
    ]
    max_t4: list[dict[str, float]] = []
    for df_i in dfs_t4:
        max_t4.append(
            {
                "mean_dedicated": _f(df_i["mean_dedicated"].max()),
                "mean_base_mean": _f(df_i["mean_base_mean"].max()),
                "mean_base_last": _f(df_i["mean_base_last"].max()),
            }
        )
    for m in methods_t4:
        cells = [fmt_method(m)]
        for d_idx, (_, dec) in enumerate(ds_full_specs):
            row = dfs_t4[d_idx].filter(pl.col("method") == m).row(0, named=True)
            for tc in ["mean_dedicated", "mean_base_mean", "mean_base_last"]:
                val = float(row[tc])
                s = f"{val:.{dec}f}"
                if abs(val - max_t4[d_idx][tc]) < 10 ** (-dec):
                    s = f"\\textbf{{{s}}}"
                cells.append(s)
        t4_lines.append(" & ".join(cells) + " \\\\")
    t4_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    (main_dir / "table4_full_results.tex").write_text("\n".join(t4_lines) + "\n")

    # Table 5: Complete compression retention (%) across all k in {512, 256, 128, 64} (4 x 5 = 20 rows!)
    ds_comp_specs = [
        "compression/fiqa_compression_ndcg.csv",
        "compression/arguana_compression_ndcg.csv",
        "compression/scidocs_compression_ndcg.csv",
        "compression/stsb_compression_spearman.csv",
        "compression/sickr_compression_spearman.csv",
        "compression/sts22_compression_spearman.csv",
    ]
    dfs_t5 = [pl.read_csv(CSV_DIR / p) for p in ds_comp_specs]
    t5_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.1pt}",
        "\\renewcommand{\\arraystretch}{0.92}",
        "\\caption{Dimensional compression retention (\\% of native $k=d$ baseline) across all target dimensions $k \\in \\{512, 256, 128, 64\\}$, 6 benchmarks, and 3 model categories. Per-model scores and retention percentages for all 5 metrics appear in Appendix Tables~\\ref{tab:app_a16_fiqa_compression_ndcg}--\\ref{tab:app_a30_sts22_compression_pearson}.}",
        "\\label{tab:main_compression_results}",
        "\\begin{tabular}{clccc@{\\hspace{4pt}}ccc@{\\hspace{4pt}}ccc@{\\hspace{5pt}}ccc@{\\hspace{4pt}}ccc@{\\hspace{4pt}}ccc}",
        "\\toprule",
        " & & \\multicolumn{3}{c}{\\textbf{FiQA-2018 (\\%)}} & \\multicolumn{3}{c}{\\textbf{ArguAna (\\%)}} & \\multicolumn{3}{c}{\\textbf{SCIDOCS (\\%)}} & \\multicolumn{3}{c}{\\textbf{STS-B (\\%)}} & \\multicolumn{3}{c}{\\textbf{SICK-R (\\%)}} & \\multicolumn{3}{c}{\\textbf{STS22.v2 (\\%)}} \\\\",
        "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11} \\cmidrule(lr){12-14} \\cmidrule(lr){15-17} \\cmidrule(lr){18-20}",
        "$k$ & \\textbf{Method} & \\textbf{Ded.} & \\textbf{B-M} & \\textbf{B-L} & \\textbf{Ded.} & \\textbf{B-M} & \\textbf{B-L} & \\textbf{Ded.} & \\textbf{B-M} & \\textbf{B-L} & \\textbf{Ded.} & \\textbf{B-M} & \\textbf{B-L} & \\textbf{Ded.} & \\textbf{B-M} & \\textbf{B-L} & \\textbf{Ded.} & \\textbf{B-M} & \\textbf{B-L} \\\\",
        "\\midrule",
    ]
    for k_i, k_val in enumerate([512, 256, 128, 64]):
        if k_i > 0:
            t5_lines.append("\\midrule")
        sub_dfs = [df_i.filter(pl.col("k") == k_val) for df_i in dfs_t5]
        max_t5 = [
            {
                tc: _f(sdf[tc].max())
                for tc in [
                    "mean_dedicated_retention_pct",
                    "mean_base_mean_retention_pct",
                    "mean_base_last_retention_pct",
                ]
            }
            for sdf in sub_dfs
        ]
        methods_k = sub_dfs[0]["method"].to_list()
        for m_i, m in enumerate(methods_k):
            k_cell = f"\\multirow{{5}}{{*}}{{{k_val}}}" if m_i == 0 else ""
            cells = [k_cell, fmt_method(m)]
            for d_idx, sdf in enumerate(sub_dfs):
                row = sdf.filter(pl.col("method") == m).row(0, named=True)
                for tc in [
                    "mean_dedicated_retention_pct",
                    "mean_base_mean_retention_pct",
                    "mean_base_last_retention_pct",
                ]:
                    val = float(row[tc])
                    s = f"{val:.1f}"
                    if abs(val - max_t5[d_idx][tc]) < 0.05:
                        s = f"\\textbf{{{s}}}"
                    cells.append(s)
            t5_lines.append(" & ".join(cells) + " \\\\")
    t5_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    (main_dir / "table5_compression_results.tex").write_text("\n".join(t5_lines) + "\n")

    # Side-by-side Tables 6 and 7 inside a single two-column table*
    t6 = """\\begin{table*}[t]
\\centering
\\scriptsize
\\renewcommand{\\arraystretch}{0.92}
\\begin{minipage}[t]{0.49\\textwidth}
\\centering
\\setlength{\\tabcolsep}{1.1pt}
\\caption{Component ablation study of COAST at $k=d$, $k=512$, and $k=64$ (complete ablation tables across all benchmarks and metrics in Appendix Tables~\\ref{tab:app_a45_retrieval_ablation_ndcg}--\\ref{tab:app_a56_geometry_ablation_compression}).}
\\label{tab:main_ablation_summary}
\\begin{tabular}{llcccc}
\\toprule
& & \\multicolumn{2}{c}{\\textbf{FiQA-2018}} & \\multicolumn{2}{c}{\\textbf{STS-B / STS22}} \\\\
\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}
\\textbf{Dim} & \\textbf{Ablation Variant} & \\textbf{Ded.} & \\textbf{B-M} & \\textbf{Ded.} & \\textbf{B-M} \\\\
\\midrule
\\multicolumn{6}{l}{\\textit{(a) Full Dimension ($k=d$; FiQA nDCG and STS-B $\\rho$)}} \\\\
$k{=}d$ & Unreg. ZCA ($\\Sigma, 1, 10^{-6}$) & 0.3923 & 0.0737 & 78.72 & 60.63 \\\\
$k{=}d$ & Soft-ZCA ($\\Sigma, 1, 10^{-4}$) & 0.4103 & 0.0791 & 80.51 & 60.54 \\\\
$k{=}d$ & Fixed $\\gamma{=}0.5$ ($\\Sigma_\\perp, 0.5, 10^{-4}$) & 0.4541 & 0.0752 & 84.12 & 56.55 \\\\
$k{=}d$ & w/o Adaptive $\\epsilon^*$ ($\\Sigma_\\perp, 10^{-4}$) & 0.4744 & 0.0794 & \\textbf{85.23} & 60.54 \\\\
$k{=}d$ & w/o Orth. Proj. ($\\Sigma, \\gamma_{\\text{crowd}}, \\epsilon^*$) & 0.4731 & 0.0814 & 85.14 & 60.76 \\\\
$k{=}d$ & \\textbf{COAST} ($\\Sigma_\\perp, \\gamma_{\\text{crowd}}, \\epsilon^*$) & \\textbf{0.4748} & \\textbf{0.0821} & \\textbf{85.23} & \\textbf{60.77} \\\\
\\midrule
\\multicolumn{6}{l}{\\textit{(b) Broad Compression ($k=512$; FiQA and STS22 Ret.\\%)}} \\\\
$512$ & Whitening-$k$ ($\\Sigma, 1, 10^{-6}$) & 86.9\\% & 141.6\\% & 52.1\\% & 89.7\\% \\\\
$512$ & SpecTemp ($\\Sigma, \\gamma_{\\text{SNR}}, 10^{-6}$) & 99.8\\% & 93.1\\% & 99.6\\% & 109.1\\% \\\\
$512$ & w/o Adaptive $\\epsilon^*$ ($\\gamma^*, 10^{-6}$) & \\textbf{100.0\\%} & 145.4\\% & \\textbf{100.0\\%} & 89.6\\% \\\\
$512$ & w/o Rank Scale $\\tau^2$ ($\\epsilon^*(d)$) & 99.8\\% & 144.9\\% & 99.9\\% & \\textbf{132.1\\%} \\\\
$512$ & \\textbf{COAST} ($\\Sigma_\\perp, \\gamma^*, \\epsilon^*(k)$) & 99.8\\% & \\textbf{146.7\\%} & 99.9\\% & \\textbf{132.1\\%} \\\\
\\midrule
\\multicolumn{6}{l}{\\textit{(c) Compact Compression ($k=64$; FiQA and STS22 Ret.\\%)}} \\\\
$64$ & SpecTemp ($\\Sigma, \\gamma_{\\text{SNR}}, 10^{-6}$) & 75.7\\% & 58.3\\% & 94.7\\% & 111.6\\% \\\\
$64$ & w/o SNR Sched. ($\\gamma_{\\text{crowd}}, \\epsilon^*$) & 77.0\\% & \\textbf{64.4\\%} & 90.9\\% & 116.6\\% \\\\
$64$ & w/o Orth. Proj. ($\\Sigma, \\gamma^*, \\epsilon^*$) & 76.0\\% & 61.4\\% & \\textbf{93.8\\%} & \\textbf{116.9\\%} \\\\
$64$ & \\textbf{COAST} ($\\Sigma_\\perp, \\gamma^*, \\epsilon^*(k)$) & \\textbf{78.2\\%} & \\textbf{64.4\\%} & 93.6\\% & 116.6\\% \\\\
\\bottomrule
\\end{tabular}
\\end{minipage}\\hfill
\\begin{minipage}[t]{0.48\\textwidth}
\\centering
\\setlength{\\tabcolsep}{2.4pt}
\\caption{Intrinsic geometry diagnostics at full dimensionality ($k=d$) on FiQA-2018 and STSBenchmark across Dedicated Encoders and Base Mean Pooling models (complete geometry tables in Appendix Tables~\\ref{tab:app_a31_fiqa_geom_full}--\\ref{tab:app_a42_sts22_geom_comp}).}
\\label{tab:main_geometry_summary}
\\begin{tabular}{llccccc}
\\toprule
\\textbf{Category} & \\textbf{Method} & $\\|\\mu\\|_2 \\downarrow$ & \\textbf{AvgCos} & \\textbf{MEV} $\\downarrow$ & \\textbf{NID} $\\uparrow$ & \\textbf{Iso} $\\uparrow$ \\\\
\\midrule
\\multicolumn{7}{l}{\\textit{(a) FiQA-2018 Corpus Geometry ($N = 57,638$)}} \\\\
Dedicated & Baseline & 0.484 & 0.237 & 0.042 & 0.838 & 0.143 \\\\
Dedicated & R2 & 0.004 & 0.000 & 0.042 & 0.837 & 0.141 \\\\
Dedicated & Soft-ZCA & 0.005 & 0.000 & 0.002 & 0.983 & 0.846 \\\\
Dedicated & \\textbf{COAST} & 0.005 & 0.000 & 0.037 & 0.848 & 0.164 \\\\
\\cmidrule(lr){1-7}
Base (Mean) & Baseline & 0.899 & 0.811 & 0.114 & 0.677 & 0.029 \\\\
Base (Mean) & R2 & 0.035 & 0.002 & 0.112 & 0.686 & 0.032 \\\\
Base (Mean) & ABTT ($D{=}1$) & 0.032 & 0.001 & 0.088 & 0.705 & 0.039 \\\\
Base (Mean) & Soft-ZCA & 0.011 & 0.000 & 0.005 & 0.942 & 0.512 \\\\
Base (Mean) & \\textbf{COAST} & 0.009 & 0.000 & 0.003 & 0.969 & 0.697 \\\\
\\midrule
\\multicolumn{7}{l}{\\textit{(b) STSBenchmark Pooled Geometry ($N = 2,758$)}} \\\\
Dedicated & Baseline & 0.497 & 0.246 & 0.067 & 0.795 & 0.089 \\\\
Dedicated & Soft-ZCA & 0.010 & 0.000 & 0.008 & 0.968 & 0.699 \\\\
Dedicated & \\textbf{COAST} & 0.008 & 0.000 & 0.056 & 0.809 & 0.107 \\\\
\\cmidrule(lr){1-7}
Base (Mean) & Baseline & 0.844 & 0.712 & 0.158 & 0.669 & 0.021 \\\\
Base (Mean) & R2 & 0.036 & 0.001 & 0.166 & 0.667 & 0.020 \\\\
Base (Mean) & Soft-ZCA & 0.014 & 0.000 & 0.010 & 0.943 & 0.511 \\\\
Base (Mean) & \\textbf{COAST} & 0.013 & 0.000 & 0.009 & 0.950 & 0.551 \\\\
\\bottomrule
\\end{tabular}
\\end{minipage}
\\end{table*}
"""
    (main_dir / "table6_ablation_summary.tex").write_text(t6)
    (main_dir / "table7_geometry_summary.tex").write_text("")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    generate_main_tables()
    s12 = generate_appendix_retrieval_and_similarity()
    s3 = generate_appendix_compression()
    s4 = generate_appendix_geometry()
    s5 = generate_appendix_parameters()
    s6 = generate_appendix_ablations()
    total = len(s12) + len(s3) + len(s4) + len(s5) + len(s6)
    logger.info(
        "Generated 7 main tables and %d modular Appendix LaTeX table files in %s",
        total,
        PAPER_DIR,
    )


if __name__ == "__main__":
    main()
