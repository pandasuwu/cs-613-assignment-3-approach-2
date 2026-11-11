"""Programmatic verification of 100% orthogonal table coverage and all paper claims."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.tables.loader import load_master_dataframe

logger = logging.getLogger("verify_paper")

TABLES_DIR = Path("tables")
DED_COLS = ["google/embeddinggemma-300m", "Qwen/Qwen3-Embedding-0.6B"]
BASE_COLS = [
    "google/gemma-3-1b-pt (mean_pooling)",
    "google/gemma-3-1b-pt (last_token_pooling)",
    "Qwen/Qwen3.5-0.8B-Base (mean_pooling)",
    "Qwen/Qwen3.5-0.8B-Base (last_token_pooling)",
]


def _f(val: object) -> float:
    return float(str(val))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    df = load_master_dataframe()

    # 1. Verify 100% orthogonal cell partition across tables/
    s1 = sum(len(pl.read_csv(f)) * 6 for f in (TABLES_DIR / "retrieval").glob("*.csv"))
    s2 = sum(len(pl.read_csv(f)) * 6 for f in (TABLES_DIR / "similarity").glob("*.csv"))
    s3 = sum(
        len(pl.read_csv(f)) * 6 for f in (TABLES_DIR / "compression").glob("*.csv")
    )
    s4 = sum(len(pl.read_csv(f)) * 5 for f in (TABLES_DIR / "geometry").glob("*.csv"))
    s6_task = sum(
        len(pl.read_csv(f)) * 6
        for f in (TABLES_DIR / "ablation").glob("*_ablation_*.csv")
        if "geometry" not in f.name
    )
    s6_geom = sum(
        len(pl.read_csv(f)) * 5
        for f in (TABLES_DIR / "ablation").glob("geometry_ablation_*.csv")
    )
    total_exported = s1 + s2 + s3 + s4 + s6_task + s6_geom
    assert len(df) == total_exported, (
        f"Coverage mismatch: {len(df)} != {total_exported}"
    )
    logger.info(
        "Orthogonal coverage verified: %d atomic rows == %d exported table cells "
        "(retrieval=%d, similarity=%d, compression=%d, geometry=%d, ablation_task=%d, ablation_geom=%d)",
        len(df),
        total_exported,
        s1,
        s2,
        s3,
        s4,
        s6_task,
        s6_geom,
    )

    # 2. Verify spectral energy partition and mean-leakage claims
    ep = pl.read_csv(TABLES_DIR / "parameters" / "energy_and_knee_parameters.csv")
    tp = pl.read_csv(TABLES_DIR / "parameters" / "adaptive_tempering_parameters.csv")
    sp = ep.join(tp, on=["dataset", "model", "native_d"])
    ded = sp.filter(pl.col("model").is_in(DED_COLS))
    base = sp.filter(~pl.col("model").is_in(DED_COLS))
    logger.info(
        "Dedicated E_par/E_perp=[%.2f, %.2f], gamma_crowd=[%.4f, %.4f], lambda_knee*1e3=[%.2f, %.2f]",
        _f(ded["crowding_to_dispersion_ratio"].min()),
        _f(ded["crowding_to_dispersion_ratio"].max()),
        _f(ded["gamma_crowd"].min()),
        _f(ded["gamma_crowd"].max()),
        _f(ded["lambda_knee"].min()) * 1e3,
        _f(ded["lambda_knee"].max()) * 1e3,
    )
    logger.info(
        "Base E_par/E_perp=[%.2f, %.2f] (%d/%d > 1.0), Tail leakage=[%.4f, %.4f], Axial var max=%.4f",
        _f(base["crowding_to_dispersion_ratio"].min()),
        _f(base["crowding_to_dispersion_ratio"].max()),
        len(base.filter(pl.col("crowding_to_dispersion_ratio") > 1.0)),
        len(base),
        _f(sp["centered_tail_leakage_norm"].min()),
        _f(sp["centered_tail_leakage_norm"].max()),
        _f(sp["axial_variance_sigma_par_sq"].max()),
    )

    # 3. Verify full-dimension win counts
    full_files = sorted(
        list((TABLES_DIR / "retrieval").glob("*.csv"))
        + list((TABLES_DIR / "similarity").glob("*.csv"))
    )
    prim_c_zca_ded = prim_c_zca_base = prim_zca_loss_ded = prim_c_r2_base = 0
    all_c_zca_ded = all_zca_loss_ded = 0
    r2_r1_ir = r2_r1_sts = 0
    for f in full_files:
        t = pl.read_csv(f)
        rc = t.filter(pl.col("method") == "coast").row(0, named=True)
        rz = t.filter(pl.col("method") == "soft_zca").row(0, named=True)
        rb = t.filter(pl.col("method") == "baseline").row(0, named=True)
        r1 = t.filter(pl.col("method") == "r1").row(0, named=True)
        r2 = t.filter(pl.col("method") == "r2").row(0, named=True)
        for c in DED_COLS:
            all_c_zca_ded += int(float(rc[c]) >= float(rz[c]))
            all_zca_loss_ded += int(float(rz[c]) < float(rb[c]))
        if f.stem.endswith("_ndcg") or f.stem.endswith("_spearman"):
            for c in DED_COLS:
                prim_c_zca_ded += int(float(rc[c]) >= float(rz[c]))
                prim_zca_loss_ded += int(float(rz[c]) < float(rb[c]))
            for c in BASE_COLS:
                prim_c_zca_base += int(float(rc[c]) >= float(rz[c]))
                prim_c_r2_base += int(float(rc[c]) > float(r2[c]))
            for c in DED_COLS + BASE_COLS:
                if f.stem.endswith("_ndcg"):
                    r2_r1_ir += int(float(r2[c]) >= float(r1[c]))
                else:
                    r2_r1_sts += int(float(r2[c]) >= float(r1[c]))

    logger.info(
        "Full-dim primary (6 datasets): Soft-ZCA<Baseline Ded=%d/12 (all metrics %d/30); "
        "COAST>=Soft-ZCA Ded=%d/12 (all metrics %d/30), Base=%d/24; COAST>R2 Base=%d/24; "
        "R2>=R1 Total=%d/36 (IR=%d/18, STS=%d/18)",
        prim_zca_loss_ded,
        all_zca_loss_ded,
        prim_c_zca_ded,
        all_c_zca_ded,
        prim_c_zca_base,
        prim_c_r2_base,
        r2_r1_ir + r2_r1_sts,
        r2_r1_ir,
        r2_r1_sts,
    )

    # 4. Verify compression win counts
    comp_primary = [
        TABLES_DIR / "compression" / "fiqa_compression_ndcg.csv",
        TABLES_DIR / "compression" / "arguana_compression_ndcg.csv",
        TABLES_DIR / "compression" / "scidocs_compression_ndcg.csv",
        TABLES_DIR / "compression" / "stsb_compression_spearman.csv",
        TABLES_DIR / "compression" / "sickr_compression_spearman.csv",
        TABLES_DIR / "compression" / "sts22_compression_spearman.csv",
    ]
    c_whi_base = c_st_base = c_whi_ded = st_loss_512 = 0
    for fp in comp_primary:
        t = pl.read_csv(fp)
        for k_val in [512, 256, 128, 64]:
            sub = t.filter(pl.col("k") == k_val)
            rc = sub.filter(pl.col("method") == "coast").row(0, named=True)
            rw = sub.filter(pl.col("method") == "whitening").row(0, named=True)
            rs = sub.filter(pl.col("method") == "spectemp").row(0, named=True)
            for c in DED_COLS:
                c_whi_ded += int(float(rc[c]) >= float(rw[c]))
            for c in BASE_COLS:
                c_whi_base += int(float(rc[c]) >= float(rw[c]))
                c_st_base += int(float(rc[c]) >= float(rs[c]))
                if k_val == 512:
                    st_loss_512 += int(float(rs[c]) < float(rw[c]))

    logger.info(
        "Compression primary (6 datasets x 4 k): COAST>=Whitening Base=%d/96, Ded=%d/48; "
        "COAST>=SpecTemp Base=%d/96; SpecTemp<Whitening at k=512 Base=%d/24",
        c_whi_base,
        c_whi_ded,
        c_st_base,
        st_loss_512,
    )


if __name__ == "__main__":
    main()
