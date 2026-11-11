from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from kneed import KneeLocator

from src.registry import (
    BASE_MODELS,
    BASE_POOLING_MODES,
    EMBEDDING_MODELS,
    get_raw_cache_dir,
    is_base_model,
)
from src.tables.common import save_table_csv
from src.tables.loader import format_canonical_model_name
from src.tables.schema import (
    CANONICAL_MODEL_COLUMNS,
    IR_DATASETS,
    LADDER_K_ORDERED,
    STS_DATASETS,
)
from src.transforms.common import compute_mean_and_cov
from src.transforms.compression.coast import (
    compute_orthogonal_spectrum_and_coast_params,
)
from src.transforms.compression.spectemp import (
    calculate_snr_curve,
    find_optimal_gamma,
)


def build_suite_5_parameters(tables_dir: Path) -> dict[str, pl.DataFrame]:
    """Build two paper-width 36-row spectral parameter tables across all datasets and models."""
    param_dir = tables_dir / "parameters"
    param_dir.mkdir(parents=True, exist_ok=True)
    old_file = param_dir / "spectral_parameters.csv"
    if old_file.exists():
        old_file.unlink()

    energy_rows: list[dict[str, Any]] = []
    tempering_rows: list[dict[str, Any]] = []
    all_specs = [(d, "retrieval") for d in IR_DATASETS] + [
        (d, "similarity") for d in STS_DATASETS
    ]

    model_configs: list[tuple[str, str | None]] = []
    for m in EMBEDDING_MODELS:
        model_configs.append((m, None))
    for m in BASE_MODELS:
        for p in BASE_POOLING_MODES:
            model_configs.append((m, p))

    def config_sort_key(cfg: tuple[str, str | None]) -> int:
        m_id, pool = cfg
        tier = "base" if is_base_model(m_id) else "embedding"
        c_name = format_canonical_model_name(m_id.replace("/", "_"), tier, pool)
        return (
            CANONICAL_MODEL_COLUMNS.index(c_name)
            if c_name in CANONICAL_MODEL_COLUMNS
            else 999
        )

    model_configs.sort(key=config_sort_key)

    for dataset, track in all_specs:
        for model_id, pooling in model_configs:
            raw_dir = get_raw_cache_dir(dataset, model_id, pooling=pooling)
            if track == "retrieval":
                calib_data = torch.load(raw_dir / "corpus.pt", weights_only=True)
            else:
                s1 = torch.load(raw_dir / "sentences1.pt", weights_only=True)
                s2 = torch.load(raw_dir / "sentences2.pt", weights_only=True)
                calib_data = torch.cat([s1, s2], dim=0)

            native_d = int(calib_data.shape[1])
            tier = "base" if is_base_model(model_id) else "embedding"
            canonical_model = format_canonical_model_name(
                model_id.replace("/", "_"), tier, pooling
            )

            # 1. Raw covariance spectrum (used by SpecTemp)
            mu_raw, eigvals_raw, eigvecs_raw = compute_mean_and_cov(calib_data)
            snr_raw, _ = calculate_snr_curve(eigvals_raw, tail_percentile=0.9)
            snr_raw_np = snr_raw.cpu().numpy()
            kn_raw = KneeLocator(
                np.arange(1, len(snr_raw_np) + 1),
                snr_raw_np,
                curve="convex",
                direction="decreasing",
                S=0.5,
            )
            knee_raw = int(kn_raw.knee) if kn_raw.knee is not None else 1

            spectemp_gammas: dict[str, float] = {}
            for k in LADDER_K_ORDERED:
                spectemp_gammas[f"spectemp_gamma_k{k}"] = find_optimal_gamma(
                    eigvals_raw, target_dim=k, kneedle_s=0.5
                )

            # 2. Orthogonal-complement spectrum and COAST parameters
            u_mu, eigvals_perp, _, eps_star_full, gamma_full, knee_perp = (
                compute_orthogonal_spectrum_and_coast_params(
                    calib_data, target_dim=native_d
                )
            )
            e_par = float((torch.linalg.norm(mu_raw) ** 2).item())
            sigma_par_sq = (
                float(torch.clamp(torch.var(calib_data @ u_mu), min=0.0).item())
                if torch.linalg.norm(u_mu) >= 1e-12
                else 0.0
            )

            second_moment = torch.matmul(calib_data.T, calib_data) / max(
                1, calib_data.shape[0]
            )
            _, evecs_m = torch.linalg.eigh(second_moment.cpu())
            pc1_uncentered = evecs_m[:, -1].to(u_mu.device)
            cos_mu_pc1_uncentered = float(
                torch.abs(torch.dot(u_mu, pc1_uncentered)).item()
            )
            cos_mu_pc1_centered = float(
                torch.abs(torch.dot(u_mu, eigvecs_raw[:, 0])).item()
            )
            tail_projs = torch.matmul(eigvecs_raw[:, knee_raw:].T, u_mu)
            centered_tail_leakage = float(torch.linalg.norm(tail_projs).item())

            e_perp = float(eigvals_perp.sum().item())
            d_perp = len(eigvals_perp)
            lam_bar_perp = e_perp / max(1, d_perp)
            rho_tr = e_par / max(e_perp, 1e-12)
            gamma_crowd = min(1.0, rho_tr**2)

            _, noise_perp = calculate_snr_curve(eigvals_perp, tail_percentile=0.9)
            lam_perp_1 = float(eigvals_perp[0].item())
            lam_perp_knee = float(eigvals_perp[knee_perp - 1].item())

            coast_params: dict[str, float] = {
                "epsilon_star_full": eps_star_full,
                "coast_gamma_full": gamma_full,
            }
            for k in LADDER_K_ORDERED:
                _, _, _, eps_k, gk, _ = compute_orthogonal_spectrum_and_coast_params(
                    calib_data, target_dim=k
                )
                coast_params[f"epsilon_star_k{k}"] = eps_k
                coast_params[f"coast_gamma_k{k}"] = gk

            energy_rows.append(
                {
                    "dataset": dataset,
                    "model": canonical_model,
                    "native_d": native_d,
                    "crowding_energy_mu_sq": e_par,
                    "axial_variance_sigma_par_sq": sigma_par_sq,
                    "dispersion_trace": e_perp,
                    "mean_eigenvalue": lam_bar_perp,
                    "crowding_to_dispersion_ratio": rho_tr,
                    "gamma_crowd": gamma_crowd,
                    "cos_mu_pc1_uncentered": cos_mu_pc1_uncentered,
                    "cos_mu_pc1_centered": cos_mu_pc1_centered,
                    "centered_tail_leakage_norm": centered_tail_leakage,
                    "lambda_perp_1": lam_perp_1,
                    "noise_floor_variance": float(noise_perp.item()),
                    "kneedle_knee_k": knee_perp,
                    "lambda_knee": lam_perp_knee,
                }
            )
            tempering_rows.append(
                {
                    "dataset": dataset,
                    "model": canonical_model,
                    "native_d": native_d,
                    "spectemp_knee_k": knee_raw,
                    **spectemp_gammas,
                    **coast_params,
                }
            )

    energy_df = pl.DataFrame(energy_rows)
    tempering_df = pl.DataFrame(tempering_rows)
    save_table_csv(energy_df, param_dir / "energy_and_knee_parameters.csv", precision=6)
    save_table_csv(
        tempering_df, param_dir / "adaptive_tempering_parameters.csv", precision=6
    )
    return {
        "energy_and_knee_parameters": energy_df,
        "adaptive_tempering_parameters": tempering_df,
    }
