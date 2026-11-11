from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from kneed import KneeLocator

from src.transforms.common import compute_mean_and_cov, l2_normalize
from src.transforms.compression.spectemp import calculate_snr_curve

_SPECTRUM_CACHE: dict[
    tuple[float, float, float, tuple[int, ...], str, bool], dict[str, Any]
] = {}


def _get_cached_spectrum(calib_data: torch.Tensor, orth: bool) -> dict[str, Any]:
    """Compute and cache raw or orthogonal-complement eigenspectrum and Kneedle knee."""
    key = (
        float(calib_data[0, 0].item()),
        float(calib_data[-1, -1].item()),
        float(calib_data.mean().item()),
        tuple(calib_data.shape),
        str(calib_data.device),
        orth,
    )
    if key in _SPECTRUM_CACHE:
        return _SPECTRUM_CACHE[key]

    # Limit cache size to prevent memory growth across combinations
    if len(_SPECTRUM_CACHE) > 8:
        _SPECTRUM_CACHE.clear()

    mu = calib_data.mean(dim=0)
    mu_norm = torch.linalg.norm(mu)
    e_par = float(torch.clamp(mu_norm**2, min=0.0, max=1.0).item())

    if orth:
        if mu_norm >= 1e-12:
            u_mu = F.normalize(mu, p=2, dim=0)
            calib_def = calib_data - (calib_data @ u_mu).unsqueeze(1) * u_mu
        else:
            u_mu = torch.zeros_like(mu)
            calib_def = calib_data
        _, eigvals_all, eigvecs_all = compute_mean_and_cov(calib_def)
        eigenvalues = eigvals_all[:-1]
        eigenvectors = eigvecs_all[:, :-1]
    else:
        u_mu = mu
        _, eigenvalues, eigenvectors = compute_mean_and_cov(calib_data)

    snr_curve, noise_variance = calculate_snr_curve(eigenvalues, tail_percentile=0.9)
    snr_np = snr_curve.cpu().numpy()
    n_points = len(snr_np)

    kneedle = KneeLocator(
        np.arange(1, n_points + 1),
        snr_np,
        curve="convex",
        direction="decreasing",
        S=0.5,
    )
    knee_point = kneedle.knee
    knee_idx = int(knee_point) if (knee_point is not None and knee_point >= 1) else 1

    entry = {
        "mu": mu,
        "u_mu": u_mu,
        "e_par": e_par,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "snr_np": snr_np,
        "noise_variance": noise_variance,
        "knee_idx": knee_idx,
    }
    _SPECTRUM_CACHE[key] = entry
    return entry


def compute_orthogonal_spectrum_and_coast_params(
    calib_data: torch.Tensor,
    target_dim: int,
    kneedle_s: float = 0.5,
    tail_percentile: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, int]:
    """Compute R2-deflated orthogonal spectrum (Lambda_perp, U_perp) and parameter-free COAST parameters.

    Mathematical Formulation:
        1. Mean-subspace projection deflation on S^{d-1}:
           u_mu = mu / ||mu||_2,   x_perp = (I - u_mu u_mu^T) x
        2. Orthogonal complement covariance eigendecomposition in H_perp = span(u_mu)^perp:
           Sigma_perp = cov(X_perp) = U_perp Lambda_perp U_perp^T   (rank r = min(N - 1, d - 1))
        3. Unit-sphere energy partition and crowding-to-dispersion regime parameter:
           E_par = ||mu||_2^2,   E_perp = Tr(Sigma_perp),   lambda_bar_perp = E_perp / (d - 1)
           gamma_crowd = min(1.0, (E_par / E_perp)^2)
        4. Subspace-scaled crowding-regularized floor and adaptive tempering exponent at dimension k:
           tau(k) = min(1.0, k / min(N - 1, d - 1))
           epsilon*(k) = (lambda_{perp, k_knee})^(1 - gamma_crowd) * (tau(k)^2 * E_perp * lambda_bar_perp)^(gamma_crowd)
           gamma*(k) = max(gamma_crowd, min(1.0, SNR(k) / SNR(k_knee)))
    """
    _ = (kneedle_s, tail_percentile)
    spec = _get_cached_spectrum(calib_data, orth=True)
    u_mu = spec["u_mu"]
    eigenvalues = spec["eigenvalues"]
    eigenvectors = spec["eigenvectors"]
    snr_np = spec["snr_np"]
    noise_variance = spec["noise_variance"]
    knee_idx = spec["knee_idx"]
    e_par = spec["e_par"]

    n_samples = calib_data.shape[0]
    d_perp = len(eigenvalues)
    r_eff = max(1, min(n_samples - 1, d_perp))
    k = min(target_dim, d_perp)

    e_perp = float(torch.clamp(eigenvalues.sum(), min=1e-12).item())
    lam_bar_perp = e_perp / max(1, d_perp)
    rho_tr = e_par / e_perp
    gamma_crowd = min(1.0, rho_tr**2)

    lam_knee = max(float(eigenvalues[knee_idx - 1].item()), 1e-12)
    tau_k = min(1.0, k / r_eff)
    disp_subspace_scale = max((tau_k**2) * e_perp * lam_bar_perp, 1e-12)
    epsilon_star = max(
        (lam_knee ** (1.0 - gamma_crowd)) * (disp_subspace_scale**gamma_crowd),
        1e-12,
    )

    n_points = len(snr_np)
    snr_knee = float(snr_np[knee_idx - 1])
    if snr_knee <= 1e-12:
        gamma_snr = 1.0
    else:
        if k <= n_points:
            snr_k = float(snr_np[k - 1])
        else:
            snr_k = float(
                torch.clamp(
                    (eigenvalues[k - 1] - noise_variance)
                    / torch.clamp(noise_variance, min=1e-12),
                    min=0.0,
                ).item()
            )
        gamma_snr = max(0.0, min(1.0, snr_k / snr_knee))

    gamma_star = max(gamma_crowd, gamma_snr)
    return u_mu, eigenvalues, eigenvectors, epsilon_star, gamma_star, knee_idx


def transform_coast(
    x1: torch.Tensor,
    x2: torch.Tensor,
    calib_data: torch.Tensor,
    target_dim: int = 512,
    kneedle_s: float = 0.5,
    tail_percentile: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Crowding-Orthogonal Adaptive Spectral Tempering (COAST) for dimensional compression (d -> k)."""
    _ = (kneedle_s, tail_percentile)
    return transform_coast_ablation(
        x1, x2, calib_data, target_dim=target_dim, orth=True, crowd_mode=1, eps_mode=2
    )


def transform_coast_ablation(
    x1: torch.Tensor,
    x2: torch.Tensor,
    calib_data: torch.Tensor,
    target_dim: int = 512,
    orth: bool = True,
    crowd_mode: int = 1,
    eps_mode: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Factorial compression ablation operator on the complete (O, C, E_tau) lattice.

    Args:
        x1: First representation tensor of shape (N, d).
        x2: Second representation tensor of shape (M, d).
        calib_data: Calibration tensor of shape (K, d).
        target_dim: Target dimension k in {512, 256, 128, 64}.
        orth: Basis switch O in {False (0: Sigma), True (1: Sigma_perp)}.
        crowd_mode: Tempering exponent mode C in {-1, 0, 1, 2, 3, 4}:
            -1 -> Untempered gamma = 0.0 (Orthogonal PCA when orth=True)
             0 -> SNR schedule only gamma(k) = gamma_SNR(k) (SpecTemp schedule, w/o gamma_crowd)
             1 -> Full joint schedule gamma*(k) = max(gamma_crowd, gamma_SNR(k)) (COAST schedule)
             2 -> Crowding floor only gamma(k) = gamma_crowd (w/o gamma_SNR(k))
             3 -> Fixed full whitening gamma(k) = 1.0 (Whitening-k exponent)
             4 -> Fixed compromise exponent gamma(k) = 0.5 (gamma-Whitening of Li et al., 2026)
        eps_mode: Regularizer mode E_tau in {0, 1, 2}:
            0 -> Unregularized epsilon = 1e-6 (SpecTemp / Whitening floor)
            1 -> Unscaled crowding regularizer epsilon_full* (tau(k) = 1)
            2 -> Subspace-rank-scaled regularizer epsilon*(k) with tau(k)^2
    """
    spec = _get_cached_spectrum(calib_data, orth=orth)
    eigenvalues = spec["eigenvalues"]
    eigenvectors = spec["eigenvectors"]
    snr_np = spec["snr_np"]
    noise_variance = spec["noise_variance"]
    knee_idx = spec["knee_idx"]
    e_par = spec["e_par"]

    n_samples = calib_data.shape[0]
    d_spec = len(eigenvalues)
    r_eff = max(1, min(n_samples - 1, d_spec))
    k = min(target_dim, d_spec)

    e_disp = float(torch.clamp(eigenvalues.sum(), min=1e-12).item())
    lam_bar = e_disp / max(1, d_spec)
    gamma_crowd = min(1.0, (e_par / e_disp) ** 2)

    n_points = len(snr_np)
    snr_knee = float(snr_np[knee_idx - 1])
    if snr_knee <= 1e-12:
        gamma_snr = 1.0
    else:
        if k <= n_points:
            snr_k = float(snr_np[k - 1])
        else:
            snr_k = float(
                torch.clamp(
                    (eigenvalues[k - 1] - noise_variance)
                    / torch.clamp(noise_variance, min=1e-12),
                    min=0.0,
                ).item()
            )
        gamma_snr = max(0.0, min(1.0, snr_k / snr_knee))

    if crowd_mode == -1:
        gamma_val = 0.0
    elif crowd_mode == 0:
        gamma_val = gamma_snr
    elif crowd_mode == 1:
        gamma_val = max(gamma_crowd, gamma_snr)
    elif crowd_mode == 2:
        gamma_val = gamma_crowd
    elif crowd_mode == 3:
        gamma_val = 1.0
    else:
        gamma_val = 0.5

    lam_knee = max(float(eigenvalues[knee_idx - 1].item()), 1e-12)
    if eps_mode == 0:
        eps_val = 1e-6
    elif eps_mode == 1:
        disp_scale = max(e_disp * lam_bar, 1e-12)
        eps_val = max(
            (lam_knee ** (1.0 - gamma_crowd)) * (disp_scale**gamma_crowd), 1e-12
        )
    else:
        tau_k = min(1.0, k / r_eff)
        disp_scale = max((tau_k**2) * e_disp * lam_bar, 1e-12)
        eps_val = max(
            (lam_knee ** (1.0 - gamma_crowd)) * (disp_scale**gamma_crowd), 1e-12
        )

    if orth:
        u_mu = spec["u_mu"]
        if torch.linalg.norm(u_mu) >= 1e-12:
            x1_def = x1 - (x1 @ u_mu).unsqueeze(1) * u_mu
            x2_def = x2 - (x2 @ u_mu).unsqueeze(1) * u_mu
        else:
            x1_def = x1
            x2_def = x2
    else:
        mu = spec["mu"]
        x1_def = x1 - mu
        x2_def = x2 - mu

    u_k = eigenvectors[:, :k]
    lam_k = eigenvalues[:k]
    if crowd_mode == -1:
        w_k = u_k
    else:
        scales = torch.clamp(lam_k + eps_val, min=1e-12) ** (-gamma_val / 2.0)
        w_k = u_k * scales.unsqueeze(0)

    x1_trans = torch.matmul(x1_def, w_k)
    x2_trans = torch.matmul(x2_def, w_k)
    return l2_normalize(x1_trans), l2_normalize(x2_trans), gamma_val
