import torch

from src.transforms.common import l2_normalize
from src.transforms.compression.coast import _get_cached_spectrum


def transform_coast_full(
    x1: torch.Tensor,
    x2: torch.Tensor,
    calib_data: torch.Tensor,
    kneedle_s: float = 0.5,
    tail_percentile: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crowding-Orthogonal Adaptive Spectral Tempering (COAST) at full dimensionality (k = d)."""
    _ = (kneedle_s, tail_percentile)
    return transform_coast_full_ablation(
        x1, x2, calib_data, orth=True, crowd_mode=1, eps_mode=1
    )


def transform_coast_full_ablation(
    x1: torch.Tensor,
    x2: torch.Tensor,
    calib_data: torch.Tensor,
    orth: bool = True,
    crowd_mode: int = 1,
    eps_mode: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-dimension factorial ablation operator on the 2 x 3 x 3 lattice (O, C, E).

    Args:
        x1: First representation tensor of shape (N, d).
        x2: Second representation tensor of shape (M, d).
        calib_data: Calibration tensor of shape (K, d).
        orth: Basis switch O in {False (0: Sigma), True (1: Sigma_perp)}.
        crowd_mode: Tempering exponent mode C in {0, 1, 2}:
            0 -> Fixed full whitening gamma = 1.0 (Soft-ZCA / Whitening)
            1 -> Adaptive crowding equipartition gamma_crowd = min(1, (E_par / E_disp)^2) (COAST)
            2 -> Fixed compromise exponent gamma = 0.5 (gamma-Whitening)
        eps_mode: Regularizer mode E in {0, 1, 2}:
            0 -> Fixed epsilon = 1e-4 (Soft-ZCA floor)
            1 -> Adaptive crowding-interpolated epsilon_full* (COAST floor)
            2 -> Unregularized epsilon = 1e-6 (Whitening / SpecTemp floor)
    """
    spec = _get_cached_spectrum(calib_data, orth=orth)
    eigenvalues = spec["eigenvalues"]
    eigenvectors = spec["eigenvectors"]
    knee_idx = spec["knee_idx"]
    e_par = spec["e_par"]

    d_spec = len(eigenvalues)
    e_disp = float(torch.clamp(eigenvalues.sum(), min=1e-12).item())
    lam_bar = e_disp / max(1, d_spec)
    gamma_crowd = min(1.0, (e_par / e_disp) ** 2)

    if crowd_mode == 0:
        gamma_val = 1.0
    elif crowd_mode == 1:
        gamma_val = gamma_crowd
    else:
        gamma_val = 0.5

    if eps_mode == 0:
        eps_val = 1e-4
    elif eps_mode == 1:
        lam_knee = max(float(eigenvalues[knee_idx - 1].item()), 1e-12)
        disp_scale = max(e_disp * lam_bar, 1e-12)
        eps_val = max(
            (lam_knee ** (1.0 - gamma_crowd)) * (disp_scale**gamma_crowd), 1e-12
        )
    else:
        eps_val = 1e-6

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

    scales = torch.clamp(eigenvalues + eps_val, min=1e-12) ** (-gamma_val / 2.0)
    w_mat = torch.matmul(eigenvectors, torch.matmul(torch.diag(scales), eigenvectors.T))

    x1_trans = torch.matmul(x1_def, w_mat.T)
    x2_trans = torch.matmul(x2_def, w_mat.T)
    return l2_normalize(x1_trans), l2_normalize(x2_trans)
