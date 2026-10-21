#!/usr/bin/env python3
"""joint_fit.py — Legacy joint SED + dilution fit for WD+MS eclipsing systems.

Used only when JOINT_FIT_METHOD = "legacy" in main_all.py. The default method
is joint_fit_ideal.py; see its docstring for how the two differ. This method
needs the lnL_dil.npz table written by dilution_fit.py, so it cannot be
combined with MS_RADIUS_PRIOR = "uniform" (dilution_uniform_fit.py writes no
such table, and main_all.py refuses that combination).

Strategy
--------
The joint posterior combines two independent constraints:

  log p(θ | data) = log KDE_sed(θ)  +  lnL_dil(T_WD, R_WD, T_MS)

where:
  - KDE_sed  : gaussian_kde built from the 2-comp SED posterior [N, 6]
                columns: T_WD, R_WD, T_MS, R_MS, d, A_V
  - lnL_dil  : 3-D dilution likelihood table T_WD × R_WD × T_MS
                (RegularGridInterpolator, linear)

UltraNest ReactiveNestedSampler explores the 6-D parameter space
[T_WD, R_WD, T_MS, R_MS, d, A_V] with flat priors bounded by the
SED posterior [p0.5, p99.5] range (±2 % margin).

All output keys carry the _joint suffix.

The dilution table is evaluated along R_MS = R_BHAC15(T_MS): the sampled R_MS
only enters through the SED KDE term.

Outputs (<output_base>/TIC{id}_Gaia{id}/joint/):
  posterior_samples_joint.npy   [N, 6]  T_WD R_WD T_MS R_MS d A_V
  params_joint.json             p16/p50/p84 for all 6 parameters + logZ
  plot_joint.pdf                6 × 1-D marginal posteriors
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde
from scipy.interpolate import RegularGridInterpolator

from wdms_common import derive_seed

try:
    from ultranest import ReactiveNestedSampler
except ImportError:
    raise ImportError("ultranest is required: pip install ultranest")


# ══════════════════════════════════════════════════════════════════════════════
# Main entry
# ══════════════════════════════════════════════════════════════════════════════

def run_joint_fit(
    gaia_id,
    tic_id,
    output_base,
    posterior_samples_sed_path,   # .npy [N, 6]: T_WD R_WD T_MS R_MS d A_V
    lnL_dil_path,                 # .npz from dilution_fit.run_dilution_fit
    mode="final",
    seed=42,
):
    """
    Parameters
    ----------
    gaia_id, tic_id : int
    output_base : Path  — results root; subdirectory created automatically
    posterior_samples_sed_path : str or Path
        2-comp SED posterior, shape [N, 6] (T_WD, R_WD, T_MS, R_MS, d, A_V).
    lnL_dil_path : str or Path
        npz produced by dilution_fit.run_dilution_fit; must contain keys
        T_WD_axis, R_WD_axis, T_MS_axis, lnL_dil.
    mode : "test" | "final"
        "test"  → 16 live points (fast)
        "final" → 400 live points (production)
    seed : int
        NumPy's global random state, which UltraNest draws from, is seeded
        with derive_seed(seed, "joint_legacy") right before the run; both
        values are stored in params_joint.json ("seed_joint").

    Returns
    -------
    dict  — all keys carry the _joint suffix
    """
    out_dir = Path(output_base) / f"TIC{tic_id}_Gaia{gaia_id}" / "joint"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_live = int(JOINT_LEGACY_N_LIVE["test" if mode == "test" else "final"])
    print(f"\n[JOINT] mode={mode}  n_live={n_live}")
    print(f"[JOINT] Output → {out_dir}")

    # ── Step 1: SED posterior → KDE ───────────────────────────────────────
    samp2 = np.load(posterior_samples_sed_path)
    if samp2.ndim != 2 or samp2.shape[1] != 6:
        raise ValueError(
            f"SED posterior must be shape [N, 6], got {samp2.shape}. "
            "Pass the 2-comp posterior (posterior_samples_sed.npy)."
        )
    print(f"[JOINT] SED posterior loaded: {samp2.shape[0]} samples")

    kde_sed = gaussian_kde(samp2.T)   # evaluate: kde_sed(point_col_vec)

    # Parameter ordering: T_WD, R_WD, T_MS, R_MS, d, A_V
    PARAM_NAMES  = ["T_WD", "R_WD", "T_MS", "R_MS", "d", "A_V"]
    lo_bounds = np.percentile(samp2, 0.5,  axis=0)
    hi_bounds = np.percentile(samp2, 99.5, axis=0)

    # Widen by 2 % to avoid hard KDE edge artefacts near the boundary
    widths    = hi_bounds - lo_bounds
    lo_bounds = lo_bounds - 0.02 * widths
    hi_bounds = hi_bounds + 0.02 * widths
    # Physical guardrails
    lo_bounds[1] = max(lo_bounds[1], 1e-4)   # R_WD > 0
    lo_bounds[3] = max(lo_bounds[3], 1e-4)   # R_MS > 0
    lo_bounds[4] = max(lo_bounds[4], 1.0)    # d > 1 pc
    lo_bounds[5] = max(lo_bounds[5], 0.0)    # A_V >= 0

    print("[JOINT] Sampling bounds (SED posterior p0.5–p99.5 ± 2 %):")
    for name, lo, hi in zip(PARAM_NAMES, lo_bounds, hi_bounds):
        print(f"  {name:5s}: [{lo:.5g}, {hi:.5g}]")

    # ── Step 2: lnL_dil → 3-D interpolator ───────────────────────────────
    dil       = np.load(lnL_dil_path)
    T_WD_axis = dil["T_WD_axis"]   # (N_GRID,)
    R_WD_axis = dil["R_WD_axis"]   # (N_GRID,)
    T_MS_axis = dil["T_MS_axis"]   # (n_TMS,)
    lnL_arr   = dil["lnL_dil"]     # (N_GRID, N_GRID, n_TMS)

    print(
        f"[JOINT] lnL_dil grid: "
        f"T_WD {T_WD_axis.shape}  R_WD {R_WD_axis.shape}  T_MS {T_MS_axis.shape}"
    )

    # Floor replaces -inf so RegularGridInterpolator can handle it
    finite_mask = np.isfinite(lnL_arr)
    if not finite_mask.any():
        raise ValueError("[JOINT] lnL_dil contains no finite values.")
    lnL_floor  = float(np.nanmin(lnL_arr[finite_mask])) - 1.0
    lnL_finite = np.where(finite_mask, lnL_arr, lnL_floor)

    interp_lnL = RegularGridInterpolator(
        (T_WD_axis, R_WD_axis, T_MS_axis),
        lnL_finite,
        method="linear",
        bounds_error=False,
        fill_value=lnL_floor,
    )

    # ── Step 3: UltraNest prior_transform & loglikelihood ─────────────────
    ranges = hi_bounds - lo_bounds

    def prior_transform(cube):
        return lo_bounds + cube * ranges

    def log_likelihood(params):
        """params shape: (6,) — [T_WD, R_WD, T_MS, R_MS, d, A_V]"""
        T_WD, R_WD, T_MS, R_MS, d, A_V = params

        if R_WD <= 0 or R_MS <= 0 or d <= 0 or A_V < 0:
            return -1e100

        # SED posterior density (acts as the SED information in the joint posterior)
        pt       = np.array([[T_WD], [R_WD], [T_MS], [R_MS], [d], [A_V]])
        kde_val  = float(kde_sed(pt)[0])
        if kde_val <= 0:
            return -1e100
        lnL_sed  = np.log(kde_val)

        # Dilution likelihood (interpolated from 3-D table)
        lnL_dil_val = float(interp_lnL([[T_WD, R_WD, T_MS]])[0])

        return lnL_sed + lnL_dil_val

    # ── Step 4: Run UltraNest ──────────────────────────────────────────────
    log_dir = str(out_dir / "ultranest_joint")
    seed_run = derive_seed(seed, "joint_legacy")
    np.random.seed(seed_run)
    sampler = ReactiveNestedSampler(
        PARAM_NAMES,
        log_likelihood,
        prior_transform,
        log_dir=log_dir,
        resume="overwrite",
    )

    result_joint = sampler.run(
        min_num_live_points=n_live,
        show_status=True,
        viz_callback=False,
    )
    sampler.print_results()

    samples_joint = np.array(result_joint["samples"])     # [N_post, 6]
    logZ_joint    = float(result_joint["logz"])
    logZerr_joint = float(result_joint["logzerr"])

    print(f"\n[JOINT] logZ = {logZ_joint:.3f} ± {logZerr_joint:.3f}")
    print(f"[JOINT] Posterior samples: {samples_joint.shape[0]}")

    # ── Step 5: Credible intervals ────────────────────────────────────────
    p16 = np.percentile(samples_joint, 16, axis=0)
    p50 = np.percentile(samples_joint, 50, axis=0)
    p84 = np.percentile(samples_joint, 84, axis=0)

    T_WD_joint, R_WD_joint, T_MS_joint, R_MS_joint, d_joint, AV_joint = p50
    T_WD_p16, R_WD_p16, T_MS_p16, R_MS_p16, d_p16, AV_p16 = p16
    T_WD_p84, R_WD_p84, T_MS_p84, R_MS_p84, d_p84, AV_p84 = p84

    print(f"[JOINT] T_WD = {T_WD_joint:.0f} K  "
          f"(-{T_WD_joint-T_WD_p16:.0f} / +{T_WD_p84-T_WD_joint:.0f})")
    print(f"[JOINT] R_WD = {R_WD_joint:.5f} R_sun  "
          f"(-{R_WD_joint-R_WD_p16:.5f} / +{R_WD_p84-R_WD_joint:.5f})")
    print(f"[JOINT] T_MS = {T_MS_joint:.0f} K  "
          f"(-{T_MS_joint-T_MS_p16:.0f} / +{T_MS_p84-T_MS_joint:.0f})")
    print(f"[JOINT] R_MS = {R_MS_joint:.4f} R_sun  "
          f"(-{R_MS_joint-R_MS_p16:.4f} / +{R_MS_p84-R_MS_joint:.4f})")
    print(f"[JOINT] d    = {d_joint:.1f} pc  "
          f"(-{d_joint-d_p16:.1f} / +{d_p84-d_joint:.1f})")
    print(f"[JOINT] A_V  = {AV_joint:.3f} mag  "
          f"(-{AV_joint-AV_p16:.3f} / +{AV_p84-AV_joint:.3f})")

    # ── Step 6: Save posterior samples ────────────────────────────────────
    path_samp = out_dir / "posterior_samples_joint.npy"
    np.save(path_samp, samples_joint.astype(np.float64))
    print(f"[JOINT] Saved posterior → {path_samp}")

    # ── Step 7: Save params_joint.json ────────────────────────────────────
    params_joint = {
        "gaia_id": gaia_id,
        "tic_id":  tic_id,
        "T_WD_joint": {"p16": float(T_WD_p16), "p50": float(T_WD_joint), "p84": float(T_WD_p84)},
        "R_WD_joint": {"p16": float(R_WD_p16), "p50": float(R_WD_joint), "p84": float(R_WD_p84)},
        "T_MS_joint": {"p16": float(T_MS_p16), "p50": float(T_MS_joint), "p84": float(T_MS_p84)},
        "R_MS_joint": {"p16": float(R_MS_p16), "p50": float(R_MS_joint), "p84": float(R_MS_p84)},
        "d_joint":    {"p16": float(d_p16),    "p50": float(d_joint),    "p84": float(d_p84)},
        "AV_joint":   {"p16": float(AV_p16),   "p50": float(AV_joint),   "p84": float(AV_p84)},
        "logZ_joint":    logZ_joint,
        "logZerr_joint": logZerr_joint,
        "seed_joint":    {"master": int(seed), "ultranest": int(seed_run)},
    }
    path_params = out_dir / "params_joint.json"
    with open(path_params, "w", encoding="utf-8") as fp:
        json.dump(params_joint, fp, indent=2)
    print(f"[JOINT] Saved params  → {path_params}")

    # ── Step 8: Plot ───────────────────────────────────────────────────────
    path_plot = out_dir / "plot_joint.pdf"
    _plot_joint(
        samples     = samples_joint,
        param_names = PARAM_NAMES,
        p16         = p16,
        p50         = p50,
        p84         = p84,
        out_path    = path_plot,
    )

    # ── Return ─────────────────────────────────────────────────────────────
    return {
        # Medians (p50)
        "T_WD_joint": float(T_WD_joint),
        "R_WD_joint": float(R_WD_joint),
        "T_MS_joint": float(T_MS_joint),
        "R_MS_joint": float(R_MS_joint),
        "d_joint":    float(d_joint),
        "AV_joint":   float(AV_joint),
        # Full 1-σ intervals
        "T_WD_p16_joint": float(T_WD_p16), "T_WD_p84_joint": float(T_WD_p84),
        "R_WD_p16_joint": float(R_WD_p16), "R_WD_p84_joint": float(R_WD_p84),
        "T_MS_p16_joint": float(T_MS_p16), "T_MS_p84_joint": float(T_MS_p84),
        "R_MS_p16_joint": float(R_MS_p16), "R_MS_p84_joint": float(R_MS_p84),
        "d_p16_joint":    float(d_p16),    "d_p84_joint":    float(d_p84),
        "AV_p16_joint":   float(AV_p16),   "AV_p84_joint":   float(AV_p84),
        # Bayesian evidence
        "logZ_joint":    logZ_joint,
        "logZerr_joint": logZerr_joint,
        # File paths
        "posterior_samples_joint_path": str(path_samp),
        "params_joint_path":            str(path_params),
        "plot_joint_path":              str(path_plot),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plot helper
# ══════════════════════════════════════════════════════════════════════════════

# UltraNest live points of the legacy joint fit, per sampling mode.
# Defaults when this module is used alone; main_all.py sets its own values
# (see its "Fit settings" section, function apply_fit_settings).
JOINT_LEGACY_N_LIVE = {"test": 16, "final": 400}

_LABELS = {
    "T_WD": r"$T_\mathrm{WD}$ [K]",
    "R_WD": r"$R_\mathrm{WD}$ [$R_\odot$]",
    "T_MS": r"$T_\mathrm{MS}$ [K]",
    "R_MS": r"$R_\mathrm{MS}$ [$R_\odot$]",
    "d":    r"$d$ [pc]",
    "A_V":  r"$A_V$ [mag]",
}
_COLORS = ["steelblue", "steelblue", "darkorange", "darkorange", "dimgray", "dimgray"]


def _plot_joint(samples, param_names, p16, p50, p84, out_path):
    """6-panel 1-D marginal posteriors for all joint parameters."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    axes = axes.ravel()

    for ax, name, col, lo, med, hi, col_data in zip(
        axes, param_names, _COLORS, p16, p50, p84, samples.T
    ):
        ax.hist(col_data, bins=50, alpha=0.75, color=col, density=True)
        ax.axvline(lo,  color="k",   ls="--", lw=1.1, label=f"p16 = {lo:.4g}")
        ax.axvline(med, color="red", ls="-",  lw=1.5, label=f"p50 = {med:.4g}")
        ax.axvline(hi,  color="k",   ls="--", lw=1.1, label=f"p84 = {hi:.4g}")
        ax.set_xlabel(_LABELS.get(name, name))
        ax.set_ylabel("density")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3)

    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[JOINT] Saved plot   → {out_path}")
