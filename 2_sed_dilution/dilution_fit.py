#!/usr/bin/env python3
"""dilution_fit.py — Transit dilution fitting for WD+MS eclipsing systems.

Physics (total primary eclipse, R_WD << R_MS):
    depth = F_WD / (F_WD + F_MS) = 1 / (1 + q)
    q = (R_MS / R_WD)^2 × F_MS_band(T_MS, logg_MS) / F_WD_band(T_WD, logg_WD)

Every band flux and gravity comes from depth_model.DepthFluxModel, shared with
the SED term and the other depth calculations:

logg_WD  : SED policy — Delaunay interpolation of the Bédard 2020 cooling track
           with the wd_logg_fixed fallback outside it, clipped to the Koester
           logg range (no inverse-distance weighting)
F_WD_band: Koester DA spectra (models/koester2, files da*.dk.dat.txt)
           integrated over the filter profile, interpolated bilinearly in ln F
           over (T_WD, logg_WD). Only DA files are read; with
           wd_atmo_type="DB" only the cooling track changes.
F_MS_band: the SED reference companion model (STScI PHOENIX with the default
           settings), interpolated bilinearly in ln F over (T_MS, logg_MS)
logg_MS  : the SED companion-gravity prescription (ms_logg_mode)
R_MS     : BHAC15 T_MS → R_MS crossmatch (local-window median)

Step 2 of the pipeline when MS_RADIUS_PRIOR is "bhac15" or "brown2022" in
main_all.py. With MS_RADIUS_PRIOR = "uniform", dilution_uniform_fit.py is used
instead.

Monte Carlo: each iteration draws (T_WD, R_WD) from the SED posterior and
samples T_MS from the depth likelihood of the *observed* depths along the
radius relation; R_MS then receives the relation's own scatter (prior scatter,
not measurement noise). The observed depth is not perturbed, so its
uncertainty enters once, through the likelihood.

Inputs from run_dilution_fit:
    posterior_samples_sed_path  — .npy [N, 6] from sed_fit.run_sed_fit
                                  (only T_WD and R_WD are used)
    depth_by_filter             — {"filter_name": (depth, err_sym)}, depths as
                                  fractions of the out-of-eclipse flux

Outputs (<output_base>/TIC{id}_Gaia{id}/dilution/):
    posterior_samples_dil.npy          [N_valid, 2]  T_MS [K], R_MS [R_sun]
    posterior_samples_dil_matched.npy  [N_valid, 4]  T_WD [K], R_WD [R_sun],
                                       T_MS [K], R_MS [R_sun] — the white dwarf
                                       actually used for each accepted sample
    lnL_dil.npz                 3D grid       T_WD × R_WD × T_MS, read by the
                                              legacy joint fit (joint_fit.py)
    params_dil.json             p16/p50/p84 for T_MS, R_MS, logg_WD
    plot_dil.pdf                posteriors + BHAC15 R_MS(T_MS) relation

Variable naming: all exported keys carry the _dil suffix.

This module also provides run_eclipse_radius_diagnostic, which inverts the very
same dilution equation for R_MS instead of solving it along the BHAC15
relation. It needs no R(T) at all and produces, in
<output_base>/TIC{id}_Gaia{id}/eclipse_radius/, the comparison between the radius the
eclipse demands and the radius the SED demands. Its exported keys carry the
_ecl suffix.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

import pandas as pd

from mass import BROWN2022_RADIUS_SCATTER
from wdms_common import M_SUN_G, T_MS_MIN_K, T_MS_MAX_K
# Shared depth-term model; the filter / Koester helpers and the depth
# inversion moved there and are re-exported here for existing imports.
from depth_model import (                                       # noqa: F401
    DepthFluxModel,
    _band_flux,
    _load_filter_profile,
    _load_koester_catalog,
    _read_koester,
    eclipse_required_radius,
)
from sed_fit import (
    ATMO_MS_LOGG_FIXED,
    ATMO_MS_LOGG_MODE,
    ATMO_WD_LOGG_FIXED,
    ATMO_WD_LOGG_MODE,
)

# ── Physical constants ─────────────────────────────────────────────────────
_G_CGS     = 6.67430e-8    # cm³/(g·s²)
_M_SUN_G   = M_SUN_G       # g, astropy.constants.M_sun (wdms_common)
_R_SUN_CM  = 6.957e10      # cm


def _load_profiles(active_filters, filter_profile_files, filter_wavelength_cuts):
    """{filter: (wave_A, transmission)} with the configured wavelength cuts."""
    profiles = {}
    for fk in active_filters:
        cuts = (filter_wavelength_cuts or {}).get(fk, {})
        profiles[fk] = _load_filter_profile(
            filter_profile_files[fk],
            min_A=cuts.get("min_A", None),
            max_A=cuts.get("max_A", None),
        )
    return profiles


# ══════════════════════════════════════════════════════════════════════════════
# BHAC15 R_MS(T_MS) relation
# ══════════════════════════════════════════════════════════════════════════════

def _load_bhac15(csv_path, age_logt_min=8.0, age_logt_max=None):
    df = pd.read_csv(csv_path)
    for col in ["M/Ms", "log t(yr)", "Teff", "g", "R/Rs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Teff", "R/Rs", "g"]).copy()

    if "log t(yr)" in df.columns:
        before = len(df)
        df = df[df["log t(yr)"] >= age_logt_min].copy()
        if age_logt_max is not None:
            df = df[df["log t(yr)"] <= age_logt_max].copy()
        print(f"[DIL] BHAC15 age filter (log_t≥{age_logt_min}): {len(df)}/{before} points kept")

    if len(df) == 0:
        raise RuntimeError("BHAC15 grid empty after age filtering")
    return df


def _bhac15_r_stats(T, df_bh, delta_T=100.0, min_pts=3, ms_relation=None):
    """p16/p50/p84 of R_MS within |ΔTeff| ≤ delta_T of T.

    With ms_relation set to a mass.MDwarfRelation, each theoretical radius is
    scaled by the ratio of the semi-empirical to the theoretical relation at the
    same mass, so the prior carries the 5-15 per cent inflation that M dwarfs
    show relative to the models (Brown et al. 2022). Scaling rather than
    substituting keeps the age spread that sets the width of this prior, and
    leaves the low-mass end untouched, where the semi-empirical relation is the
    BHAC15 isochrone itself. The intrinsic scatter of the relation is then added
    to the width so the fit is not handed a relation it must believe exactly.
    """
    local = df_bh[np.abs(df_bh["Teff"].to_numpy(float) - float(T)) <= float(delta_T)]
    if len(local) < min_pts:
        return np.nan, np.nan, np.nan

    r_vals = local["R/Rs"].to_numpy(float)
    if ms_relation is not None:
        r_vals = r_vals * ms_relation.inflation_factor(local["M/Ms"].to_numpy(float))

    r16, r50, r84 = np.percentile(r_vals, [16, 50, 84])
    if ms_relation is not None:
        half = np.hypot(0.5 * (r84 - r16), BROWN2022_RADIUS_SCATTER * r50)
        r16, r84 = r50 - half, r50 + half
    return float(r16), float(r50), float(r84)


def draw_two_sided_half_normal(rng, p16, p50, p84):
    """One draw from a median-centred, two-sided half-normal distribution.

    The side is chosen first, with probability 0.5 each, and then the width of
    that side is used:

        below the median : p50 - |N(0, p50 - p16)|
        above the median : p50 + |N(0, p84 - p50)|

    Each side therefore carries exactly half of the probability, so p50 is the
    median of the draws, and the lower and upper widths stay distinct when the
    interval is asymmetric. This is not the usual split-normal distribution,
    whose density is continuous at the mode and whose side probabilities are
    proportional to the two widths.

    Widths are floored at 1e-8 (as before), which also absorbs a degenerate or
    mis-ordered interval: a negative width becomes the floor rather than
    flipping the side. The caller applies any physical constraint (e.g. R > 0).
    """
    if rng.random() < 0.5:
        return float(p50) - abs(rng.normal(0.0, max(float(p50) - float(p16), 1e-8)))
    return float(p50) + abs(rng.normal(0.0, max(float(p84) - float(p50), 1e-8)))


def _build_tms_grid(df_bh, t_ms_min, t_ms_max, t_ms_step,
                    bhac_delta_t, bhac_min_pts, active_filters,
                    ms_relation=None, depth_model=None):
    """Discrete T_MS grid with the R(T) statistics and, optionally, F_MS.

    ms_relation, when given, switches the radius prior from the theoretical
    BHAC15 radii to the semi-empirical ones of Brown et al. 2022; see
    _bhac15_r_stats.

    With depth_model (depth_model.DepthFluxModel), the companion gravity at
    each node is the SED prescription evaluated at (T_MS, R50), and F_MS is
    interpolated in that model; nodes where it is undefined are dropped.
    """
    T2_grid = np.arange(float(t_ms_min), float(t_ms_max) + float(t_ms_step),
                         float(t_ms_step), dtype=float)
    # Never tabulate beyond the requested upper bound (possible when the
    # interval is not a multiple of the step).
    T2_grid = T2_grid[T2_grid <= float(t_ms_max) + 1e-9]
    n = len(T2_grid)

    R2_p16 = np.zeros(n); R2_p50 = np.zeros(n); R2_p84 = np.zeros(n)
    for i, T in enumerate(T2_grid):
        r16, r50, r84 = _bhac15_r_stats(T, df_bh, bhac_delta_t, bhac_min_pts,
                                        ms_relation)
        R2_p16[i] = r16; R2_p50[i] = r50; R2_p84[i] = r84

    valid = np.isfinite(R2_p50) & (R2_p50 > 0)
    logg2 = np.full(n, np.nan)
    F_MS_grid = {fk: np.full(n, np.nan) for fk in active_filters}
    if depth_model is not None:
        logg2[valid] = depth_model.ms_logg(T2_grid[valid], R2_p50[valid])
        for fk in active_filters:
            F_MS_grid[fk][valid] = depth_model.ms_band_flux(fk, T2_grid[valid], logg2[valid])
            valid &= np.isfinite(F_MS_grid[fk]) & (F_MS_grid[fk] > 0)

    if not valid.any():
        raise RuntimeError("[DIL] T_MS grid is empty after BHAC15/companion-model filtering")
    t_lo_v, t_hi_v = float(T2_grid[valid].min()), float(T2_grid[valid].max())
    print(f"[DIL] T_MS grid: {valid.sum()}/{n} valid points [{t_lo_v:.0f}–{t_hi_v:.0f} K]")
    if t_lo_v > float(t_ms_min) + 1e-6 or t_hi_v < float(t_ms_max) - 1e-6:
        print(f"[DIL] WARNING: BHAC15/companion-model coverage narrows the requested "
              f"T_MS bound [{float(t_ms_min):.0f}, {float(t_ms_max):.0f}] K to "
              f"[{t_lo_v:.0f}, {t_hi_v:.0f}] K (no extrapolation)")
    print(f"[DIL] R_MS grid: {R2_p50[valid].min():.4f}–{R2_p50[valid].max():.4f} R_sun "
          f"({'Brown 2022 semi-empirical' if ms_relation is not None else 'BHAC15 theoretical'})")

    return {
        "T2_grid": T2_grid[valid],
        "R2_p16":  R2_p16[valid],
        "R2_p50":  R2_p50[valid],
        "R2_p84":  R2_p84[valid],
        "logg2":   logg2[valid],
        "F_MS":    {fk: F_MS_grid[fk][valid] for fk in active_filters},
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_dilution_fit(
    gaia_id,
    tic_id,
    output_base,
    depth_by_filter,
    filter_profile_files,
    posterior_samples_sed_path,
    bhac15_csv,
    koester_dir,
    da_track_path,
    db_track_path,
    wd_atmo_type="DA",
    mode="final",
    t_ms_min=T_MS_MIN_K,
    t_ms_max=T_MS_MAX_K,
    t_ms_step=100.0,
    bhac_local_delta_t=100.0,
    bhac_local_min_points=3,
    bhac_age_logt_min=8.0,
    bhac_age_logt_max=None,
    photon_counting=True,
    filter_wavelength_cuts=None,
    seed=42,
    ms_relation=None,
    ms_atmosphere="phoenix",
    atmo_cache_dir=None,
    atmo_grid_ranges=None,
    ms_logg_mode=ATMO_MS_LOGG_MODE,
    ms_logg_fixed=ATMO_MS_LOGG_FIXED,
    wd_logg_mode=ATMO_WD_LOGG_MODE,
    wd_logg_fixed=ATMO_WD_LOGG_FIXED,
):
    """Run transit dilution T_MS/R_MS inference for a WD+MS eclipsing system.

    Parameters
    ----------
    gaia_id, tic_id : int
    output_base : str or Path
    depth_by_filter : dict
        {"filter_name": (depth, err_sym)}   e.g. {"iz": (0.685, 0.025)}
    filter_profile_files : dict
        {"filter_name": Path}  ASCII transmission curve files
    posterior_samples_sed_path : str or Path
        .npy [N, 6] from sed_fit.run_sed_fit  (T_WD, R_WD, T_MS, R_MS, d, AV)
    bhac15_csv : str or Path
    koester_dir : str or Path   directory of the Koester DA spectra
    da_track_path, db_track_path : str or Path   Bédard 2020 DA / DB CSVs
    wd_atmo_type : {"DA", "DB"}   selects the cooling track only
    mode : {"test", "final"}
        test  → N_MC = 5 000,  lnL grid 10×10×n_T_MS
        final → N_MC = 50 000, lnL grid 30×30×n_T_MS
    t_ms_min, t_ms_max, t_ms_step : T_MS grid range [K]
    bhac_local_delta_t : half-width of BHAC15 local window [K]
    bhac_local_min_points : minimum BHAC15 track points in window
    bhac_age_logt_min/max : BHAC15 age filter [log10 yr]
    photon_counting : bool   photon-counting vs energy-weighted band integration
    filter_wavelength_cuts : dict or None
        {"filter_name": {"min_A": float, "max_A": float}}
    seed : int
    ms_relation : mass.MDwarfRelation or None   (brown2022 radius prior)
    ms_atmosphere, atmo_cache_dir, atmo_grid_ranges, ms_logg_mode, ms_logg_fixed
        SED companion model and gravity prescription (shared depth model).
    wd_logg_mode, wd_logg_fixed
        SED white-dwarf gravity policy (shared depth model).

    Returns
    -------
    dict with _dil-suffixed keys
    """
    gaia_id        = int(gaia_id)
    tic_id         = int(tic_id)
    active_filters = list(depth_by_filter.keys())
    atmo           = str(wd_atmo_type).strip().upper()

    out_dir = Path(output_base) / f"TIC{tic_id}_Gaia{gaia_id}" / "dilution"
    out_dir.mkdir(parents=True, exist_ok=True)

    sampling = DILUTION_SAMPLING["test" if mode == "test" else "final"]
    N_MC   = int(sampling["n_mc"])
    N_GRID = int(sampling["n_grid"])               # T_WD and R_WD axes of lnL grid

    rng = np.random.default_rng(seed)

    print(f"\n{'='*60}")
    print(f"[DIL] Gaia DR3 {gaia_id}  /  TIC {tic_id}  /  mode={mode}")
    print(f"[DIL] Filters : {active_filters}")
    print(f"[DIL] N_MC={N_MC}  lnL_grid={N_GRID}×{N_GRID}×n_TMS")
    print(f"{'='*60}")

    # ── Step 1: Load SED posterior samples ────────────────────────────────
    samp2 = np.load(posterior_samples_sed_path)
    if samp2.ndim != 2 or samp2.shape[1] < 2:
        raise ValueError(f"posterior_samples_sed must be [N, ≥2], got {samp2.shape}")

    T_WD_post = samp2[:, 0].astype(float)
    R_WD_post = samp2[:, 1].astype(float)
    valid_post = np.isfinite(T_WD_post) & np.isfinite(R_WD_post) & (R_WD_post > 0)
    T_WD_post  = T_WD_post[valid_post]
    R_WD_post  = R_WD_post[valid_post]
    print(f"[DIL] SED posterior samples: {len(T_WD_post)} valid")
    print(f"[DIL] T_WD p50 = {np.nanpercentile(T_WD_post, 50):.0f} K  "
          f"R_WD p50 = {np.nanpercentile(R_WD_post, 50):.5f} R_sun")

    # ── Step 2: Shared depth model (filters, Koester, companion, gravities)
    profiles = _load_profiles(active_filters, filter_profile_files, filter_wavelength_cuts)
    for fk, (pw, pt) in profiles.items():
        active_range = pw[pt > 0]
        print(f"[DIL] Filter {fk}: {active_range.min():.0f}–{active_range.max():.0f} Å")
    track_path = da_track_path if atmo == "DA" else db_track_path
    depth = DepthFluxModel(
        profiles, koester_dir, wd_cooling_track=track_path,
        ms_atmosphere=ms_atmosphere, atmo_cache_dir=atmo_cache_dir,
        atmo_grid_ranges=atmo_grid_ranges, ms_logg_mode=ms_logg_mode,
        ms_logg_fixed=ms_logg_fixed, wd_logg_mode=wd_logg_mode,
        wd_logg_fixed=wd_logg_fixed, photon_counting=photon_counting)

    logg_WD_post = depth.wd_logg(T_WD_post, R_WD_post)
    logg_WD_p16 = float(np.nanpercentile(logg_WD_post, 16))
    logg_WD_med = float(np.nanpercentile(logg_WD_post, 50))
    logg_WD_p84 = float(np.nanpercentile(logg_WD_post, 84))
    print(f"[DIL] logg_WD: {logg_WD_p16:.3f} / {logg_WD_med:.3f} / {logg_WD_p84:.3f}  "
          f"[p16/p50/p84]")

    # ── Step 3: BHAC15 relation and companion fluxes on the T_MS grid ─────
    df_bh = _load_bhac15(bhac15_csv, bhac_age_logt_min, bhac_age_logt_max)
    grid  = _build_tms_grid(
        df_bh, t_ms_min, t_ms_max, t_ms_step,
        bhac_local_delta_t, bhac_local_min_points, active_filters,
        ms_relation=ms_relation, depth_model=depth,
    )
    T2_grid = grid["T2_grid"]
    R2_p16  = grid["R2_p16"]
    R2_p50  = grid["R2_p50"]
    R2_p84  = grid["R2_p84"]
    F_MS    = grid["F_MS"]

    # Continuous refinement axis around each node j: _N_SUB points spanning
    # [T_{j-1}, T_{j+1}], with R50 interpolated along the relation and F_MS
    # evaluated directly in the depth model (no nearest-node selection).
    _N_SUB = 200
    fine = {}
    for j in range(len(T2_grid)):
        j_lo = max(0, j - 1)
        j_hi = min(len(T2_grid) - 1, j + 1)
        T_fine = np.linspace(T2_grid[j_lo], T2_grid[j_hi], _N_SUB)
        r50_fine = np.interp(T_fine, T2_grid, R2_p50)
        g_fine = depth.ms_logg(T_fine, r50_fine)
        fine[j] = (T_fine, r50_fine,
                   {fk: depth.ms_band_flux(fk, T_fine, g_fine) for fk in active_filters})

    # ── Step 4: Monte Carlo inference ─────────────────────────────────────
    print(f"\n[DIL] Running MC ({N_MC} iterations)…")

    n_post   = len(T_WD_post)
    idx_draw = rng.integers(0, n_post, size=N_MC)
    T_WD_mc  = T_WD_post[idx_draw]
    R_WD_mc  = R_WD_post[idx_draw]
    logg_WD_mc = logg_WD_post[idx_draw]
    F_WD_mc  = {fk: depth.wd_band_flux(fk, T_WD_mc, logg_WD_mc) for fk in active_filters}

    T_MS_samples = np.full(N_MC, np.nan)
    R_MS_samples = np.full(N_MC, np.nan)
    depth_mod_samp = {fk: np.full(N_MC, np.nan) for fk in active_filters}

    for k in range(N_MC):
        if k % 5000 == 0:
            print(f"[DIL]   MC {k}/{N_MC}", end="\r", flush=True)

        r1 = float(R_WD_mc[k])
        if not (np.isfinite(T_WD_mc[k]) and np.isfinite(r1) and r1 > 0):
            continue
        F1 = {fk: float(F_WD_mc[fk][k]) for fk in active_filters}
        if not all(np.isfinite(v) and v > 0 for v in F1.values()):
            continue

        # Depth likelihood of the observed depths at the grid nodes
        chi2 = np.zeros(len(T2_grid))
        for fk in active_filters:
            d_obs, d_err = depth_by_filter[fk]
            d_mod = depth.depth(fk, r1, F1[fk], R2_p50, F_MS[fk])
            chi2 += ((float(d_obs) - d_mod) / float(d_err))**2
        ok = np.isfinite(chi2)
        if not ok.any():
            continue

        loglike = np.full(len(T2_grid), -np.inf)
        loglike[ok] = -0.5 * chi2[ok]
        loglike    -= np.nanmax(loglike[ok])
        weights = np.where(ok, np.exp(loglike), 0.0)
        w_sum = weights.sum()
        if w_sum <= 0:
            continue
        weights /= w_sum

        # ── Discrete selection: identify the mode region ───────────────
        j = int(rng.choice(len(T2_grid), p=weights))

        # ── Continuous refinement within neighbourhood [j-1, j+1] ─────
        # The same likelihood evaluated directly on the fine axis.
        T_fine, r50_fine, F_fine = fine[j]
        chi2_f = np.zeros(_N_SUB)
        for fk in active_filters:
            d_obs, d_err = depth_by_filter[fk]
            dm = depth.depth(fk, r1, F1[fk], r50_fine, F_fine[fk])
            chi2_f += ((float(d_obs) - dm) / float(d_err))**2
        ok_f = np.isfinite(chi2_f)
        if not ok_f.any():
            continue
        logL_fine = np.where(ok_f, -0.5 * chi2_f, -np.inf)
        logL_fine -= np.max(logL_fine[ok_f])
        w_fine = np.where(ok_f, np.exp(logL_fine), 0.0)
        w_fine /= w_fine.sum()
        i_f = int(rng.choice(_N_SUB, p=w_fine))
        T_ms_cont = float(T_fine[i_f])
        T_MS_samples[k] = T_ms_cont

        # ── R_MS: the relation's own scatter at T_ms_cont (prior scatter) ──
        r50 = float(np.interp(T_ms_cont, T2_grid, R2_p50))
        r16 = float(np.interp(T_ms_cont, T2_grid, R2_p16))
        r84 = float(np.interp(T_ms_cont, T2_grid, R2_p84))
        draw = draw_two_sided_half_normal(rng, r16, r50, r84)
        R_MS_samples[k] = max(float(draw), 1e-6)

        # ── Model depth at the selected T_MS, on the relation (R = r50) ──
        for fk in active_filters:
            depth_mod_samp[fk][k] = float(depth.depth(fk, r1, F1[fk], r50_fine[i_f],
                                                      F_fine[fk][i_f]))

    valid_mc = np.isfinite(T_MS_samples) & np.isfinite(R_MS_samples)
    n_valid  = int(valid_mc.sum())
    print(f"\n[DIL] MC done: {n_valid}/{N_MC} valid samples")

    if n_valid < 10:
        raise RuntimeError(f"[DIL] Too few valid MC samples ({n_valid})")

    T_MS_p16, T_MS_p50, T_MS_p84 = np.percentile(T_MS_samples[valid_mc], [16, 50, 84])
    R_MS_p16, R_MS_p50, R_MS_p84 = np.percentile(R_MS_samples[valid_mc], [16, 50, 84])

    print(f"[DIL] T_MS = {T_MS_p50:.0f} K  "
          f"(-{T_MS_p50-T_MS_p16:.0f}/+{T_MS_p84-T_MS_p50:.0f})")
    print(f"[DIL] R_MS = {R_MS_p50:.4f} R_sun  "
          f"(-{R_MS_p50-R_MS_p16:.4f}/+{R_MS_p84-R_MS_p50:.4f})")

    for fk in active_filters:
        dm = depth_mod_samp[fk][valid_mc]
        dm = dm[np.isfinite(dm)]
        if len(dm) > 0:
            d16, d50, d84 = np.percentile(dm, [16, 50, 84])
            d_obs, d_err  = depth_by_filter[fk]
            print(f"[DIL] depth model [{fk}] = {d50:.4f} "
                  f"(-{d50-d16:.4f}/+{d84-d50:.4f})")
            print(f"[DIL] depth obs   [{fk}] = {d_obs:.4f} ± {d_err:.4f}")

    # ── Step 5: Save posterior samples ────────────────────────────────────
    post_arr  = np.column_stack([T_MS_samples[valid_mc],
                                 R_MS_samples[valid_mc]]).astype(np.float64)
    path_samp = out_dir / "posterior_samples_dil.npy"
    np.save(path_samp, post_arr)
    # Same rows, with the white dwarf each sample was actually computed with,
    # so that plots can show the real T_WD/R_WD–T_MS/R_MS correlations.
    matched = np.column_stack([T_WD_mc[valid_mc], R_WD_mc[valid_mc],
                               T_MS_samples[valid_mc], R_MS_samples[valid_mc]]
                              ).astype(np.float64)
    path_matched = out_dir / "posterior_samples_dil_matched.npy"
    np.save(path_matched, matched)
    print(f"[DIL] Saved posterior samples → {path_samp} (+ matched [T_WD, R_WD, T_MS, R_MS])")

    # ── Step 6: Tabulate lnL_dil [T_WD, R_WD, T_MS] ──────────────────────
    # T_WD and R_WD axes span the [p0.5, p99.5] credible region of the SED
    # posterior. At each (T_WD_i, R_WD_j) node logg_WD follows the shared
    # white-dwarf gravity policy, and F_WD is interpolated in the depth model.
    T_WD_axis = np.linspace(
        float(np.nanpercentile(T_WD_post, 0.5)),
        float(np.nanpercentile(T_WD_post, 99.5)),
        N_GRID,
    )
    R_WD_axis = np.linspace(
        float(np.nanpercentile(R_WD_post, 0.5)),
        float(np.nanpercentile(R_WD_post, 99.5)),
        N_GRID,
    )
    T_MS_axis = T2_grid.copy()    # discrete BHAC15 grid (n_T_MS points)
    R_MS_axis = R2_p50.copy()     # BHAC15 median R_MS at each T_MS node

    n_tms = len(T_MS_axis)
    print(f"\n[DIL] Tabulating lnL_dil [{N_GRID}×{N_GRID}×{n_tms}]…")

    lnL_dil = np.full((N_GRID, N_GRID, n_tms), -np.inf, dtype=np.float64)

    for i, T1 in enumerate(T_WD_axis):
        for j, R1 in enumerate(R_WD_axis):
            if R1 <= 0:
                continue
            logg1 = depth.wd_logg(T1, R1)
            F_WD = {fk: depth.wd_band_flux(fk, T1, logg1) for fk in active_filters}
            if not all(np.isfinite(v) and v > 0 for v in F_WD.values()):
                continue

            logL = np.zeros(n_tms)
            for fk in active_filters:
                d_obs, d_err = depth_by_filter[fk]
                sigma     = max(float(d_err), 1e-6)
                depth_mod = depth.depth(fk, R1, F_WD[fk], R_MS_axis, F_MS[fk])
                logL     += -0.5 * ((float(d_obs) - depth_mod) / sigma)**2
                logL     += -np.log(sigma * np.sqrt(2.0 * np.pi))
            lnL_dil[i, j, :] = np.where(np.isfinite(logL), logL, -np.inf)

    path_lnL = out_dir / "lnL_dil.npz"
    np.savez(
        path_lnL,
        T_WD_axis   = T_WD_axis,
        R_WD_axis   = R_WD_axis,
        T_MS_axis   = T_MS_axis,
        R_MS_axis   = R_MS_axis,
        lnL_dil     = lnL_dil,
        description = np.array(
            "lnL_dil[i,j,k] = ln p(depth|T_WD_i,R_WD_j,T_MS_k,R_MS_BHAC(T_MS_k)); "
            "logg_WD from the shared cooling-track policy, band fluxes "
            "interpolated bilinearly in ln F (depth_model)"
        ),
    )
    print(f"[DIL] Saved lnL_dil → {path_lnL}")

    # ── Step 7: Save params_dil.json ──────────────────────────────────────
    depth_summary_dil = {}
    for fk in active_filters:
        dm = depth_mod_samp[fk][valid_mc]
        dm = dm[np.isfinite(dm)]
        d_obs, d_err = depth_by_filter[fk]
        entry = {"obs": float(d_obs), "err_sym": float(d_err)}
        if len(dm) > 0:
            d16, d50, d84 = np.percentile(dm, [16, 50, 84])
            entry.update({"model_p16": float(d16),
                          "model_p50": float(d50),
                          "model_p84": float(d84)})
        depth_summary_dil[fk] = entry

    params_dil = {
        "gaia_id": gaia_id, "tic_id": tic_id,
        "T_MS_dil":    {"p16": float(T_MS_p16), "p50": float(T_MS_p50), "p84": float(T_MS_p84)},
        "R_MS_dil":    {"p16": float(R_MS_p16), "p50": float(R_MS_p50), "p84": float(R_MS_p84)},
        "logg_WD_dil": {"p16": logg_WD_p16,     "p50": logg_WD_med,     "p84": logg_WD_p84},
        "wd_atmo_type_dil":  atmo,
        "depth_summary_dil": depth_summary_dil,
        "depth_model_dil":   depth.describe(),
        "depth_noise_dil":   "observed depth used in the likelihood, not perturbed",
        "N_MC_dil":          N_MC,
        "N_valid_dil":       n_valid,
        "filters_dil":       active_filters,
    }
    params_path = out_dir / "params_dil.json"
    with open(params_path, "w", encoding="utf-8") as fp:
        json.dump(params_dil, fp, indent=2)
    print(f"[DIL] Saved params → {params_path}")

    # ── Step 8: Plot ───────────────────────────────────────────────────────
    plot_path = out_dir / "plot_dil.pdf"
    _plot_dil(
        T_MS_samples = T_MS_samples[valid_mc],
        R_MS_samples = R_MS_samples[valid_mc],
        T_MS_p16=T_MS_p16, T_MS_p50=T_MS_p50, T_MS_p84=T_MS_p84,
        R_MS_p16=R_MS_p16, R_MS_p50=R_MS_p50, R_MS_p84=R_MS_p84,
        T2_grid=T2_grid,
        R2_p16_grid=R2_p16, R2_p50_grid=R2_p50, R2_p84_grid=R2_p84,
        bhac_delta_t=bhac_local_delta_t,
        out_path=plot_path,
    )

    # ── Return ─────────────────────────────────────────────────────────────
    return {
        # Medians (p50)
        "T_MS_dil":        float(T_MS_p50),
        "R_MS_dil":        float(R_MS_p50),
        "logg_WD_dil":     logg_WD_med,
        # Full intervals
        "T_MS_p16_dil":    float(T_MS_p16),
        "T_MS_p84_dil":    float(T_MS_p84),
        "R_MS_p16_dil":    float(R_MS_p16),
        "R_MS_p84_dil":    float(R_MS_p84),
        "logg_WD_p16_dil": logg_WD_p16,
        "logg_WD_p84_dil": logg_WD_p84,
        # MC samples (numpy arrays)
        "T_MS_samples_dil": T_MS_samples[valid_mc],
        "R_MS_samples_dil": R_MS_samples[valid_mc],
        # File paths
        "posterior_samples_dil_path":         str(path_samp),
        "posterior_samples_dil_matched_path": str(path_matched),
        "lnL_dil_path":               str(path_lnL),
        "params_dil_path":            str(params_path),
        "plot_dil_path":              str(plot_path),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plot helper
# ══════════════════════════════════════════════════════════════════════════════

def _plot_dil(T_MS_samples, R_MS_samples,
              T_MS_p16, T_MS_p50, T_MS_p84,
              R_MS_p16, R_MS_p50, R_MS_p84,
              T2_grid, R2_p16_grid, R2_p50_grid, R2_p84_grid,
              bhac_delta_t, out_path):

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)

    # — Panel 1: T_MS posterior —
    bins_T = np.arange(T2_grid.min() - 50, T2_grid.max() + 150, 100)
    axes[0].hist(T_MS_samples, bins=bins_T, alpha=0.75, color="steelblue")
    axes[0].axvline(T_MS_p16, color="k",   ls="--", lw=1.2,
                    label=f"p16 = {T_MS_p16:.0f} K")
    axes[0].axvline(T_MS_p50, color="red", ls="-",  lw=1.5,
                    label=f"p50 = {T_MS_p50:.0f} K")
    axes[0].axvline(T_MS_p84, color="k",   ls="--", lw=1.2,
                    label=f"p84 = {T_MS_p84:.0f} K")
    axes[0].set_xlabel("T_MS [K]");  axes[0].set_ylabel("counts")
    axes[0].legend(frameon=False, fontsize=8); axes[0].grid(alpha=0.3)

    # — Panel 2: R_MS posterior —
    axes[1].hist(R_MS_samples, bins=50, alpha=0.75, color="darkorange")
    axes[1].axvline(R_MS_p16, color="k",   ls="--", lw=1.2,
                    label=f"p16 = {R_MS_p16:.3f}")
    axes[1].axvline(R_MS_p50, color="red", ls="-",  lw=1.5,
                    label=f"p50 = {R_MS_p50:.3f}")
    axes[1].axvline(R_MS_p84, color="k",   ls="--", lw=1.2,
                    label=f"p84 = {R_MS_p84:.3f}")
    axes[1].set_xlabel(r"R_MS [$R_\odot$]"); axes[1].set_ylabel("counts")
    axes[1].legend(frameon=False, fontsize=8); axes[1].grid(alpha=0.3)

    # — Panel 3: BHAC15 R_MS(T_MS) relation —
    axes[2].fill_between(T2_grid, R2_p16_grid, R2_p84_grid,
                         color="steelblue", alpha=0.25,
                         label=f"BHAC15 ±1σ (|ΔT|≤{bhac_delta_t:.0f} K)")
    axes[2].plot(T2_grid, R2_p50_grid, "o-", ms=3, color="steelblue",
                 label="BHAC15 R_MS(T_MS)")
    axes[2].axvline(T_MS_p50, color="red", ls="-",  lw=1.2,
                    label=f"T_MS = {T_MS_p50:.0f} K")
    axes[2].axhline(R_MS_p50, color="red", ls="--", lw=1.2,
                    label=f"R_MS = {R_MS_p50:.3f} R☉")
    axes[2].set_xlabel("T_MS [K]"); axes[2].set_ylabel(r"R_MS [$R_\odot$]")
    axes[2].legend(frameon=False, fontsize=8); axes[2].grid(alpha=0.3)

    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[DIL] Saved plot → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Model-free eclipse-radius diagnostic in the (T_MS, R_MS) plane
# ══════════════════════════════════════════════════════════════════════════════
#
# The dilution diagnostic above answers "where does the depth put the companion
# along the BHAC15 relation", so the relation is part of the answer and cannot
# be tested by it. The diagnostic below answers "what radius does the depth
# demand at each temperature", which needs no radius-temperature relation at
# all, and can therefore be laid over the SED posterior as an independent
# constraint. What is left of the models is the pair of band fluxes, taken
# from the shared depth model (SED companion model and gravity prescription);
# the logg scan quantifies the residual gravity dependence.

_DEPTH_CLIP = (1e-6, 1.0 - 1e-6)

# Monte Carlo draws (n_mc) and T_WD / R_WD axes of the tabulated likelihood
# (n_grid), per sampling mode. dilution_uniform_fit.py uses the same n_mc.
# Defaults when this module is used alone; main_all.py sets its own values
# (see its "Fit settings" section, function apply_fit_settings).
DILUTION_SAMPLING = {
    "test":  dict(n_mc=5_000,  n_grid=10),
    "final": dict(n_mc=50_000, n_grid=30),
}
_REF = "prescription"   # key of the reference band (shared gravity prescription)

_FILTER_COLORS = {"iz": "#d62728", "i": "#ff7f0e",
                  "z": "#8c564b", "exo": "#2ca02c"}
_FALLBACK_COLORS = ["#d62728", "#2ca02c", "#8c564b", "#9467bd", "#17becf"]
_LOGG_LINESTYLES = ["--", ":", "-.", (0, (3, 1, 1, 1))]


def run_eclipse_radius_diagnostic(
    gaia_id,
    tic_id,
    output_base,
    depth_by_filter,
    filter_profile_files,
    posterior_samples_sed_path,
    koester_dir,
    da_track_path,
    db_track_path,
    wd_atmo_type="DA",
    t_ms_min=T_MS_MIN_K,
    t_ms_max=T_MS_MAX_K,
    t_ms_step=25.0,
    ms_logg_values=(4.5, 5.0, 5.5),
    photon_counting=True,
    filter_wavelength_cuts=None,
    posterior_samples_joint_path=None,
    bhac15_csv=None,
    bhac_age_logt_min=8.0,
    bhac_age_logt_max=None,
    bhac_local_delta_t=100.0,
    bhac_local_min_points=3,
    n_draws=4000,
    seed=42,
    ms_atmosphere="phoenix",
    atmo_cache_dir=None,
    atmo_grid_ranges=None,
    ms_logg_mode=ATMO_MS_LOGG_MODE,
    ms_logg_fixed=ATMO_MS_LOGG_FIXED,
    wd_logg_mode=ATMO_WD_LOGG_MODE,
    wd_logg_fixed=ATMO_WD_LOGG_FIXED,
):
    """Compare what the eclipse demands with what the SED demands, in (T_MS, R_MS).

    For every draw of (T_WD, R_WD) from the SED posterior and every measured
    depth, the required companion radius is evaluated along a T_MS axis with
    the shared depth model (DepthFluxModel.required_radius). The spread over
    draws gives a band, drawn on top of the SED posterior in the same plane:
    the figure shows the two independent constraints and their intersection,
    with no BHAC15 R(T) anywhere in the construction.

    Parameters
    ----------
    gaia_id, tic_id : int
    output_base : str or Path
    depth_by_filter : dict
        {"filter_name": (depth, err_sym)} — measured primary-eclipse depths.
    filter_profile_files : dict
        {"filter_name": Path} — ASCII transmission curves.
    posterior_samples_sed_path : str or Path
        .npy [N, >=4] from sed_fit.run_sed_fit (T_WD, R_WD, T_MS, R_MS, ...).
    koester_dir, da_track_path, db_track_path : str or Path
    wd_atmo_type : {"DA", "DB"}   selects the cooling track only
    t_ms_min, t_ms_max, t_ms_step : float
        Temperature axis of the required-radius curves [K]; widened to the
        SED posterior coverage within the companion grid (display axis).
    ms_logg_values : sequence of float
        Fixed companion gravities at which the band is recomputed, to show the
        residual gravity dependence against the reference band, which uses
        the shared prescription (ms_logg_mode). Values outside the companion
        grid give no curve (no extrapolation).
    posterior_samples_joint_path : str or Path or None
        Optional joint posterior, overlaid in black when given.
    bhac15_csv : str or Path or None
        Optional. Only drawn as a faint reference curve to show what the
        relation would have imposed; nothing in the diagnostic uses it.
    n_draws : int
        Number of SED posterior draws folded through the depth inversion.
    seed : int
    ms_atmosphere, atmo_cache_dir, atmo_grid_ranges, ms_logg_mode, ms_logg_fixed,
    wd_logg_mode, wd_logg_fixed
        Shared depth-model settings (the SED settings).

    Returns
    -------
    dict with _ecl-suffixed keys
    """
    gaia_id        = int(gaia_id)
    tic_id         = int(tic_id)
    active_filters = list(depth_by_filter.keys())
    atmo           = str(wd_atmo_type).strip().upper()
    logg_values    = sorted({float(g) for g in ms_logg_values})

    out_dir = Path(output_base) / f"TIC{tic_id}_Gaia{gaia_id}" / "eclipse_radius"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    print(f"\n{'='*60}")
    print(f"[ECL] Gaia DR3 {gaia_id}  /  TIC {tic_id}")
    print(f"[ECL] Model-free eclipse radius R_MS(T_MS) vs SED posterior")
    print(f"[ECL] Filters : {active_filters}   logg_MS scan : {logg_values}")
    print(f"{'='*60}")

    # ── Step 1: SED posterior ─────────────────────────────────────────────
    samp = np.load(posterior_samples_sed_path)
    if samp.ndim != 2 or samp.shape[1] < 4:
        raise ValueError(f"posterior_samples_sed must be [N, >=4], got {samp.shape}")

    T_WD_post, R_WD_post = samp[:, 0].astype(float), samp[:, 1].astype(float)
    T_MS_sed,  R_MS_sed  = samp[:, 2].astype(float), samp[:, 3].astype(float)
    ok_post = (np.isfinite(T_WD_post) & np.isfinite(R_WD_post) & (R_WD_post > 0)
               & np.isfinite(T_MS_sed) & np.isfinite(R_MS_sed) & (R_MS_sed > 0))
    T_WD_post, R_WD_post = T_WD_post[ok_post], R_WD_post[ok_post]
    T_MS_sed,  R_MS_sed  = T_MS_sed[ok_post],  R_MS_sed[ok_post]
    print(f"[ECL] SED posterior samples: {ok_post.sum()} valid")

    t_sed_lo, t_sed_hi = np.percentile(T_MS_sed, [1, 99])
    t_sed_16, t_sed_50, t_sed_84 = np.percentile(T_MS_sed, [16, 50, 84])

    idx = rng.integers(0, T_WD_post.size, size=int(n_draws))

    # ── Step 2: shared depth model ────────────────────────────────────────
    profiles = _load_profiles(active_filters, filter_profile_files, filter_wavelength_cuts)
    depth = DepthFluxModel(
        profiles, koester_dir,
        wd_cooling_track=da_track_path if atmo == "DA" else db_track_path,
        ms_atmosphere=ms_atmosphere, atmo_cache_dir=atmo_cache_dir,
        atmo_grid_ranges=atmo_grid_ranges, ms_logg_mode=ms_logg_mode,
        ms_logg_fixed=ms_logg_fixed, wd_logg_mode=wd_logg_mode,
        wd_logg_fixed=wd_logg_fixed, photon_counting=photon_counting)

    # ── Step 3: T_MS axis (display axis, within the companion grid) ───────
    ms_lo, ms_hi = depth.ms_teff_bounds
    t_axis_lo = max(min(float(t_ms_min), float(t_sed_lo)), float(ms_lo))
    t_axis_hi = min(max(float(t_ms_max), float(t_sed_hi)), float(ms_hi))
    T_grid = np.arange(t_axis_lo, t_axis_hi + float(t_ms_step),
                       float(t_ms_step), dtype=float)
    T_grid = T_grid[T_grid <= t_axis_hi]
    if T_grid.size < 5:
        raise RuntimeError(f"[ECL] Empty T_MS axis: [{t_axis_lo}, {t_axis_hi}]")
    print(f"[ECL] T_MS axis: [{T_grid.min():.0f}, {T_grid.max():.0f}] K "
          f"(SED covers [{t_sed_lo:.0f}, {t_sed_hi:.0f}] K)")
    scan_ok = [g for g in logg_values
               if depth.ms_logg_bounds[0] <= g <= depth.ms_logg_bounds[1]]
    for g in sorted(set(logg_values) - set(scan_ok)):
        print(f"[ECL] logg_MS {g:.2f} is outside the companion grid "
              f"{depth.ms_logg_bounds}: no curve (no extrapolation)")

    # ── Step 4: white-dwarf side of every draw ────────────────────────────
    T_WD_draw = T_WD_post[idx]
    R_WD_draw = R_WD_post[idx]
    logg_draw = depth.wd_logg(T_WD_draw, R_WD_draw)
    F_WD_draw = {fk: depth.wd_band_flux(fk, T_WD_draw, logg_draw) for fk in active_filters}

    # Draws outside the Koester temperature range have no flux and are
    # dropped, exactly as the joint fit rejects them.
    inside = np.ones(T_WD_draw.size, bool)
    for fk in active_filters:
        inside &= np.isfinite(F_WD_draw[fk]) & (F_WD_draw[fk] > 0)
    T_WD_draw, R_WD_draw, logg_draw = (a[inside] for a in (T_WD_draw, R_WD_draw, logg_draw))
    F_WD_draw = {fk: v[inside] for fk, v in F_WD_draw.items()}
    idx_kept = idx[inside]
    n_keep   = int(inside.sum())
    if n_keep < 50:
        raise RuntimeError(f"[ECL] Only {n_keep} draws inside the Koester grid")
    print(f"[ECL] Draws used: {n_keep}/{int(n_draws)} "
          f"({int(n_draws) - n_keep} outside the Koester grid)")

    # The measured depth carries its own error bar, so each draw gets its own
    # depth; the band therefore mixes the white-dwarf and the depth uncertainty.
    depth_draw = {
        fk: np.clip(rng.normal(float(d_obs), max(float(d_err), 1e-9), n_keep),
                    *_DEPTH_CLIP)
        for fk, (d_obs, d_err) in depth_by_filter.items()
    }

    # ── Step 5: required radius, band and residual logg dependence ────────
    bands = {_REF: {}}   # bands[key][fk] = {"p16","p50","p84"} over the T_MS axis
    T2d = np.broadcast_to(T_grid[None, :], (n_keep, T_grid.size))
    for fk in active_filters:
        R_req = depth.required_radius(fk, depth_draw[fk][:, None], R_WD_draw[:, None],
                                      F_WD_draw[fk][:, None], T2d)
        with np.errstate(invalid="ignore"):
            p16, p50, p84 = np.nanpercentile(R_req, [16, 50, 84], axis=0)
        bands[_REF][fk] = {"p16": p16, "p50": p50, "p84": p84}
    for g in scan_ok:
        bands[g] = {}
        for fk in active_filters:
            F_MS = depth.ms_band_flux(fk, T_grid, np.full(T_grid.shape, g))
            R_req = eclipse_required_radius(depth_draw[fk][:, None], R_WD_draw[:, None],
                                            F_WD_draw[fk][:, None], F_MS[None, :])
            with np.errstate(invalid="ignore"):
                p16, p50, p84 = np.nanpercentile(R_req, [16, 50, 84], axis=0)
            bands[g][fk] = {"p16": p16, "p50": p50, "p84": p84}

    in_sed  = (T_grid >= t_sed_lo) & (T_grid <= t_sed_hi)
    in_1sig = (T_grid >= t_sed_16) & (T_grid <= t_sed_84)

    logg_sensitivity = {}
    for fk in active_filters:
        ref = bands[_REF][fk]
        half_width = 0.5 * (ref["p84"] - ref["p16"])
        per_g = {}
        for g in scan_ok:
            with np.errstate(invalid="ignore", divide="ignore"):
                frac  = bands[g][fk]["p50"] / ref["p50"] - 1.0
                shift = np.abs(bands[g][fk]["p50"] - ref["p50"]) / half_width
            m = in_1sig & np.isfinite(frac)
            if not m.any():
                m = in_sed & np.isfinite(frac)
            if not m.any():
                continue
            per_g[f"{g:.2f}"] = {
                "frac_shift_at_T_MS_p50": float(np.interp(t_sed_50, T_grid, frac)),
                "max_abs_frac_shift": float(np.nanmax(np.abs(frac[m]))),
                "max_shift_over_sigma": float(np.nanmax(shift[m & np.isfinite(shift)])),
            }
        logg_sensitivity[fk] = per_g

    # ── Step 6: sample-by-sample consistency of the two constraints ───────
    # Each SED sample is a complete solution, so its own T_MS can be fed to the
    # inversion and the radius it predicts compared with the radius the SED
    # fitted. A ratio distribution straddling 1 means the eclipse depth and the
    # photometry ask for the same star.
    consistency = {}
    for fk in active_filters:
        R_req_sample = depth.required_radius(
            fk, depth_draw[fk], R_WD_draw, F_WD_draw[fk], T_MS_sed[idx_kept])
        ratio = R_MS_sed[idx_kept] / R_req_sample
        ratio = ratio[np.isfinite(ratio)]
        if ratio.size < 20:
            continue
        r16, r50, r84 = np.percentile(ratio, [16, 50, 84])
        if r16 <= 1.0 <= r84:
            verdict = "consistent within 1 sigma"
        elif r16 > 1.0:
            verdict = "SED radius larger than the eclipse requires"
        else:
            verdict = "SED radius smaller than the eclipse requires"
        consistency[fk] = {"p16": float(r16), "p50": float(r50),
                           "p84": float(r84), "verdict": verdict,
                           "n_samples": int(ratio.size)}
        print(f"[ECL] R_MS(SED) / R_MS(eclipse) [{fk}] = {r50:.3f} "
              f"(-{r50-r16:.3f}/+{r84-r50:.3f}) — {verdict}")

    print(f"[ECL] T_MS from the SED: {t_sed_50:.0f} K "
          f"[{t_sed_16:.0f}, {t_sed_84:.0f}] — logg scan quoted over that range")
    for fk, per_g in logg_sensitivity.items():
        for g, entry in per_g.items():
            print(f"[ECL] logg_MS {g} vs prescription [{fk}]: required radius "
                  f"moves by {100*entry['frac_shift_at_T_MS_p50']:+.1f} per cent "
                  f"(max {100*entry['max_abs_frac_shift']:.1f}), i.e. "
                  f"{entry['max_shift_over_sigma']:.2f} sigma of the band")

    # ── Step 7: optional reference curves, used by nothing above ──────────
    bhac_curve = None
    if bhac15_csv is not None:
        # Tabulated on its own coarse axis: the local window is 2*delta_T wide,
        # so sampling it every 25 K would only draw a staircase.
        df_bh   = _load_bhac15(bhac15_csv, bhac_age_logt_min, bhac_age_logt_max)
        T_bhac  = np.arange(float(T_grid.min()), float(T_grid.max()) + 100.0, 100.0)
        stats   = np.array([_bhac15_r_stats(T, df_bh, bhac_local_delta_t,
                                            bhac_local_min_points)
                            for T in T_bhac], dtype=float)
        good = np.isfinite(stats[:, 1])
        if good.any():
            bhac_curve = {"T": T_bhac[good], "p16": stats[good, 0],
                          "p50": stats[good, 1], "p84": stats[good, 2]}

    joint_TR = None
    if posterior_samples_joint_path is not None:
        sj = np.load(posterior_samples_joint_path)
        if sj.ndim == 2 and sj.shape[1] >= 4:
            joint_TR = (sj[:, 2].astype(float), sj[:, 3].astype(float))

    # ── Step 8: save curves, summary and figure ───────────────────────────
    def _key(g):
        return _REF if g == _REF else f"logg{g:.2f}"

    path_curves = out_dir / "eclipse_radius_curves.npz"
    np.savez(
        path_curves,
        T_MS_grid_K = T_grid,
        logg_values = np.array(scan_ok, dtype=float),
        filters     = np.array(active_filters),
        description = np.array(
            "R_MS_req[key][filter][p16|p50|p84](T_MS) = R_WD sqrt((1/depth - 1) "
            "F_WD / F_MS(T_MS)); key 'prescription' uses the shared companion "
            "gravity prescription, 'loggX.XX' a fixed gravity; no "
            "radius-temperature relation used"),
        **{f"R_req_{fk}_{_key(g)}_{q}": bands[g][fk][q]
           for g in bands for fk in active_filters
           for q in ("p16", "p50", "p84")},
    )
    print(f"[ECL] Saved curves → {path_curves}")

    params_ecl = {
        "gaia_id": gaia_id,
        "tic_id":  tic_id,
        "method":  "model_free_eclipse_radius",
        "filters_ecl": active_filters,
        "wd_atmo_type_ecl": atmo,
        "n_draws_ecl": int(n_draws),
        "n_draws_used_ecl": n_keep,
        "T_MS_grid_ecl": {"min": float(T_grid.min()), "max": float(T_grid.max()),
                          "step": float(t_ms_step)},
        "T_MS_sed_range_ecl": [float(t_sed_lo), float(t_sed_hi)],
        "T_MS_sed_1sigma_ecl": [float(t_sed_16), float(t_sed_84)],
        "T_MS_sed_p50_ecl": float(t_sed_50),
        "ms_logg_reference_ecl": f"prescription: {depth.ms_logg_mode}",
        "ms_logg_values_ecl": [float(g) for g in scan_ok],
        "depths_ecl": {fk: {"obs": float(depth_by_filter[fk][0]),
                            "err_sym": float(depth_by_filter[fk][1])}
                       for fk in active_filters},
        "consistency_ecl": consistency,
        "logg_sensitivity_ecl": logg_sensitivity,
        "logg_sensitivity_quoted_over_ecl": "T_MS interval T_MS_sed_1sigma_ecl",
        "relations_used_ecl": {
            "R_MS_of_T_MS": "none — radius inverted from the depth",
            "logg_MS": f"shared prescription '{depth.ms_logg_mode}' "
                       f"(scanned at fixed {scan_ok})",
            "logg_WD": depth.wd_logg_policy,
            "band_fluxes": "Koester DA and the SED companion model, integrated over "
                           "the filters, bilinear in ln F",
        },
        "depth_model_ecl": depth.describe(),
    }
    path_params = out_dir / "params_eclipse_radius.json"
    with open(path_params, "w", encoding="utf-8") as fp:
        json.dump(params_ecl, fp, indent=2)
    print(f"[ECL] Saved params → {path_params}")

    path_plot = out_dir / "plot_eclipse_radius.pdf"
    _plot_eclipse_radius(
        T_grid=T_grid, bands=bands, g_ref=_REF, logg_values=scan_ok,
        active_filters=active_filters, depth_by_filter=depth_by_filter,
        T_MS_sed=T_MS_sed, R_MS_sed=R_MS_sed, in_sed=in_sed,
        t_sed_1sigma=(t_sed_16, t_sed_84),
        consistency=consistency, bhac_curve=bhac_curve, joint_TR=joint_TR,
        label=f"Gaia {gaia_id} / TIC {tic_id}", out_path=path_plot, rng=rng,
    )

    out = {
        "consistency_ecl":          consistency,
        "logg_sensitivity_ecl":     logg_sensitivity,
        "T_MS_grid_ecl":            T_grid,
        "R_req_bands_ecl":          bands,
        "ms_logg_reference_ecl":    _REF,
        "n_draws_used_ecl":         n_keep,
        "curves_ecl_path":          str(path_curves),
        "params_ecl_path":          str(path_params),
        "plot_ecl_path":            str(path_plot),
    }
    return out


def _plot_eclipse_radius(T_grid, bands, g_ref, logg_values, active_filters,
                         depth_by_filter, T_MS_sed, R_MS_sed, in_sed,
                         t_sed_1sigma, consistency, bhac_curve, joint_TR,
                         label, out_path, rng):
    """Three panels: the two constraints, the logg scan, and their ratio.

    bands[g_ref] is the reference band (shared gravity prescription); the
    other keys of `bands` are the fixed-gravity scan values in logg_values.
    """
    try:
        from display import _kde_contour
    except Exception:                                   # pragma: no cover
        _kde_contour = None

    fig, axd = plt.subplot_mosaic(
        [["main", "sens"], ["main", "ratio"]],
        figsize=(13.5, 6.6), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )

    colors = {fk: _FILTER_COLORS.get(fk, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])
              for i, fk in enumerate(active_filters)}

    # ── Panel 1: what the SED asks for, and what the eclipse asks for ─────
    ax = axd["main"]
    if bhac_curve is not None:
        ax.plot(bhac_curve["T"], bhac_curve["p50"], color="0.6", ls="-", lw=1.2,
                zorder=2, label="BHAC15 R(T) — shown for reference, not used")

    if _kde_contour is not None:
        _kde_contour(ax, T_MS_sed, R_MS_sed, color="tab:blue", rng=rng,
                     scatter_zorder=1, contour_zorder=6)
    else:
        ax.scatter(T_MS_sed, R_MS_sed, s=3, alpha=0.08, color="tab:blue",
                   rasterized=True, zorder=1)

    if joint_TR is not None:
        if _kde_contour is not None:
            _kde_contour(ax, joint_TR[0], joint_TR[1], color="0.55", rng=rng,
                         contour_color="black", linewidth=1.6,
                         scatter_zorder=3, contour_zorder=12)
        else:
            ax.scatter(joint_TR[0], joint_TR[1], s=3, alpha=0.1, color="0.4",
                       rasterized=True, zorder=3)

    for fk in active_filters:
        c   = colors[fk]
        ref = bands[g_ref][fk]
        d_obs, d_err = depth_by_filter[fk]
        ax.fill_between(T_grid, ref["p16"], ref["p84"], color=c, alpha=0.28,
                        lw=0, zorder=4)
        ax.plot(T_grid, ref["p50"], color=c, lw=2.0, zorder=5,
                label=f"eclipse {fk}: depth = {d_obs:.3f} ± {d_err:.3f}")
        for k, g in enumerate([g for g in logg_values if g != g_ref]):
            ax.plot(T_grid, bands[g][fk]["p50"], color=c, lw=1.0, zorder=5,
                    ls=_LOGG_LINESTYLES[k % len(_LOGG_LINESTYLES)],
                    label=fr"      same, $\log g_{{\rm MS}}$ = {g:.1f}")

    ax.set_xlabel(r"$T_{\rm MS}$ [K]", fontsize=13)
    ax.set_ylabel(r"$R_{\rm MS}$ [$R_\odot$]", fontsize=13)
    ax.set_xlim(T_grid.min(), T_grid.max())

    # Frame the crossing rather than the full extent of the curves, which run
    # off to small radii at temperatures the SED has already excluded.
    lo_parts = [bands[g_ref][fk]["p16"][in_sed] for fk in active_filters]
    hi_parts = [bands[g_ref][fk]["p84"][in_sed] for fk in active_filters]
    lo_parts.append(np.nanpercentile(R_MS_sed, 0.5))
    hi_parts.append(np.nanpercentile(R_MS_sed, 99.5))
    lo = np.nanmin(np.concatenate([np.atleast_1d(a) for a in lo_parts]))
    hi = np.nanmax(np.concatenate([np.atleast_1d(a) for a in hi_parts]))
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        pad = 0.12 * (hi - lo)
        ax.set_ylim(max(0.0, lo - pad), hi + pad)
    ax.grid(alpha=0.25)
    ax.set_title(f"{label} — what the eclipse demands vs what the SED demands",
                 fontsize=11)

    handles = [Line2D([0], [0], color="tab:blue", lw=1.6, label="SED posterior")]
    if joint_TR is not None:
        handles.append(Line2D([0], [0], color="black", lw=1.6,
                              label="joint posterior"))
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + h, fontsize=8.5, loc="upper right",
              frameon=True, framealpha=0.9)

    # ── Panel 2: how much the companion gravity moves that curve ──────────
    ax = axd["sens"]
    for fk in active_filters:
        ref = bands[g_ref][fk]["p50"]
        for k, g in enumerate([g for g in logg_values if g != g_ref]):
            with np.errstate(invalid="ignore", divide="ignore"):
                frac = 100.0 * (bands[g][fk]["p50"] / ref - 1.0)
            ax.plot(T_grid, frac, color=colors[fk], lw=1.3,
                    ls=_LOGG_LINESTYLES[k % len(_LOGG_LINESTYLES)],
                    label=fr"{fk}, $\log g$ = {g:.1f}")
    ax.axvspan(float(t_sed_1sigma[0]), float(t_sed_1sigma[1]),
               color="tab:blue", alpha=0.16, lw=0,
               label=r"$T_{\rm MS}$, SED $1\sigma$")
    if in_sed.any():
        ax.axvspan(T_grid[in_sed].min(), T_grid[in_sed].max(),
                   color="tab:blue", alpha=0.07, lw=0, zorder=0)
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_xlim(T_grid.min(), T_grid.max())
    ax.set_xlabel(r"$T_{\rm MS}$ [K]", fontsize=11)
    ax.set_ylabel(r"$\Delta R_{\rm MS}$ vs gravity prescription [%]",
                  fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, ncol=2)

    # ── Panel 3: the two constraints reduced to one number ────────────────
    ax = axd["ratio"]
    if consistency:
        for fk in active_filters:
            entry = consistency.get(fk)
            if entry is None:
                continue
            ax.errorbar(entry["p50"], fk, fmt="o", color=colors[fk], ms=7,
                        xerr=[[entry["p50"] - entry["p16"]],
                              [entry["p84"] - entry["p50"]]],
                        capsize=4, lw=1.8)
        ax.axvline(1.0, color="k", ls="--", lw=1.2)
        ax.set_xlabel(r"$R_{\rm MS}$(SED) / $R_{\rm MS}$(eclipse)", fontsize=11)
        ax.margins(y=0.45)
        ax.grid(alpha=0.25, axis="x")
    else:
        ax.text(0.5, 0.5, "no usable consistency sample", ha="center",
                va="center", transform=ax.transAxes, color="grey")

    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[ECL] Saved plot → {out_path}")
