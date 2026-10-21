#!/usr/bin/env python3
"""joint_fit_ideal.py — Direct joint SED + eclipse-depth fit for WD+MS systems.

Purpose
-------
This module implements the "ideal" (direct) joint fit. A single UltraNest run
explores the shared physical parameter space and evaluates, at every sample,
the raw VizieR SED photometry, the measured eclipse depths, and the Gaia
parallax. Nothing is pre-summarised: no KDE of the SED posterior, no tabulated
dilution likelihood, no sampling bounds derived from posterior percentiles.

Joint posterior
---------------
    ln p(theta | SED, depths, parallax)  =  ln L_SED(theta)
                                          + ln L_dilution(theta)
                                          + ln L_Gaia_parallax(theta)
                                          + ln pi_physical(theta)   [+ const]

Shared parameters (sampled once, used everywhere)
-------------------------------------------------
    1. T_WD     [K]        white-dwarf effective temperature
    2. R_WD     [R_sun]    white-dwarf radius
    3. T_MS     [K]        companion effective temperature
    4. R_MS     [R_sun]    companion radius
    5. plx_mas  [mas]      parallax; the distance is derived as d = 1000/plx
    6. A_V      [mag]      V-band extinction

The very same T_WD, R_WD, T_MS and R_MS enter both the SED forward model and
the eclipse-depth forward model. No parameter is drawn twice, and no quantity
is replaced by a relation-based substitute inside one of the two models.

Conditional independence
------------------------
The SED photometry and the eclipse-depth measurements are independent datasets
obtained with different instruments and different observing strategies. Given
the shared physical parameters they are treated as conditionally independent,
so their log-likelihoods simply add. Each dataset therefore contributes its
information exactly once.

Physical priors
---------------
    T_WD    ~ Uniform(T_WD_MIN, T_WD_MAX)                   [sed_fit.py limits]
    R_WD    ~ LogUniform(R_WD_MIN_RSUN, R_WD_MAX_RSUN)      [sed_fit.py limits]
    T_MS    ~ Uniform over the interval simultaneously covered by the requested
              T_MS limits (2300–4000 K by default), the BHAC15 relation, the
              depth-term companion model and the SED companion grid
    R_MS    ~ Uniform(R_MS_MIN_RSUN, R_MS_MAX_RSUN)         [sed_fit.py limits]
              with the default ms_radius_prior="uniform"; see below
    plx_mas ~ Uniform over the positive +/- N_SIGMA_PLX interval used by
              sed_fit.py (the Gaia measurement itself enters the likelihood)
    A_V     ~ Uniform(AV_MIN, AV_MAX) in the prior transform, weighted in the
              log-likelihood by the same Gaussian as sed_fit.py 2-comp:
              N(A_V_1comp, sigma_1comp) with
              sigma_1comp = max(AV_2C_SIGMA_MIN, 0.5*(p84-p16) of the 1-comp A_V)

Companion radius prior
----------------------
ms_radius_prior decides what the (T_MS, R_MS) plane is allowed to look like
before the data speak.

  "uniform"   R_MS ~ Uniform(R_MS_MIN_RSUN, R_MS_MAX_RSUN), exactly the prior
              sed_fit.py uses. T_MS and R_MS are then independent a priori and
              no stellar-evolution relation enters the sampled parameters at
              all: the SED and the eclipse depths alone decide where the
              companion sits in the temperature-radius plane, so a radius that
              disagrees with the evolutionary models is a measurement rather
              than something the prior had already ruled out. A mass-radius
              relation can of course still be applied downstream to convert the
              measured radius into a mass (mass.py); it simply no longer shapes
              the posterior. This is the default.

  "bhac15"    R_MS | T_MS ~ Normal(R50_BHAC15(T_MS), sigma_R) truncated to
              [R_MS_MIN_RSUN, R_MS_MAX_RSUN], with
              sigma_R = 0.5 * (R84_BHAC15(T_MS) - R16_BHAC15(T_MS)).

  "brown2022" the same conditional prior, with the BHAC15 radii rescaled to the
              semi-empirical relation of Brown et al. 2022 and widened by its
              intrinsic scatter. Requires the ms_relation argument.

Both conditional options are applied through the UltraNest prior transform,
using a truncated-normal inverse CDF whose centre and width are evaluated at
the already transformed T_MS coordinate. The relation is therefore built into
the prior measure itself, no relation term appears in the log-likelihood, and
the radius relation is applied exactly once.

Direct SED likelihood
---------------------
Two-component model, identical to the 2-component model of sed_fit.py:

    F_model(lambda) = [ (R_WD/d)^2 * F_WD(lambda, T_WD)
                      + (R_MS/d)^2 * F_MS(lambda, T_MS) ] * extinction(lambda, A_V)

The white dwarf follows the wd_atmosphere argument (blackbody or Koester DA)
and the companion the ms_atmosphere one (blackbody, PHOENIX or BT-Settl), so
the systematics measured by sed_fit.py apply unchanged here. The eclipse-depth
term uses the same companion model and gravity prescription (see below); its
white dwarf is always a Koester DA model.

with d_pc = 1000 / plx_mas and the CCM89 + O'Donnell extinction law of
sed_fit.py. The photometry is prepared with the same rules as sed_fit.py
(cached VizieR table, positive finite fluxes, catalogue/filter deduplication,
wavelength interval, recovery of a missing sigma from the source catalogue,
rejection of points without sigma or with sigma/F above sed_max_rel_err,
sigma floored at FRAC_FLOOR * F as in sed_fit.py, ABS_FLOOR as a numerical
guard). The Gaussian likelihood includes its normalisation term. Outlier
clipping is never applied here. main_all.py passes the same sed_max_rel_err
(0.25) to this function and to sed_fit.run_sed_fit.

Direct dilution likelihood
--------------------------
For every sample the eclipse depths are predicted with the shared depth model
(depth_model.DepthFluxModel):

    logg_WD  SED policy: Bedard 2020 cooling track (Delaunay) at (T_WD, R_WD),
             fallback wd_logg_fixed outside it, clipped to the Koester range
    F_WD,b   Koester DA band flux, bilinear in ln F over (T_WD, logg_WD)
    logg_MS  SED prescription (ms_logg_mode), clipped to the companion grid
    F_MS,b   the SED companion model (STScI PHOENIX by default), bilinear in
             ln F over (T_MS, logg_MS), unreddened

logg_WD and logg_MS are computed once per sample and passed to both the SED
term and the depth term. BHAC15 is still read in every case, including
ms_radius_prior="uniform": it sets the T_MS coverage and the conditional
radius prior, but no longer the companion gravity.

    q_b            = (R_MS / R_WD)^2 * F_MS,b / F_WD,b
    depth_model,b  = 1 / (1 + q_b)

    ln L_dilution  = sum_b [ -0.5 * ((depth_obs,b - depth_model,b)/sigma_b)^2
                             - ln(sigma_b * sqrt(2*pi)) ]

The observed depths are used as measured; they are never perturbed with Monte
Carlo noise, T_WD and R_WD are never drawn from the SED posterior, and the
sampled R_MS is used directly instead of the BHAC15 median radius.

Gaia parallax likelihood
------------------------
    ln L_parallax = -0.5 * ((plx_mas - plx_obs)/sigma_plx)^2
                    - ln(sigma_plx * sqrt(2*pi))

applied exactly once; the prior transform uses a plain uniform interval so the
Gaia measurement is not counted twice.

Physical assumptions
--------------------
  * the white dwarf is completely occulted during the measured primary eclipse;
  * the eclipse depth is entirely described by flux dilution;
  * there is no unmodelled third light in the photometric aperture;
  * the eclipse-depth uncertainties are Gaussian;
  * the SED and eclipse-depth measurements are conditionally independent given
    the shared physical parameters;
  * the same physical parameters describe both datasets.

Distinction from the legacy method (joint_fit.py)
-------------------------------------------------
joint_fit.py multiplies a gaussian_kde representation of the SED posterior by a
3-D interpolated dilution likelihood table, and samples inside bounds taken
from SED posterior percentiles. That construction smooths the SED information,
discretises the dilution model, re-uses the SED prior implicitly through the
posterior it samples, and truncates the exploration to the SED credible region.
The present module removes all of these intermediate representations: the raw
measurements are evaluated at every sample and the priors are stated explicitly.

Sampling
--------
UltraNest ReactiveNestedSampler, 16 live points in test mode and 400 in final
mode, exploring the six-dimensional model directly. The live points are moved
with a slice step sampler because the joint posterior is a thin curved sliver
of the physical prior. Physically impossible samples and temperatures outside
the Koester or companion grids return a very low finite log-likelihood; they are
never clipped onto a grid boundary. Gravities follow the SED policy and are
clipped to the grid logg range (see depth_model).

Outputs (<output_base>/TIC{tic_id}_Gaia{gaia_id}/joint_ideal/)
--------------------------------------------------------------
    posterior_samples_joint_ideal.npy  [N, 7]: T_WD [K], R_WD [R_sun],
                                       T_MS [K], R_MS [R_sun], d [pc],
                                       A_V [mag], plx [mas]
    params_joint_ideal.json            p16/p50/p84 + logZ + method name
    plot_joint_ideal.pdf               7 marginal posteriors
    ultranest_joint_ideal/             UltraNest run directory (overwritten
                                       on every run: resume="overwrite")

All exported keys carry the _joint_ideal suffix. Aliases with the plain _joint
suffix are also returned so that the downstream steps of main_all.py (summary
printout, corner plot, mass stage) work unchanged.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import astropy.units as u
from scipy.special import ndtr, ndtri

try:
    from ultranest import ReactiveNestedSampler
    from ultranest.stepsampler import (
        SliceSampler,
        generate_mixture_random_direction,
    )
except ImportError:
    raise ImportError("ultranest is required: pip install ultranest")

# ── Helpers and prior limits reused from sed_fit.py (module left untouched) ───
from sed_fit import (
    fetch_gaia_astrometry,
    fetch_vizier_sed,
    _get_wave,
    _get_flux_err,
    _get_bandpass,
    _get_catalog,
    _deduplicate,
    _lnlike_gauss,
    _SedModel,
    _resolve_atmo_key,
    ATMO_GRID_RANGES,
    ATMO_MS_LOGG_FIXED,
    ATMO_MS_LOGG_MODE,
    ATMO_WD_LOGG_FIXED,
    ATMO_WD_LOGG_MODE,
    fetch_catalog_rel_errors,
    _SIGMA_LOOKUP_BANDS,
    SIGMA_LOOKUP_RADIUS_ARCSEC,
    ABS_FLOOR,
    FRAC_FLOOR,
    SED_MAX_REL_ERR_DEFAULT,
    VIZIER_RADIUS_ARCSEC_DEFAULT,
    T_WD_MIN,
    T_WD_MAX,
    R_WD_MIN_RSUN,
    R_WD_MAX_RSUN,
    R_MS_MIN_RSUN,
    R_MS_MAX_RSUN,
    AV_MIN,
    AV_MAX,
    AV_2C_SIGMA_MIN,
    N_SIGMA_PLX,
    MIN_PLX_SIGMA,
    PLX_MIN_MAS,
)

# ── Shared depth model and BHAC15 helpers ─────────────────────────────────────
from depth_model import DepthFluxModel
from dilution_fit import (
    _load_bhac15,
    _build_tms_grid,
    _load_profiles,
)

from wdms_common import T_MS_MIN_K, T_MS_MAX_K, derive_seed

# ── Numerical constants ───────────────────────────────────────────────────────
_LN_INVALID   = -1e100    # finite "impossible" value for invalid combinations
_SIGMA_R_MIN  = 1e-4      # R_sun; lower bound on the BHAC15 conditional width
_DEPTH_SIG_MIN = 1e-6     # guard on the eclipse-depth uncertainty
_CGS_FLAM     = u.erg / u.s / u.cm**2 / u.AA

PARAM_NAMES_IDEAL = ["T_WD", "R_WD", "T_MS", "R_MS", "plx_mas", "A_V"]

# The joint posterior is a thin curved sliver of the full physical prior: the
# SED constrains (R/d)^2 far more tightly than R and d separately, and the
# eclipse depth adds a second narrow ridge. Region-based rejection sampling
# stalls on such geometry, so the live points are moved with a slice step
# sampler instead. This changes only how the posterior is explored, not how it
# is defined.
_N_SLICE_STEPS = 4 * len(PARAM_NAMES_IDEAL)

# UltraNest settings of the ideal joint fit, per sampling mode.
# Defaults when this module is used alone; main_all.py sets its own values
# (see its "Fit settings" section, function apply_fit_settings).
JOINT_IDEAL_SAMPLING = {
    "test":  dict(min_num_live_points=16, max_iters=20_000),
    "final": dict(min_num_live_points=400, dlogz=0.2),
}


# ══════════════════════════════════════════════════════════════════════════════
# SED data preparation (same rules as sed_fit.py, no outlier rejection)
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_sed_data(sed_table,
                      gaia_id,
                      sed_exclude_windows_um,
                      sed_exclude_below_um,
                      sed_exclude_above_um,
                      sed_max_rel_err,
                      ra_deg=None,
                      dec_deg=None,
                      vizier_radius_arcsec=SIGMA_LOOKUP_RADIUS_ARCSEC,
                      recover_sigma_from_catalogs=True):
    """Return (w_fit [A], f_fit, e_fit [cgs], band_fit) for the SED likelihood."""
    wave_q          = _get_wave(sed_table)
    flux_q, eflux_q = _get_flux_err(sed_table, wave_q)

    wl_um_all    = wave_q.to(u.um).value.astype(float)
    flux_cgs_all = flux_q.to_value(_CGS_FLAM).astype(float)
    err_cgs_all  = (np.full_like(flux_cgs_all, np.nan) if eflux_q is None
                    else eflux_q.to_value(_CGS_FLAM).astype(float))
    band_all = _get_bandpass(sed_table)
    cat_all  = _get_catalog(sed_table)

    # Remove invalid or non-positive fluxes, then deduplicate (catalogue, filter)
    valid = np.isfinite(wl_um_all) & np.isfinite(flux_cgs_all) & (flux_cgs_all > 0)
    wl_um    = wl_um_all[valid].copy()
    flux_cgs = flux_cgs_all[valid].copy()
    err_cgs  = err_cgs_all[valid].copy()
    band     = band_all[valid].copy()
    cat      = cat_all[valid].copy()

    keep = _deduplicate(band, cat, err_cgs)
    wl_um, flux_cgs, err_cgs, band = (a[keep] for a in (wl_um, flux_cgs, err_cgs, band))

    n    = len(wl_um)
    mask = np.ones(n, dtype=bool)

    for lo, hi in (sed_exclude_windows_um or []):
        mask &= ~((wl_um >= lo) & (wl_um <= hi))
    if sed_exclude_below_um is not None:
        mask &= ~(wl_um < sed_exclude_below_um)
    if sed_exclude_above_um is not None:
        mask &= ~(wl_um > sed_exclude_above_um)

    # Same recovery as sed_fit.py: photometry delivered without an uncertainty
    # takes the sigma/F published by its source catalogue.
    if recover_sigma_from_catalogs:
        missing   = ~(np.isfinite(err_cgs) & (err_cgs > 0))
        candidate = {str(b).strip() for b in band[missing]} & _SIGMA_LOOKUP_BANDS
        if candidate:
            rel_from_cat = fetch_catalog_rel_errors(
                gaia_id, ra_deg=ra_deg, dec_deg=dec_deg,
                wanted_bands=candidate, radius_arcsec=vizier_radius_arcsec)
            n_rec = 0
            for i in np.where(missing)[0]:
                rel = rel_from_cat.get(str(band[i]).strip())
                if rel is not None and rel > 0 and flux_cgs[i] > 0:
                    err_cgs[i] = rel * flux_cgs[i]
                    n_rec += 1
            if n_rec:
                print(f"[JOINT-IDEAL] sigma recovered from source catalogues: "
                      f"{n_rec} point(s)")

    # Points without a usable sigma are dropped, then the sigma/F cut is applied
    has_sigma = np.isfinite(err_cgs) & (err_cgs > 0)
    mask &= has_sigma

    rel_err = np.full(n, np.nan)
    rel_err[has_sigma] = err_cgs[has_sigma] / flux_cgs[has_sigma]
    bad_rel = has_sigma & (rel_err > sed_max_rel_err)
    mask &= ~bad_rel

    n_used = int(np.sum(mask))
    print(f"[JOINT-IDEAL] SED points used={n_used}/{n}  "
          f"(no sigma: {int(np.sum(~has_sigma))}, "
          f"sigma/F > {sed_max_rel_err:.0%}: {int(np.sum(bad_rel))})")

    # Same sigma floor as sed_fit.py. Without it the catalogues' 0.3-2 % error
    # bars make the SED term worth hundreds in log-likelihood more than the
    # eclipse depths and the Gaia parallax combined, and the joint fit stops
    # being a compromise between the three datasets.
    w_fit = wl_um[mask] * 1e4
    f_fit = flux_cgs[mask].astype(float)
    e_fit = np.maximum(err_cgs[mask].astype(float),
                       np.maximum(FRAC_FLOOR * f_fit, ABS_FLOOR))
    band_fit = band[mask]

    if w_fit.size < 6:
        raise RuntimeError(
            f"[JOINT-IDEAL] Only {w_fit.size} SED points after filtering (need >= 6)"
        )
    return w_fit, f_fit, e_fit, band_fit


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_joint_fit_ideal(
    gaia_id,
    tic_id,
    output_base,
    depth_by_filter,
    filter_profile_files,
    bhac15_csv,
    koester_dir,
    da_track_path,
    db_track_path,
    wd_atmo_type="DA",
    mode="final",
    sed_exclude_windows_um=None,
    sed_exclude_below_um=0.35,
    sed_exclude_above_um=10.0,
    sed_max_rel_err=SED_MAX_REL_ERR_DEFAULT,
    recover_sigma_from_catalogs=True,
    vizier_radius_arcsec=VIZIER_RADIUS_ARCSEC_DEFAULT,
    download_root=None,
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
    av_1c_mu=None,
    av_1c_sigma=None,
    ms_atmosphere="blackbody",
    atmo_cache_dir=None,
    atmo_grid_ranges=None,
    ms_logg_mode=ATMO_MS_LOGG_MODE,
    ms_logg_fixed=ATMO_MS_LOGG_FIXED,
    ms_radius_prior="uniform",
    ms_relation=None,
    wd_atmosphere="blackbody",
    wd_logg_mode=ATMO_WD_LOGG_MODE,
    wd_logg_fixed=ATMO_WD_LOGG_FIXED,
    wd_cooling_track=None,
):
    """Run the direct joint fit on raw SED + eclipse-depth + parallax data.

    Parameters
    ----------
    gaia_id, tic_id : int
    output_base : str or Path
        Root of the results directory tree; the joint_ideal/ subdirectory is
        created automatically.
    depth_by_filter : dict
        {"filter_name": (depth, err_sym)} — measured primary-eclipse depths.
    filter_profile_files : dict
        {"filter_name": Path} — ASCII transmission curves.
    bhac15_csv, koester_dir : str or Path
    da_track_path, db_track_path : str or Path
        Bedard 2020 cooling tracks.
    wd_atmo_type : {"DA", "DB"}
    mode : {"test", "final"}
        "test"  → 16 live points (fast)
        "final" → 400 live points (production)
    sed_exclude_windows_um, sed_exclude_below_um, sed_exclude_above_um,
    sed_max_rel_err, recover_sigma_from_catalogs, vizier_radius_arcsec,
    download_root
        SED retrieval and filtering settings, identical to sed_fit.run_sed_fit.
    t_ms_min, t_ms_max, t_ms_step : float
        Requested T_MS interval and BHAC15 tabulation step [K].
    bhac_local_delta_t, bhac_local_min_points : BHAC15 local-window settings.
    bhac_age_logt_min, bhac_age_logt_max : BHAC15 age filter [log10 yr].
    photon_counting : photon (True) or energy (False) band weighting.
    filter_wavelength_cuts : dict or None
        {"filter_name": {"min_A": float, "max_A": float}}
    seed : int
        Master seed. NumPy's global random state is seeded with
        derive_seed(seed, "joint_ideal") right before the UltraNest run; both
        values are stored in params_joint_ideal.json ("seed_joint_ideal").
    av_1c_mu, av_1c_sigma : float
        1-comp A_V posterior median and width, used exactly as in sed_fit.py
        2-comp: A_V ~ N(av_1c_mu, av_1c_sigma) inside the log-likelihood, with
        the prior transform still Uniform(AV_MIN, AV_MAX).
        av_1c_sigma must already include the AV_2C_SIGMA_MIN floor.
    ms_radius_prior : {"uniform", "bhac15", "brown2022"}
        Prior on the companion radius; see the module docstring. "uniform"
        keeps the fit free of any evolutionary relation, "brown2022" needs
        ms_relation.
    ms_relation : mass.MDwarfRelation or None
        Semi-empirical relation used by ms_radius_prior="brown2022" only.

    Returns
    -------
    dict
        Keys with the _joint_ideal suffix, plus _joint aliases for the
        downstream main_all.py workflow.
    """
    if filter_wavelength_cuts is None:
        filter_wavelength_cuts = {}

    if av_1c_mu is None or av_1c_sigma is None:
        raise ValueError(
            "[JOINT-IDEAL] av_1c_mu and av_1c_sigma are required "
            "(same A_V prior as sed_fit.py 2-comp)."
        )
    av_1c_mu    = float(av_1c_mu)
    av_1c_sigma = max(float(av_1c_sigma), float(AV_2C_SIGMA_MIN))

    ms_radius_prior = str(ms_radius_prior).strip().lower()
    if ms_radius_prior not in {"uniform", "bhac15", "brown2022"}:
        raise ValueError(
            "[JOINT-IDEAL] ms_radius_prior must be 'uniform', 'bhac15' or "
            f"'brown2022', got {ms_radius_prior!r}"
        )
    if ms_radius_prior == "brown2022" and ms_relation is None:
        raise ValueError(
            "[JOINT-IDEAL] ms_radius_prior='brown2022' requires ms_relation "
            "(a mass.MDwarfRelation instance)."
        )
    # The relation only ever enters through the conditional radius prior, so it
    # is dropped everywhere else rather than silently reshaping the R(T) grid.
    r_prior_relation = ms_relation if ms_radius_prior == "brown2022" else None

    gaia_id        = int(gaia_id)
    tic_id         = int(tic_id)
    active_filters = list(depth_by_filter.keys())
    atmo           = str(wd_atmo_type).strip().upper()

    out_dir = Path(output_base) / f"TIC{tic_id}_Gaia{gaia_id}" / "joint_ideal"
    out_dir.mkdir(parents=True, exist_ok=True)

    if download_root is None:
        download_root = Path(output_base) / "downloads"

    sampling = dict(JOINT_IDEAL_SAMPLING["test" if mode == "test" else "final"])
    n_live = int(sampling.pop("min_num_live_points"))

    print(f"\n{'='*60}")
    print(f"[JOINT-IDEAL] Gaia DR3 {gaia_id}  /  TIC {tic_id}  /  mode={mode}")
    print(f"[JOINT-IDEAL] Direct joint fit: SED + eclipse depths + parallax")
    print(f"[JOINT-IDEAL] Filters : {active_filters}")
    print(f"[JOINT-IDEAL] n_live={n_live}")
    print(f"[JOINT-IDEAL] Output → {out_dir}")
    print(f"{'='*60}")

    # ── Step 1: Gaia astrometry and raw SED photometry ────────────────────
    ra, dec, plx_obs_mas, e_plx_mas = fetch_gaia_astrometry(gaia_id)
    _, sed_table = fetch_vizier_sed(ra, dec, gaia_id, download_root,
                                    vizier_radius_arcsec=vizier_radius_arcsec)
    if sed_table is None:
        raise RuntimeError(f"[JOINT-IDEAL] VizieR retrieval failed for Gaia {gaia_id}")

    w_fit, f_fit, e_fit, band_fit = _prepare_sed_data(
        sed_table,
        gaia_id,
        sed_exclude_windows_um,
        sed_exclude_below_um,
        sed_exclude_above_um,
        sed_max_rel_err,
        ra_deg=ra,
        dec_deg=dec,
        vizier_radius_arcsec=vizier_radius_arcsec,
        recover_sigma_from_catalogs=recover_sigma_from_catalogs,
    )

    # The SED term uses the same companion spectral model as sed_fit.py, so a
    # systematics study run there carries over to the joint fit. The eclipse
    # depths use the same companion model and gravities (DepthFluxModel).
    ms_atmo_key = _resolve_atmo_key(ms_atmosphere)
    sed_model = _SedModel(
        ms_atmo_key, w_fit, band_fit, cache_dir=atmo_cache_dir,
        grid_ranges=atmo_grid_ranges, ms_logg_mode=ms_logg_mode,
        ms_logg_fixed=ms_logg_fixed, wd_atmosphere=wd_atmosphere,
        wd_logg_mode=wd_logg_mode, wd_logg_fixed=wd_logg_fixed,
        wd_cooling_track=wd_cooling_track, t_ms_min=t_ms_min,
        t_ms_max=t_ms_max, verbose=True)
    print(f"[JOINT-IDEAL] SED companion model: {sed_model.label}")
    print(f"[JOINT-IDEAL] SED white-dwarf model: {sed_model.wd_label}")

    # ── Step 2: Gaia parallax term ────────────────────────────────────────
    # The published Gaia DR3 parallax and uncertainty are used as they are:
    # no zero-point correction and no uncertainty inflation is applied. Both
    # may be considered in a future improvement; their effect on this sample
    # has not been evaluated.
    # The Gaussian measurement enters the likelihood only; the prior transform
    # uses a plain uniform interval so the parallax is never counted twice.
    if (np.isfinite(plx_obs_mas) and plx_obs_mas > 0
            and np.isfinite(e_plx_mas) and e_plx_mas > 0):
        plx0_mas = float(plx_obs_mas)
        splx_mas = float(max(e_plx_mas, MIN_PLX_SIGMA))
    else:
        print("[JOINT-IDEAL] Invalid parallax — falling back to 5.0 +/- 3.75 mas")
        plx0_mas, splx_mas = 5.0, 3.75

    plx_min_mas = max(PLX_MIN_MAS, plx0_mas - N_SIGMA_PLX * splx_mas)
    plx_max_mas = plx0_mas + N_SIGMA_PLX * splx_mas
    _ln_plx_norm = -np.log(splx_mas * np.sqrt(2.0 * np.pi))
    print(f"[JOINT-IDEAL] Parallax: {plx0_mas:.4f} +/- {splx_mas:.4f} mas  "
          f"sampled on [{plx_min_mas:.4f}, {plx_max_mas:.4f}]")

    # ── Step 3: Shared depth model and BHAC15 relation ────────────────────
    profiles = _load_profiles(active_filters, filter_profile_files, filter_wavelength_cuts)
    for fk, (pw, pt) in profiles.items():
        active_range = pw[pt > 0]
        print(f"[JOINT-IDEAL] Filter {fk}: "
              f"{active_range.min():.0f}–{active_range.max():.0f} Å")
    track_path = da_track_path if atmo == "DA" else db_track_path
    depth = DepthFluxModel(
        profiles, koester_dir, wd_cooling_track=track_path,
        ms_atmosphere=ms_atmosphere, atmo_cache_dir=atmo_cache_dir,
        atmo_grid_ranges=atmo_grid_ranges, ms_logg_mode=ms_logg_mode,
        ms_logg_fixed=ms_logg_fixed, wd_logg_mode=wd_logg_mode,
        wd_logg_fixed=wd_logg_fixed, photon_counting=photon_counting)

    df_bh    = _load_bhac15(bhac15_csv, bhac_age_logt_min, bhac_age_logt_max)

    # BHAC15 R(T) statistics tabulated on the requested T_MS interval. They set
    # the T_MS coverage in every case, and the conditional radius prior when one
    # is requested; they are never used as a fixed radius.
    grid    = _build_tms_grid(
        df_bh, t_ms_min, t_ms_max, t_ms_step,
        bhac_local_delta_t, bhac_local_min_points, active_filters,
        ms_relation=r_prior_relation, depth_model=depth,
    )
    T2_grid = grid["T2_grid"]
    R2_p16  = grid["R2_p16"]
    R2_p50  = grid["R2_p50"]
    R2_p84  = grid["R2_p84"]
    if len(T2_grid) == 0:
        raise RuntimeError("[JOINT-IDEAL] BHAC15/companion-model T_MS coverage is empty")

    # Temperature ranges of the depth-term grids: samples outside are rejected
    # rather than clipped onto the grid boundary.
    pho_t_lo, pho_t_hi = (float(v) for v in depth.ms_teff_bounds)

    # ── Step 4: T_MS prior interval ───────────────────────────────────────
    # The requested bound (t_ms_min, t_ms_max; 2300–4000 K by default, the same
    # as the SED and dilution stages), narrowed only where the BHAC15 relation,
    # the depth-term companion model or the SED companion grid do not cover it.
    t_ms_lo = max(float(t_ms_min), float(T2_grid.min()), pho_t_lo,
                  sed_model.t_ms_bounds[0])
    t_ms_hi = min(float(t_ms_max), float(T2_grid.max()), pho_t_hi,
                  sed_model.t_ms_bounds[1])
    if not (t_ms_hi > t_ms_lo):
        raise RuntimeError(
            f"[JOINT-IDEAL] Empty T_MS prior interval: [{t_ms_lo}, {t_ms_hi}]"
        )
    if (t_ms_lo, t_ms_hi) != (float(t_ms_min), float(t_ms_max)):
        print(f"[JOINT-IDEAL] WARNING: model coverage narrows the requested T_MS "
              f"bound [{float(t_ms_min):.0f}, {float(t_ms_max):.0f}] K to "
              f"[{t_ms_lo:.0f}, {t_ms_hi:.0f}] K (BHAC15 "
              f"[{float(T2_grid.min()):.0f}, {float(T2_grid.max()):.0f}], depth model "
              f"[{pho_t_lo:.0f}, {pho_t_hi:.0f}], SED grid "
              f"[{sed_model.t_ms_bounds[0]:.0f}, {sed_model.t_ms_bounds[1]:.0f}] K)")
    print(f"[JOINT-IDEAL] T_MS prior: [{t_ms_lo:.0f}, {t_ms_hi:.0f}] K")
    if ms_radius_prior == "uniform":
        print(f"[JOINT-IDEAL] R_MS prior: uniform on "
              f"[{R_MS_MIN_RSUN:.3f}, {R_MS_MAX_RSUN:.3f}] R_sun "
              f"(no evolutionary relation in the sampled parameters)")
    else:
        relation_label = ("Brown et al. 2022 semi-empirical"
                          if ms_radius_prior == "brown2022"
                          else "BHAC15 theoretical")
        print(f"[JOINT-IDEAL] R_MS prior: {relation_label} truncated normal "
              f"conditional on T_MS, on "
              f"[{R_MS_MIN_RSUN:.3f}, {R_MS_MAX_RSUN:.3f}] R_sun")

    # A Koester white dwarf in the SED term narrows T_WD to the extent of that
    # grid; a blackbody leaves the sed_fit.py limits untouched.
    t_wd_lo = max(T_WD_MIN, sed_model.t_wd_bounds[0])
    t_wd_hi = min(T_WD_MAX, sed_model.t_wd_bounds[1])
    if not (t_wd_hi > t_wd_lo):
        raise RuntimeError(
            f"[JOINT-IDEAL] Empty T_WD prior interval: [{t_wd_lo}, {t_wd_hi}]"
        )
    print(f"[JOINT-IDEAL] T_WD prior: [{t_wd_lo:.0f}, {t_wd_hi:.0f}] K")

    # ── Step 5: Observed eclipse depths ───────────────────────────────────
    depth_obs = {fk: float(depth_by_filter[fk][0]) for fk in active_filters}
    depth_sig = {fk: max(float(depth_by_filter[fk][1]), _DEPTH_SIG_MIN)
                 for fk in active_filters}
    _ln_depth_norm = float(np.sum([
        -np.log(depth_sig[fk] * np.sqrt(2.0 * np.pi)) for fk in active_filters
    ]))
    for fk in active_filters:
        print(f"[JOINT-IDEAL] depth obs [{fk}] = "
              f"{depth_obs[fk]:.4f} +/- {depth_sig[fk]:.4f}")

    # Same A_V prior as sed_fit.py 2-comp: Gaussian from the 1-comp posterior,
    # with the prior transform still Uniform(AV_MIN, AV_MAX).
    _ln_av_norm = -0.5 * np.log(2.0 * np.pi * av_1c_sigma**2)

    def _ln_av_prior(A_V):
        return _ln_av_norm - 0.5 * ((A_V - av_1c_mu) / av_1c_sigma)**2

    print(f"[JOINT-IDEAL] A_V prior: N({av_1c_mu:.4f}, {av_1c_sigma:.4f}) "
          f"from SED 1-comp (same as sed_fit 2-comp)")

    # ── Step 6: Prior transform ───────────────────────────────────────────
    _log_R_WD_lo   = np.log(R_WD_MIN_RSUN)
    _log_R_WD_span = np.log(R_WD_MAX_RSUN / R_WD_MIN_RSUN)

    _uniform_R_MS = (ms_radius_prior == "uniform")

    def prior_transform(cube):
        """Unit cube → physical parameters.

        With ms_radius_prior="uniform" the companion radius is drawn from the
        same flat interval as in sed_fit.py, independently of T_MS. Otherwise it
        is drawn from the conditional truncated normal evaluated at the T_MS
        coordinate transformed just above, so the radius relation is carried
        entirely by the prior measure.
        """
        uT1, uR1, uT2, uR2, uP, uA = cube

        T_WD = t_wd_lo + uT1 * (t_wd_hi - t_wd_lo)
        R_WD = np.exp(_log_R_WD_lo + uR1 * _log_R_WD_span)
        T_MS = t_ms_lo + uT2 * (t_ms_hi - t_ms_lo)

        if _uniform_R_MS:
            R_MS = R_MS_MIN_RSUN + uR2 * (R_MS_MAX_RSUN - R_MS_MIN_RSUN)
        else:
            # Conditional prior R_MS | T_MS ~ N(R50(T_MS), sigma_R), truncated
            # to the physical R_MS limits.
            r50 = float(np.interp(T_MS, T2_grid, R2_p50))
            r16 = float(np.interp(T_MS, T2_grid, R2_p16))
            r84 = float(np.interp(T_MS, T2_grid, R2_p84))
            sigma_R = max(0.5 * (r84 - r16), _SIGMA_R_MIN)

            z_lo = (R_MS_MIN_RSUN - r50) / sigma_R
            z_hi = (R_MS_MAX_RSUN - r50) / sigma_R
            cdf_lo, cdf_hi = ndtr(z_lo), ndtr(z_hi)
            if cdf_hi - cdf_lo < 1e-12:
                # Degenerate truncation: fall back to the uniform physical range
                R_MS = R_MS_MIN_RSUN + uR2 * (R_MS_MAX_RSUN - R_MS_MIN_RSUN)
            else:
                z    = ndtri(cdf_lo + uR2 * (cdf_hi - cdf_lo))
                R_MS = float(np.clip(r50 + sigma_R * z,
                                     R_MS_MIN_RSUN, R_MS_MAX_RSUN))

        plx = plx_min_mas + uP * (plx_max_mas - plx_min_mas)
        A_V = AV_MIN + uA * (AV_MAX - AV_MIN)

        return np.array([T_WD, R_WD, T_MS, R_MS, plx, A_V])

    # ── Step 7: Joint log-likelihood ──────────────────────────────────────
    def log_likelihood(params):
        """ln L_SED + ln L_dilution + ln L_parallax for one sample."""
        T_WD, R_WD, T_MS, R_MS, plx, A_V = params

        if not (R_WD > 0 and R_MS > 0 and plx > 0 and A_V >= 0):
            return _LN_INVALID

        d_pc = 1000.0 / plx

        # -- Gravities, computed once and shared by both terms -------------
        logg_WD = depth.wd_logg(T_WD, R_WD)
        logg_MS = depth.ms_logg(T_MS, R_MS)

        # -- SED: same companion model and physics as sed_fit.py -----------
        model_sed = sed_model.bands_2comp([T_WD, R_WD, T_MS, R_MS, d_pc, A_V],
                                          wd_logg=logg_WD, ms_logg=logg_MS)
        if not np.all(np.isfinite(model_sed)):
            return _LN_INVALID
        lnL_sed = _lnlike_gauss(model_sed, f_fit, e_fit)

        # -- Gaia parallax --------------------------------------------------
        lnL_plx = _ln_plx_norm - 0.5 * ((plx - plx0_mas) / splx_mas)**2

        # -- A_V prior (identical to sed_fit.py 2-comp) ---------------------
        ln_av = _ln_av_prior(A_V)

        # -- Eclipse depths: interpolated band fluxes, no extrapolation -----
        # (NaN outside the Koester / companion temperature ranges → invalid)
        chi2 = 0.0
        for fk in active_filters:
            F_WD_b = depth.wd_band_flux(fk, T_WD, logg_WD)
            if not (np.isfinite(F_WD_b) and F_WD_b > 0):
                return _LN_INVALID
            F_MS_b = depth.ms_band_flux(fk, T_MS, logg_MS)
            if not (np.isfinite(F_MS_b) and F_MS_b > 0):
                return _LN_INVALID
            q           = (R_MS / R_WD)**2 * F_MS_b / F_WD_b
            depth_model = 1.0 / (1.0 + q)
            chi2       += ((depth_obs[fk] - depth_model) / depth_sig[fk])**2

        lnL_dil = -0.5 * chi2 + _ln_depth_norm

        total = lnL_sed + lnL_dil + lnL_plx + ln_av
        return float(total) if np.isfinite(total) else _LN_INVALID

    # ── Step 8: UltraNest ─────────────────────────────────────────────────
    log_dir = str(out_dir / "ultranest_joint_ideal")
    # UltraNest (and its slice step sampler) draw from NumPy's global random
    # state: seed it right before the sampler is built and run.
    seed_run = derive_seed(seed, "joint_ideal")
    np.random.seed(seed_run)
    sampler = ReactiveNestedSampler(
        PARAM_NAMES_IDEAL,
        log_likelihood,
        prior_transform,
        log_dir=log_dir,
        resume="overwrite",
    )
    sampler.stepsampler = SliceSampler(
        nsteps=_N_SLICE_STEPS,
        generate_direction=generate_mixture_random_direction,
    )

    run_kw = dict(min_num_live_points=n_live, show_status=True,
                  viz_callback=False, frac_remain=0.01, Lepsilon=0.1,
                  max_num_improvement_loops=3)
    run_kw.update(sampling)

    result = sampler.run(**run_kw)
    sampler.print_results()

    samples = np.array(result["samples"], dtype=float)   # [N, 6]
    logZ_joint_ideal    = float(result["logz"])
    logZerr_joint_ideal = float(result["logzerr"])

    print(f"\n[JOINT-IDEAL] logZ = {logZ_joint_ideal:.3f} "
          f"± {logZerr_joint_ideal:.3f}")
    print(f"[JOINT-IDEAL] Posterior samples: {samples.shape[0]}")

    # ── Step 9: Derived distance and credible intervals ───────────────────
    # Every parallax posterior sample is converted with d_pc = 1000 / plx_mas.
    plx_samples = samples[:, 4]
    d_samples   = 1000.0 / plx_samples

    T_WD_p16, T_WD_p50, T_WD_p84 = np.percentile(samples[:, 0], [16, 50, 84])
    R_WD_p16, R_WD_p50, R_WD_p84 = np.percentile(samples[:, 1], [16, 50, 84])
    T_MS_p16, T_MS_p50, T_MS_p84 = np.percentile(samples[:, 2], [16, 50, 84])
    R_MS_p16, R_MS_p50, R_MS_p84 = np.percentile(samples[:, 3], [16, 50, 84])
    plx_p16,  plx_p50,  plx_p84  = np.percentile(plx_samples,   [16, 50, 84])
    AV_p16,   AV_p50,   AV_p84   = np.percentile(samples[:, 5], [16, 50, 84])
    d_p16,    d_p50,    d_p84    = np.percentile(d_samples,     [16, 50, 84])

    print(f"[JOINT-IDEAL] T_WD = {T_WD_p50:.0f} K  "
          f"(-{T_WD_p50-T_WD_p16:.0f} / +{T_WD_p84-T_WD_p50:.0f})")
    print(f"[JOINT-IDEAL] R_WD = {R_WD_p50:.5f} R_sun  "
          f"(-{R_WD_p50-R_WD_p16:.5f} / +{R_WD_p84-R_WD_p50:.5f})")
    print(f"[JOINT-IDEAL] T_MS = {T_MS_p50:.0f} K  "
          f"(-{T_MS_p50-T_MS_p16:.0f} / +{T_MS_p84-T_MS_p50:.0f})")
    print(f"[JOINT-IDEAL] R_MS = {R_MS_p50:.4f} R_sun  "
          f"(-{R_MS_p50-R_MS_p16:.4f} / +{R_MS_p84-R_MS_p50:.4f})")
    print(f"[JOINT-IDEAL] plx  = {plx_p50:.4f} mas  "
          f"(-{plx_p50-plx_p16:.4f} / +{plx_p84-plx_p50:.4f})")
    print(f"[JOINT-IDEAL] d    = {d_p50:.1f} pc  "
          f"(-{d_p50-d_p16:.1f} / +{d_p84-d_p50:.1f})")
    print(f"[JOINT-IDEAL] A_V  = {AV_p50:.3f} mag  "
          f"(-{AV_p50-AV_p16:.3f} / +{AV_p84-AV_p50:.3f})")

    # ── Step 10: Save posterior samples ───────────────────────────────────
    # Columns: T_WD, R_WD, T_MS, R_MS, d, A_V, plx — the first six match the
    # convention of the legacy products, the parallax samples are kept as well.
    post_arr = np.column_stack([
        samples[:, 0], samples[:, 1], samples[:, 2], samples[:, 3],
        d_samples, samples[:, 5], plx_samples,
    ]).astype(np.float64)
    path_samp = out_dir / "posterior_samples_joint_ideal.npy"
    np.save(path_samp, post_arr)
    print(f"[JOINT-IDEAL] Saved posterior → {path_samp}")

    # ── Step 11: Save params_joint_ideal.json ─────────────────────────────
    params_joint_ideal = {
        "gaia_id": gaia_id,
        "tic_id":  tic_id,
        "method":  "direct_joint",
        "T_WD_joint_ideal": {"p16": float(T_WD_p16), "p50": float(T_WD_p50),
                             "p84": float(T_WD_p84)},
        "R_WD_joint_ideal": {"p16": float(R_WD_p16), "p50": float(R_WD_p50),
                             "p84": float(R_WD_p84)},
        "T_MS_joint_ideal": {"p16": float(T_MS_p16), "p50": float(T_MS_p50),
                             "p84": float(T_MS_p84)},
        "R_MS_joint_ideal": {"p16": float(R_MS_p16), "p50": float(R_MS_p50),
                             "p84": float(R_MS_p84)},
        "plx_joint_ideal":  {"p16": float(plx_p16),  "p50": float(plx_p50),
                             "p84": float(plx_p84)},
        "d_joint_ideal":    {"p16": float(d_p16),    "p50": float(d_p50),
                             "p84": float(d_p84)},
        "AV_joint_ideal":   {"p16": float(AV_p16),   "p50": float(AV_p50),
                             "p84": float(AV_p84)},
        "logZ_joint_ideal":    logZ_joint_ideal,
        "logZerr_joint_ideal": logZerr_joint_ideal,
        "wd_atmo_type_joint_ideal": atmo,
        "filters_joint_ideal":      active_filters,
        "n_sed_points_joint_ideal": int(w_fit.size),
        "parallax_joint_ideal": {
            "plx0_mas":    float(plx0_mas),
            "e_plx_mas":   float(splx_mas),
            "plx_min_mas": float(plx_min_mas),
            "plx_max_mas": float(plx_max_mas),
        },
        "av_prior_joint_ideal": {
            "kind":  "gaussian_from_1comp",
            "mu":    float(av_1c_mu),
            "sigma": float(av_1c_sigma),
            "min":   float(AV_MIN),
            "max":   float(AV_MAX),
        },
        "T_MS_prior_joint_ideal": {"lo": float(t_ms_lo), "hi": float(t_ms_hi),
                                   "requested": [float(t_ms_min), float(t_ms_max)]},
        "sed_max_rel_err_joint_ideal": float(sed_max_rel_err),
        "seed_joint_ideal": {"master": int(seed), "ultranest": int(seed_run)},
        "R_MS_prior_joint_ideal": {
            "kind": ("uniform" if ms_radius_prior == "uniform"
                     else "truncated_normal_conditional_on_T_MS"),
            "relation": ({"uniform":   "none (model-free)",
                          "bhac15":    "BHAC15 theoretical",
                          "brown2022": "Brown et al. 2022 semi-empirical"}
                         [ms_radius_prior]),
            "min": float(R_MS_MIN_RSUN),
            "max": float(R_MS_MAX_RSUN),
        },
        "wd_model_joint_ideal": {
            "sed_term": sed_model.wd_model,
            "label": sed_model.wd_label,
            "wd_logg_mode": (str(wd_logg_mode) if sed_model.wd_key != "blackbody"
                             else "not applicable"),
            "t_wd_prior_K": [float(t_wd_lo), float(t_wd_hi)],
            "depth_term": "Koester DA (always)",
        },
        "depth_model_joint_ideal": depth.describe(),
        "gravities_joint_ideal": "logg_WD and logg_MS computed once per sample "
                                 "and shared by the SED and depth terms",
    }
    path_params = out_dir / "params_joint_ideal.json"
    with open(path_params, "w", encoding="utf-8") as fp:
        json.dump(params_joint_ideal, fp, indent=2)
    print(f"[JOINT-IDEAL] Saved params  → {path_params}")

    # ── Step 12: Plot ─────────────────────────────────────────────────────
    path_plot = out_dir / "plot_joint_ideal.pdf"
    _plot_joint_ideal(
        columns = [samples[:, 0], samples[:, 1], samples[:, 2], samples[:, 3],
                   plx_samples, d_samples, samples[:, 5]],
        p16 = [T_WD_p16, R_WD_p16, T_MS_p16, R_MS_p16, plx_p16, d_p16, AV_p16],
        p50 = [T_WD_p50, R_WD_p50, T_MS_p50, R_MS_p50, plx_p50, d_p50, AV_p50],
        p84 = [T_WD_p84, R_WD_p84, T_MS_p84, R_MS_p84, plx_p84, d_p84, AV_p84],
        out_path = path_plot,
    )

    # ── Return ─────────────────────────────────────────────────────────────
    out = {
        # Medians (p50)
        "T_WD_joint_ideal": float(T_WD_p50),
        "R_WD_joint_ideal": float(R_WD_p50),
        "T_MS_joint_ideal": float(T_MS_p50),
        "R_MS_joint_ideal": float(R_MS_p50),
        "plx_joint_ideal":  float(plx_p50),
        "d_joint_ideal":    float(d_p50),
        "AV_joint_ideal":   float(AV_p50),
        # Full 1-sigma intervals
        "T_WD_p16_joint_ideal": float(T_WD_p16), "T_WD_p84_joint_ideal": float(T_WD_p84),
        "R_WD_p16_joint_ideal": float(R_WD_p16), "R_WD_p84_joint_ideal": float(R_WD_p84),
        "T_MS_p16_joint_ideal": float(T_MS_p16), "T_MS_p84_joint_ideal": float(T_MS_p84),
        "R_MS_p16_joint_ideal": float(R_MS_p16), "R_MS_p84_joint_ideal": float(R_MS_p84),
        "plx_p16_joint_ideal":  float(plx_p16),  "plx_p84_joint_ideal":  float(plx_p84),
        "d_p16_joint_ideal":    float(d_p16),    "d_p84_joint_ideal":    float(d_p84),
        "AV_p16_joint_ideal":   float(AV_p16),   "AV_p84_joint_ideal":   float(AV_p84),
        # Bayesian evidence and method identifier
        "logZ_joint_ideal":    logZ_joint_ideal,
        "logZerr_joint_ideal": logZerr_joint_ideal,
        "method_joint_ideal":  "direct_joint",
        # Posterior samples kept in memory
        "plx_samples_joint_ideal": plx_samples,
        # File paths
        "posterior_samples_joint_ideal_path": str(path_samp),
        "params_joint_ideal_path":            str(path_params),
        "plot_joint_ideal_path":              str(path_plot),
    }

    # Aliases with the plain _joint suffix so that the main_all.py summary, the
    # corner plot of display.py and mass.py keep working without modification.
    out.update({
        "T_WD_joint": out["T_WD_joint_ideal"],
        "R_WD_joint": out["R_WD_joint_ideal"],
        "T_MS_joint": out["T_MS_joint_ideal"],
        "R_MS_joint": out["R_MS_joint_ideal"],
        "d_joint":    out["d_joint_ideal"],
        "AV_joint":   out["AV_joint_ideal"],
        "T_WD_p16_joint": out["T_WD_p16_joint_ideal"], "T_WD_p84_joint": out["T_WD_p84_joint_ideal"],
        "R_WD_p16_joint": out["R_WD_p16_joint_ideal"], "R_WD_p84_joint": out["R_WD_p84_joint_ideal"],
        "T_MS_p16_joint": out["T_MS_p16_joint_ideal"], "T_MS_p84_joint": out["T_MS_p84_joint_ideal"],
        "R_MS_p16_joint": out["R_MS_p16_joint_ideal"], "R_MS_p84_joint": out["R_MS_p84_joint_ideal"],
        "d_p16_joint":    out["d_p16_joint_ideal"],    "d_p84_joint":    out["d_p84_joint_ideal"],
        "AV_p16_joint":   out["AV_p16_joint_ideal"],   "AV_p84_joint":   out["AV_p84_joint_ideal"],
        "logZ_joint":     logZ_joint_ideal,
        "logZerr_joint":  logZerr_joint_ideal,
        "posterior_samples_joint_path": str(path_samp),
        "params_joint_path":            str(path_params),
        "plot_joint_path":              str(path_plot),
    })
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Plot helper
# ══════════════════════════════════════════════════════════════════════════════

_LABELS_IDEAL = [
    r"$T_\mathrm{WD}$ [K]",
    r"$R_\mathrm{WD}$ [$R_\odot$]",
    r"$T_\mathrm{MS}$ [K]",
    r"$R_\mathrm{MS}$ [$R_\odot$]",
    r"$\varpi$ [mas]",
    r"$d$ [pc]  (derived)",
    r"$A_V$ [mag]",
]
_COLORS_IDEAL = ["steelblue", "steelblue", "darkorange", "darkorange",
                 "seagreen", "seagreen", "dimgray"]


def _plot_joint_ideal(columns, p16, p50, p84, out_path):
    """Seven 1-D marginal posteriors: the six sampled parameters + derived d."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 7), constrained_layout=True)
    axes = axes.ravel()

    for ax, label, col, data, lo, med, hi in zip(
        axes, _LABELS_IDEAL, _COLORS_IDEAL, columns, p16, p50, p84
    ):
        ax.hist(data, bins=50, alpha=0.75, color=col, density=True)
        ax.axvline(lo,  color="k",   ls="--", lw=1.1, label=f"p16 = {lo:.4g}")
        ax.axvline(med, color="red", ls="-",  lw=1.5, label=f"p50 = {med:.4g}")
        ax.axvline(hi,  color="k",   ls="--", lw=1.1, label=f"p84 = {hi:.4g}")
        ax.set_xlabel(label, fontsize=14)
        ax.set_ylabel("density", fontsize=14)
        ax.legend(frameon=False, fontsize=11)
        ax.grid(alpha=0.3)

    axes[len(_LABELS_IDEAL)].set_visible(False)

    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[JOINT-IDEAL] Saved plot   → {out_path}")
