#!/usr/bin/env python3
"""dilution_uniform_fit.py — Baraffe-free dilution posterior for WD+MS systems.

The historical dilution stage (dilution_fit.py) closes the system with the
BHAC15 R_MS(T_MS) relation. That is the wrong prior for inflated M dwarfs in
WD+MS binaries: the theoretical isochrone was not built for them, and tying
the sampled radius to it hides the very inflation one would like to measure.

This module is the dilution counterpart of joint_fit_ideal with
ms_radius_prior="uniform". T_MS and R_MS are independent a priori,

    T_MS  ~ Uniform(t_ms_min, t_ms_max)
    R_MS  ~ Uniform(R_MS_MIN_RSUN, R_MS_MAX_RSUN)

and the eclipse depths alone draw a thin band in the (T_MS, R_MS) plane:

    R_MS(T_MS) = R_WD * sqrt[ (1/depth - 1) * F_WD / F_MS(T_MS) ]

No BHAC15 (or Brown) radius–temperature relation is read. The band fluxes and
gravities come from the shared depth model (depth_model.DepthFluxModel): the
SED companion model and gravity prescription, the Koester DA grid with the
SED white-dwarf gravity policy, both interpolated bilinearly in ln F.

Monte Carlo, one iteration per sample:
    1. (T_WD, R_WD) drawn from the SED posterior (white-dwarf uncertainty);
    2. T_MS drawn from its uniform prior;
    3. each observed depth perturbed ONCE by its measurement error;
    4. R_MS obtained by deterministic inversion of the depth equation (an
       inverse-variance weighted mean over filters when there are several);
    5. samples outside the R_MS prior interval are rejected.
The depth uncertainty therefore enters once, through step 3; no additional
radius scatter is drawn.

T_WD and R_WD are still drawn from the SED posterior, as in dilution_fit.py:
this stage answers "what the eclipse demands of the companion, given the
white dwarf the SED has already measured".

main_all.py calls run_dilution_uniform_fit when MS_RADIUS_PRIOR == "uniform"
and keeps run_dilution_fit for "bhac15" / "brown2022". Outputs share the
dilution/ directory and the same filenames so display.py does not change.

Outputs (<output_base>/TIC{id}_Gaia{id}/dilution/):
    posterior_samples_dil.npy          [N_valid, 2]  T_MS [K], R_MS [R_sun]
    posterior_samples_dil_matched.npy  [N_valid, 4]  T_WD [K], R_WD [R_sun],
                                       T_MS [K], R_MS [R_sun] (same rows)
    params_dil.json             priors, p16/p50/p84 for T_MS, R_MS, logg_WD
    plot_dil.pdf                T_MS and R_MS histograms, (T_MS, R_MS) ridge
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sed_fit import (
    R_MS_MIN_RSUN,
    R_MS_MAX_RSUN,
    ATMO_MS_LOGG_FIXED,
    ATMO_MS_LOGG_MODE,
    ATMO_WD_LOGG_FIXED,
    ATMO_WD_LOGG_MODE,
)
import sed_atmosphere_models as sam
from wdms_common import T_MS_MIN_K, T_MS_MAX_K
from depth_model import DepthFluxModel
import dilution_fit
from dilution_fit import _DEPTH_CLIP, _load_profiles


def run_dilution_uniform_fit(
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
    mode="final",
    t_ms_min=T_MS_MIN_K,
    t_ms_max=T_MS_MAX_K,
    photon_counting=True,
    filter_wavelength_cuts=None,
    seed=42,
    ms_atmosphere="phoenix",
    atmo_cache_dir=None,
    atmo_grid_ranges=None,
    ms_logg_mode=ATMO_MS_LOGG_MODE,
    ms_logg_fixed=ATMO_MS_LOGG_FIXED,
    wd_logg_mode=ATMO_WD_LOGG_MODE,
    wd_logg_fixed=ATMO_WD_LOGG_FIXED,
):
    """Infer (T_MS, R_MS) from eclipse depths with a uniform radius prior.

    Same inputs and returned keys as dilution_fit.run_dilution_fit, except
    there is no BHAC15 table, no ms_relation, and no lnL_dil cube (that cube
    is defined along R_MS = R_BHAC(T_MS) and has no counterpart here). The
    legacy joint fit cannot consume this posterior; use joint_fit_ideal.

    ms_atmosphere, atmo_cache_dir, atmo_grid_ranges, ms_logg_mode,
    ms_logg_fixed, wd_logg_mode, wd_logg_fixed are the SED settings, used by
    the shared depth model; ms_logg_fixed only matters for ms_logg_mode="fixed".
    """
    gaia_id = int(gaia_id)
    tic_id = int(tic_id)
    active_filters = list(depth_by_filter.keys())
    atmo = str(wd_atmo_type).strip().upper()

    out_dir = Path(output_base) / f"TIC{tic_id}_Gaia{gaia_id}" / "dilution"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_mc = int(dilution_fit.DILUTION_SAMPLING["test" if mode == "test" else "final"]["n_mc"])
    rng = np.random.default_rng(seed)

    print(f"\n{'='*60}")
    print(f"[DIL-U] Gaia DR3 {gaia_id}  /  TIC {tic_id}  /  mode={mode}")
    print(f"[DIL-U] Uniform R_MS prior — no BHAC15 R(T)")
    print(f"[DIL-U] Filters : {active_filters}   logg_MS: '{ms_logg_mode}' prescription")
    print(f"[DIL-U] N_MC={n_mc}")
    print(f"{'='*60}")

    # ── SED white dwarf ───────────────────────────────────────────────────
    samp2 = np.load(posterior_samples_sed_path)
    if samp2.ndim != 2 or samp2.shape[1] < 2:
        raise ValueError(f"posterior_samples_sed must be [N, ≥2], got {samp2.shape}")
    t_wd_post = samp2[:, 0].astype(float)
    r_wd_post = samp2[:, 1].astype(float)
    ok_post = np.isfinite(t_wd_post) & np.isfinite(r_wd_post) & (r_wd_post > 0)
    t_wd_post, r_wd_post = t_wd_post[ok_post], r_wd_post[ok_post]
    print(f"[DIL-U] SED posterior samples: {len(t_wd_post)} valid")

    # ── Shared depth model ────────────────────────────────────────────────
    profiles = _load_profiles(active_filters, filter_profile_files, filter_wavelength_cuts)
    depth = DepthFluxModel(
        profiles, koester_dir,
        wd_cooling_track=da_track_path if atmo == "DA" else db_track_path,
        ms_atmosphere=ms_atmosphere, atmo_cache_dir=atmo_cache_dir,
        atmo_grid_ranges=atmo_grid_ranges, ms_logg_mode=ms_logg_mode,
        ms_logg_fixed=ms_logg_fixed, wd_logg_mode=wd_logg_mode,
        wd_logg_fixed=wd_logg_fixed, photon_counting=photon_counting)

    logg_wd_post = depth.wd_logg(t_wd_post, r_wd_post)
    logg_wd_p16 = float(np.nanpercentile(logg_wd_post, 16))
    logg_wd_med = float(np.nanpercentile(logg_wd_post, 50))
    logg_wd_p84 = float(np.nanpercentile(logg_wd_post, 84))
    print(f"[DIL-U] logg_WD: {logg_wd_p16:.3f} / {logg_wd_med:.3f} / {logg_wd_p84:.3f}")

    # ── T_MS prior: requested bound within the companion model coverage ───
    # The requested bound (2300–4000 K by default, the same as the SED and
    # joint stages) is narrowed only where the companion model has no flux.
    t_axis = np.arange(float(t_ms_min), float(t_ms_max) + 12.5, 25.0)
    t_axis = t_axis[t_axis <= float(t_ms_max) + 1e-9]
    finite_t = np.ones(t_axis.size, dtype=bool)
    for fk in active_filters:
        # Coverage test at the empirical main-sequence radius (only used by
        # the gravity when ms_logg_mode="from_radius").
        g_axis = depth.ms_logg(t_axis, sam.radius_main_sequence(t_axis))
        f_axis = depth.ms_band_flux(fk, t_axis, g_axis)
        finite_t &= np.isfinite(f_axis) & (f_axis > 0)
    if not finite_t.any():
        raise RuntimeError("[DIL-U] No companion-model coverage on the requested T_MS interval")
    t_lo = max(float(t_axis[finite_t].min()), float(t_ms_min))
    t_hi = min(float(t_axis[finite_t].max()), float(t_ms_max))
    if not (t_hi > t_lo):
        raise RuntimeError(f"[DIL-U] Empty T_MS interval: [{t_lo}, {t_hi}]")
    if (t_lo, t_hi) != (float(t_ms_min), float(t_ms_max)):
        print(f"[DIL-U] WARNING: companion-model coverage narrows the requested T_MS "
              f"bound [{float(t_ms_min):.0f}, {float(t_ms_max):.0f}] K to "
              f"[{t_lo:.0f}, {t_hi:.0f}] K (no extrapolation)")
    print(f"[DIL-U] T_MS prior: uniform [{t_lo:.0f}, {t_hi:.0f}] K")
    print(f"[DIL-U] R_MS prior: uniform [{R_MS_MIN_RSUN:.3f}, {R_MS_MAX_RSUN:.3f}] R_sun")

    # ── Monte Carlo: one depth perturbation, deterministic inversion ──────
    print(f"\n[DIL-U] Running MC ({n_mc} iterations)…")
    n_post = len(t_wd_post)
    idx = rng.integers(0, n_post, size=n_mc)
    t_wd_mc = t_wd_post[idx]
    r_wd_mc = r_wd_post[idx]
    logg_wd_mc = logg_wd_post[idx]
    t_ms_mc = rng.uniform(t_lo, t_hi, size=n_mc)

    # The only use of the depth uncertainty: one Gaussian perturbation of each
    # observed depth per sample.
    depth_draw = {
        fk: np.clip(rng.normal(float(d_obs), float(d_err), n_mc),
                    _DEPTH_CLIP[0], _DEPTH_CLIP[1])
        for fk, (d_obs, d_err) in depth_by_filter.items()
    }

    f_wd = {fk: depth.wd_band_flux(fk, t_wd_mc, logg_wd_mc) for fk in active_filters}
    r_req = {fk: depth.required_radius(fk, depth_draw[fk], r_wd_mc, f_wd[fk], t_ms_mc)
             for fk in active_filters}

    # Several filters: combine the deterministic radii with fixed inverse-
    # variance weights, w ∝ 1 / (σ_R/R · R)^2 with σ_R/R = 0.5 σ_d / (d (1-d))
    # from the observed depth. This is a weighted mean, not an extra draw: for
    # a single filter r_hat is exactly the inverted radius.
    num = np.zeros(n_mc); den = np.zeros(n_mc); ok = np.ones(n_mc, bool)
    for fk in active_filters:
        d_obs, d_err = depth_by_filter[fk]
        frac = 0.5 * float(d_err) / max(float(d_obs) * (1.0 - float(d_obs)), 1e-8)
        r = r_req[fk]
        ok &= np.isfinite(r) & (r > 0) & np.isfinite(f_wd[fk])
        with np.errstate(invalid="ignore", divide="ignore"):
            w = 1.0 / np.maximum(frac * r, 1e-8) ** 2
        num += np.where(ok, w * r, 0.0); den += np.where(ok, w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_hat = np.where(ok & (den > 0), num / den, np.nan)
    valid = ok & np.isfinite(r_hat) & (r_hat >= R_MS_MIN_RSUN) & (r_hat <= R_MS_MAX_RSUN)

    t_ms_out = np.where(valid, t_ms_mc, np.nan)
    r_ms_out = np.where(valid, r_hat, np.nan)
    depth_mod = {}
    g_ms = depth.ms_logg(t_ms_mc, r_hat)
    for fk in active_filters:
        f_ms = depth.ms_band_flux(fk, t_ms_mc, g_ms)
        depth_mod[fk] = np.where(valid, depth.depth(fk, r_wd_mc, f_wd[fk], r_hat, f_ms), np.nan)

    n_valid = int(valid.sum())
    if n_valid < 50:
        raise RuntimeError(f"[DIL-U] Only {n_valid} valid MC samples")
    print(f"[DIL-U] Valid samples: {n_valid}/{n_mc}")

    t_ms_s, r_ms_s = t_ms_out[valid], r_ms_out[valid]
    t_p16, t_p50, t_p84 = np.percentile(t_ms_s, [16, 50, 84])
    r_p16, r_p50, r_p84 = np.percentile(r_ms_s, [16, 50, 84])
    print(f"[DIL-U] T_MS = {t_p50:.0f} K  (-{t_p50-t_p16:.0f}/+{t_p84-t_p50:.0f})  "
          f"[uniform prior — the depth does not pick a temperature alone]")
    print(f"[DIL-U] R_MS = {r_p50:.4f} R_sun  "
          f"(-{r_p50-r_p16:.4f}/+{r_p84-r_p50:.4f})")

    path_samp = out_dir / "posterior_samples_dil.npy"
    np.save(path_samp, np.column_stack([t_ms_s, r_ms_s]).astype(np.float64))
    path_matched = out_dir / "posterior_samples_dil_matched.npy"
    np.save(path_matched, np.column_stack([t_wd_mc[valid], r_wd_mc[valid],
                                           t_ms_s, r_ms_s]).astype(np.float64))
    print(f"[DIL-U] Saved samples → {path_samp} (+ matched [T_WD, R_WD, T_MS, R_MS])")

    depth_summary = {}
    for fk in active_filters:
        dm = depth_mod[fk][valid]
        dm = dm[np.isfinite(dm)]
        d_obs, d_err = depth_by_filter[fk]
        entry = {"obs": float(d_obs), "err_sym": float(d_err)}
        if dm.size > 0:
            d16, d50, d84 = np.percentile(dm, [16, 50, 84])
            entry.update({"model_p16": float(d16),
                          "model_p50": float(d50),
                          "model_p84": float(d84)})
        depth_summary[fk] = entry

    params = {
        "gaia_id": gaia_id, "tic_id": tic_id,
        "method_dil": "uniform",
        "R_MS_prior_dil": {
            "kind": "uniform",
            "relation": "none (model-free)",
            "min": float(R_MS_MIN_RSUN),
            "max": float(R_MS_MAX_RSUN),
        },
        "T_MS_prior_dil": {"kind": "uniform", "min": float(t_lo), "max": float(t_hi)},
        "T_MS_dil": {"p16": float(t_p16), "p50": float(t_p50), "p84": float(t_p84)},
        "R_MS_dil": {"p16": float(r_p16), "p50": float(r_p50), "p84": float(r_p84)},
        "logg_WD_dil": {"p16": logg_wd_p16, "p50": logg_wd_med, "p84": logg_wd_p84},
        "wd_atmo_type_dil": atmo,
        "depth_summary_dil": depth_summary,
        "depth_model_dil": depth.describe(),
        "depth_noise_dil": "observed depth perturbed once per sample; R_MS by "
                           "deterministic inversion (no additional radius scatter)",
        "N_MC_dil": n_mc,
        "N_valid_dil": n_valid,
        "filters_dil": active_filters,
    }
    params_path = out_dir / "params_dil.json"
    with open(params_path, "w", encoding="utf-8") as fp:
        json.dump(params, fp, indent=2)
    print(f"[DIL-U] Saved params → {params_path}")

    plot_path = out_dir / "plot_dil.pdf"
    _plot_dil_uniform(
        t_ms_s, r_ms_s,
        t_p16, t_p50, t_p84,
        r_p16, r_p50, r_p84,
        f"{depth.ms_model}, log g '{depth.ms_logg_mode}'", plot_path,
    )

    return {
        "T_MS_dil": float(t_p50),
        "R_MS_dil": float(r_p50),
        "logg_WD_dil": logg_wd_med,
        "T_MS_p16_dil": float(t_p16),
        "T_MS_p84_dil": float(t_p84),
        "R_MS_p16_dil": float(r_p16),
        "R_MS_p84_dil": float(r_p84),
        "logg_WD_p16_dil": logg_wd_p16,
        "logg_WD_p84_dil": logg_wd_p84,
        "T_MS_samples_dil": t_ms_s,
        "R_MS_samples_dil": r_ms_s,
        "posterior_samples_dil_path": str(path_samp),
        "posterior_samples_dil_matched_path": str(path_matched),
        "lnL_dil_path": None,
        "params_dil_path": str(params_path),
        "plot_dil_path": str(plot_path),
    }


def _plot_dil_uniform(t_ms, r_ms, t16, t50, t84, r16, r50, r84,
                       logg_ms, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)

    axes[0].hist(t_ms, bins=40, alpha=0.75, color="steelblue")
    axes[0].axvline(t16, color="k", ls="--", lw=1.2, label=f"p16 = {t16:.0f} K")
    axes[0].axvline(t50, color="red", ls="-", lw=1.5, label=f"p50 = {t50:.0f} K")
    axes[0].axvline(t84, color="k", ls="--", lw=1.2, label=f"p84 = {t84:.0f} K")
    axes[0].set_xlabel("T_MS [K]")
    axes[0].set_ylabel("counts")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_title("T_MS  (uniform prior)", fontsize=10)

    axes[1].hist(r_ms, bins=50, alpha=0.75, color="darkorange")
    axes[1].axvline(r16, color="k", ls="--", lw=1.2, label=f"p16 = {r16:.3f}")
    axes[1].axvline(r50, color="red", ls="-", lw=1.5, label=f"p50 = {r50:.3f}")
    axes[1].axvline(r84, color="k", ls="--", lw=1.2, label=f"p84 = {r84:.3f}")
    axes[1].set_xlabel(r"R_MS [$R_\odot$]")
    axes[1].set_ylabel("counts")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].set_title("R_MS  (eclipse-required)", fontsize=10)

    axes[2].scatter(t_ms, r_ms, s=4, alpha=0.12, color="peru", rasterized=True,
                    label="eclipse-required samples")
    # Running median so the ridge is readable on top of the cloud.
    order = np.argsort(t_ms)
    t_s, r_s = t_ms[order], r_ms[order]
    n_bin = max(12, len(t_s) // 200)
    edges = np.linspace(t_s[0], t_s[-1], n_bin + 1)
    t_mid, r_mid = [], []
    for i in range(n_bin):
        m = (t_s >= edges[i]) & (t_s < edges[i + 1])
        if m.sum() >= 8:
            t_mid.append(np.median(t_s[m]))
            r_mid.append(np.median(r_s[m]))
    if t_mid:
        axes[2].plot(t_mid, r_mid, color="k", lw=1.6,
                     label="running median")
    axes[2].set_xlabel("T_MS [K]")
    axes[2].set_ylabel(r"R_MS [$R_\odot$]")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(alpha=0.3)
    axes[2].set_title(f"no BHAC15 — companion: {logg_ms}", fontsize=10)

    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[DIL-U] Saved plot → {out_path}")
