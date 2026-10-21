#!/usr/bin/env python3
"""depth_model.py — Band fluxes and gravities for the eclipse-depth term.

Every eclipse-depth calculation of the pipeline uses this module: the ideal
joint fit (joint_fit_ideal.py), the BHAC15 dilution and the eclipse-radius
diagnostic (dilution_fit.py), and the uniform dilution
(dilution_uniform_fit.py). The depth model itself is unchanged:

    depth = 1 / (1 + q),   q = (R_MS / R_WD)^2 * F_MS,band / F_WD,band

with unreddened surface band fluxes (no extinction, third light, reflection or
partial eclipse; see docs/KNOWN_ISSUES.md).

Companion (M dwarf) — same model and gravity as the SED term
    Model     : the SED reference companion model, `ms_atmosphere`, resolved
                through sed_fit.SED_ATMO_VARIANTS. With the default "phoenix"
                this is the STScI PHOENIX grid of sed_atmosphere_models, read
                from the same cached file as the SED fit (same Teff/logg range,
                solar metallicity, surface F_lambda in erg/s/cm²/Å). The local
                PHOENIX-ACES SpecInt files are no longer used.
    Gravity   : sed_atmosphere_models.companion_logg(ms_logg_mode), clipped to
                the grid's logg range — identical to sed_fit._SedModel._ms_logg.
    Band flux : grid spectra weighted with sed_atmosphere_models.band_weights
                (photon weighting, as in the SED term), or energy weighting
                when photon_counting=False.
    Interpolation: bilinear in ln F over the (Teff, logg) grid nodes. This is
                the SED convention: sed_atmosphere_models.BandPhotometry
                interpolates ln F trilinearly in (Teff, logg, A_V), which at
                A_V = 0 is the same bilinear interpolation.

White dwarf
    Model     : Koester DA spectra (da*.dk.dat.txt in koester_dir), integrated
                over each filter at native resolution with _band_flux, as
                before (photon or energy weighting).
    Gravity   : the SED policy — Delaunay interpolation of the cooling track
                (sed_fit._WDCoolingLogg) with the fallback wd_logg_fixed
                outside the track domain, or wd_logg_fixed when
                wd_logg_mode="fixed" — then clipped to the Koester logg range.
    Interpolation: bilinear in ln F over the (Teff, logg) Koester nodes.

No extrapolation: a temperature or gravity outside the node range returns
NaN (callers reject the sample), as does a point whose bracketing nodes are
missing or non-positive. At an exact node the node value is returned, even
when a neighbouring node is missing.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

import sed_atmosphere_models as sam
from sed_fit import (
    ATMO_GRID_RANGES,
    ATMO_MS_LOGG_FIXED,
    ATMO_MS_LOGG_MODE,
    ATMO_WD_LOGG_FIXED,
    ATMO_WD_LOGG_MODE,
    SED_ATMO_VARIANTS,
    _WDCoolingLogg,
    _resolve_atmo_key,
)


# ══════════════════════════════════════════════════════════════════════════════
# Filter profiles and band integration (moved unchanged from dilution_fit.py)
# ══════════════════════════════════════════════════════════════════════════════

def _load_filter_profile(path, min_A=None, max_A=None):
    """Load ASCII filter (wave_A, transmission). Apply optional wavelength cuts."""
    arr   = np.loadtxt(path, comments="#")
    wave  = np.asarray(arr[:, 0], float)
    trans = np.asarray(arr[:, 1], float)

    ok = np.isfinite(wave) & np.isfinite(trans)
    wave, trans = wave[ok], trans[ok]
    if min_A is not None:
        trans = np.where(wave >= min_A, trans, 0.0)
    if max_A is not None:
        trans = np.where(wave <= max_A, trans, 0.0)
    trans = np.clip(trans, 0.0, None)
    order = np.argsort(wave)
    return wave[order], trans[order]


def _band_flux(wave_A, flux, pw, pt, photon_counting=True):
    """Photon-counting (or energy) band-integrated flux of a native spectrum."""
    t    = np.interp(wave_A, pw, pt, left=0.0, right=0.0)
    good = t > 0
    if good.sum() < 3:
        return np.nan
    x  = wave_A[good]; y = flux[good]; tt = t[good]
    if photon_counting:
        num = np.trapezoid(y * tt * x, x)
        den = np.trapezoid(tt * x, x)
    else:
        num = np.trapezoid(y * tt, x)
        den = np.trapezoid(tt, x)
    return float(num / den) if den > 0 else np.nan


def _grid_band_weights(wave_A, throughput, grid_wave, photon_counting=True):
    """(indices, weights) giving <F> = weights · F[indices] on a model grid.

    Photon weighting reuses sed_atmosphere_models.band_weights, i.e. exactly
    the SED band integration. Energy weighting is the same construction with
    the λ factor removed.
    """
    band = sam.Bandpass(name="depth", wave_AA=np.asarray(wave_A, float),
                        throughput=np.asarray(throughput, float))
    if photon_counting:
        return sam.band_weights(band, grid_wave)
    lo, hi = band.wave_AA[0], band.wave_AA[-1]
    idx = np.where((grid_wave >= lo) & (grid_wave <= hi))[0]
    w = grid_wave[idx]
    t = np.interp(w, band.wave_AA, band.throughput, left=0.0, right=0.0)
    dw = np.empty_like(w)
    dw[1:-1] = 0.5 * (w[2:] - w[:-2])
    dw[0] = 0.5 * (w[1] - w[0])
    dw[-1] = 0.5 * (w[-1] - w[-2])
    den = float(np.sum(t * dw))
    if idx.size < 3 or den <= 0:
        raise ValueError("Filter profile does not overlap the model wavelength grid")
    return idx, t * dw / den


# ══════════════════════════════════════════════════════════════════════════════
# Koester DA models (moved unchanged from dilution_fit.py)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_koester_name(name):
    m = re.match(r"da(?P<t>\d{5})_(?P<g>\d{3})\.dk\.dat\.txt$", name)
    if not m:
        return None
    return float(m.group("t")), float(m.group("g")) / 100.0


def _load_koester_catalog(root):
    rows = []
    for fp in sorted(Path(root).glob("da*.dk.dat.txt")):
        p = _parse_koester_name(fp.name)
        if p is not None:
            rows.append({"path": fp, "teff": p[0], "logg": p[1]})
    if not rows:
        raise FileNotFoundError(f"No Koester models found in {root}")
    df = pd.DataFrame(rows)
    print(f"[DEPTH] Koester catalog: {len(df)} models")
    return df


def _read_koester(path):
    arr = np.loadtxt(path, comments="#")
    return arr[:, 0].astype(float), arr[:, 1].astype(float)


# ══════════════════════════════════════════════════════════════════════════════
# Interpolation and depth inversion
# ══════════════════════════════════════════════════════════════════════════════

def interp_bilinear_lnf(ln_table, x_nodes, y_nodes, x, y):
    """Bilinear interpolation of ln F on a rectangular grid; returns F.

    ln_table : (nx, ny) array of ln F; NaN marks a missing node.
    x, y     : scalars or arrays (broadcast together).

    Returns NaN outside [x_nodes[0], x_nodes[-1]] × [y_nodes[0], y_nodes[-1]]
    (no extrapolation) and where a bracketing node with non-zero weight is
    missing. A node with zero weight is ignored, so exact nodes are recovered
    even next to a missing node. An axis with a single node only accepts that
    exact value.
    """
    x_nodes = np.asarray(x_nodes, float)
    y_nodes = np.asarray(y_nodes, float)
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    out = np.full(x.shape, np.nan)

    inside = (np.isfinite(x) & np.isfinite(y)
              & (x >= x_nodes[0]) & (x <= x_nodes[-1])
              & (y >= y_nodes[0]) & (y <= y_nodes[-1]))
    if not inside.any():
        return out if out.ndim else float(out)

    def _bracket(nodes, v):
        if nodes.size == 1:
            return np.zeros(v.shape, int), np.zeros(v.shape, int), np.zeros(v.shape)
        i = np.clip(np.searchsorted(nodes, v, side="right") - 1, 0, nodes.size - 2)
        t = (v - nodes[i]) / (nodes[i + 1] - nodes[i])
        return i, i + 1, t

    xi, yi = x[inside], y[inside]
    i0, i1, tx = _bracket(x_nodes, xi)
    j0, j1, ty = _bracket(y_nodes, yi)

    total = np.zeros(xi.shape)
    ok = np.ones(xi.shape, bool)
    for ii, jj, w in ((i0, j0, (1 - tx) * (1 - ty)), (i1, j0, tx * (1 - ty)),
                      (i0, j1, (1 - tx) * ty), (i1, j1, tx * ty)):
        val = ln_table[ii, jj]
        used = w > 0
        ok &= ~used | np.isfinite(val)
        total += np.where(used, w * np.where(np.isfinite(val), val, 0.0), 0.0)

    res = np.where(ok, np.exp(total), np.nan)
    out[inside] = res
    return out if out.ndim else float(out)


def eclipse_required_radius(depth, R_WD_rsun, F_WD_band, F_MS_band):
    """Companion radius demanded by an eclipse depth, from dilution alone.

        depth = 1 / (1 + q),   q = (R_MS / R_WD)^2 * F_MS / F_WD

    inverts exactly to

        R_MS = R_WD * sqrt[ (1/depth - 1) * F_WD / F_MS ]

    No mass-radius or radius-temperature relation appears: given the white
    dwarf and the two band fluxes, the measured depth fixes the companion
    radius. NaN where the result is not finite and positive.
    """
    depth = np.asarray(depth, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.asarray(R_WD_rsun, float) * np.sqrt(
            ((1.0 / depth) - 1.0)
            * np.asarray(F_WD_band, float) / np.asarray(F_MS_band, float)
        )
    return np.where(np.isfinite(R) & (R > 0), R, np.nan)


# ══════════════════════════════════════════════════════════════════════════════
# Shared depth-term model
# ══════════════════════════════════════════════════════════════════════════════

class DepthFluxModel:
    """Unreddened surface band fluxes and gravities of both components.

    Parameters
    ----------
    filter_profiles : dict
        {filter_key: (wave_A, transmission)}, already cut (_load_filter_profile).
    koester_dir : str or Path
        Directory of the Koester DA spectra.
    wd_cooling_track : str or Path or None
        Bédard 2020 CSV for wd_logg_mode="cooling_track".
    ms_atmosphere : str
        Companion model of the SED reference fit (main_all.SED_MS_ATMOSPHERE).
    atmo_cache_dir, atmo_grid_ranges : as in sed_fit.run_sed_fit
        Identify the same cached companion grid as the SED term.
    ms_logg_mode, ms_logg_fixed : companion gravity prescription (SED settings).
    wd_logg_mode, wd_logg_fixed : white-dwarf gravity policy (SED settings).
    photon_counting : bool
        Photon (True) or energy (False) weighting in the band integrals.
    """

    def __init__(self, filter_profiles, koester_dir, wd_cooling_track=None,
                 ms_atmosphere="phoenix", atmo_cache_dir=None, atmo_grid_ranges=None,
                 ms_logg_mode=ATMO_MS_LOGG_MODE, ms_logg_fixed=ATMO_MS_LOGG_FIXED,
                 wd_logg_mode=ATMO_WD_LOGG_MODE, wd_logg_fixed=ATMO_WD_LOGG_FIXED,
                 photon_counting=True, verbose=True):
        self.filters = list(filter_profiles)
        self.photon_counting = bool(photon_counting)

        # ── White dwarf: Koester band-flux table, ln F[teff, logg] ────────
        ko = _load_koester_catalog(koester_dir)
        self.wd_teff_nodes = np.unique(ko["teff"].to_numpy(float))
        self.wd_logg_nodes = np.unique(ko["logg"].to_numpy(float))
        shape = (self.wd_teff_nodes.size, self.wd_logg_nodes.size)
        self._wd_ln = {fk: np.full(shape, np.nan) for fk in self.filters}
        for _, row in ko.iterrows():
            i = int(np.searchsorted(self.wd_teff_nodes, row["teff"]))
            j = int(np.searchsorted(self.wd_logg_nodes, row["logg"]))
            wave, flux = _read_koester(row["path"])
            for fk, (pw, pt) in filter_profiles.items():
                f = _band_flux(wave, flux, pw, pt, self.photon_counting)
                if np.isfinite(f) and f > 0:
                    self._wd_ln[fk][i, j] = np.log(f)
        n_missing = int(sum(np.sum(~np.isfinite(t)) for t in self._wd_ln.values()))
        self.wd_teff_bounds = (float(self.wd_teff_nodes[0]), float(self.wd_teff_nodes[-1]))
        self.wd_logg_bounds = (float(self.wd_logg_nodes[0]), float(self.wd_logg_nodes[-1]))

        # ── White-dwarf gravity: SED policy ───────────────────────────────
        self.wd_logg_mode  = str(wd_logg_mode)
        self.wd_logg_fixed = float(wd_logg_fixed)
        self._wd_logg_fn = None
        if self.wd_logg_mode == "cooling_track" and wd_cooling_track is not None:
            self._wd_logg_fn = _WDCoolingLogg(wd_cooling_track, fallback=self.wd_logg_fixed,
                                              verbose=verbose)
        self.wd_logg_policy = (
            f"cooling track {Path(wd_cooling_track).name} (Delaunay), fallback "
            f"{self.wd_logg_fixed:.2f} outside the track, clipped to "
            f"[{self.wd_logg_bounds[0]:.2f}, {self.wd_logg_bounds[1]:.2f}]"
            if self._wd_logg_fn is not None else
            f"fixed {self.wd_logg_fixed:.2f}, clipped to "
            f"[{self.wd_logg_bounds[0]:.2f}, {self.wd_logg_bounds[1]:.2f}]")

        # ── Companion: SED reference model ────────────────────────────────
        self.ms_key   = _resolve_atmo_key(ms_atmosphere)
        self.ms_model = SED_ATMO_VARIANTS[self.ms_key]["ms_model"]
        self.ms_label = SED_ATMO_VARIANTS[self.ms_key]["label"]
        self.ms_logg_mode  = str(ms_logg_mode)
        self.ms_logg_fixed = float(ms_logg_fixed)
        wave, _ = sam._log_wave_grid()
        self._ms_blackbody = self.ms_model in ("blackbody", "blackbody-exact")
        if self._ms_blackbody:
            # Blackbody companion: Planck evaluated exactly at each Teff on the
            # common wavelength grid, no gravity, no Teff limit of its own.
            self._ms_wave = wave
            self._ms_weights = {fk: _grid_band_weights(pw, pt, wave, self.photon_counting)
                                for fk, (pw, pt) in filter_profiles.items()}
            self.ms_teff_bounds = (0.0, np.inf)
            self.ms_logg_bounds = (0.0, 0.0)
            self.ms_grid_label = "exact Planck function"
        else:
            ranges = dict(ATMO_GRID_RANGES.get(self.ms_model, {}))
            ranges.update((atmo_grid_ranges or {}).get(self.ms_model, {}))
            kw = {} if atmo_cache_dir is None else {"cache_dir": Path(atmo_cache_dir)}
            grid = sam.SpectralGrid.load(
                source_key=self.ms_model,
                teff_range=ranges.get("teff", (2300.0, 5900.0)),
                logg_range=ranges.get("logg", (4.0, 5.5)),
                verbose=verbose, **kw)
            self.ms_teff_nodes = np.asarray(grid.teff, float)
            self.ms_logg_nodes = np.asarray(grid.logg, float)
            self._ms_ln = {}
            for fk, (pw, pt) in filter_profiles.items():
                idx, w = _grid_band_weights(pw, pt, grid.wave_AA, self.photon_counting)
                band = np.asarray(grid.flux[:, :, idx], dtype=float) @ w
                with np.errstate(divide="ignore", invalid="ignore"):
                    self._ms_ln[fk] = np.where(np.isfinite(band) & (band > 0),
                                               np.log(band), np.nan)
            self.ms_teff_bounds = grid.teff_bounds
            self.ms_logg_bounds = grid.logg_bounds
            self.ms_grid_label = (f"{self.ms_model} grid, Teff "
                                  f"{self.ms_teff_bounds[0]:.0f}-{self.ms_teff_bounds[1]:.0f} K, "
                                  f"logg {self.ms_logg_bounds[0]:.1f}-{self.ms_logg_bounds[1]:.1f}")

        if verbose:
            print(f"[DEPTH] Companion: {self.ms_label} ({self.ms_grid_label}); "
                  f"logg prescription '{self.ms_logg_mode}'")
            print(f"[DEPTH] White dwarf: Koester DA, {shape[0]} Teff x {shape[1]} logg nodes"
                  f"{'' if not n_missing else f', {n_missing} missing band fluxes'}; "
                  f"logg: {self.wd_logg_policy}")
            print(f"[DEPTH] Band fluxes interpolated bilinearly in ln F; "
                  f"{'photon' if self.photon_counting else 'energy'} weighting; no reddening")

    # ── Gravities ─────────────────────────────────────────────────────────
    def wd_logg(self, t_wd, r_wd):
        """White-dwarf logg (SED policy), clipped to the Koester logg range."""
        t = np.atleast_1d(np.asarray(t_wd, float))
        r = np.broadcast_to(np.asarray(r_wd, float), t.shape)
        if self._wd_logg_fn is None:
            raw = np.full(t.shape, self.wd_logg_fixed)
        else:
            raw = np.array([self._wd_logg_fn(a, b) if np.isfinite(a) and np.isfinite(b) and b > 0
                            else np.nan for a, b in zip(t, r)])
        out = np.clip(raw, *self.wd_logg_bounds)
        return out if np.ndim(t_wd) else float(out[0])

    def ms_logg(self, t_ms, r_ms):
        """Companion logg (SED prescription), clipped to the companion grid."""
        if self._ms_blackbody:
            return np.zeros(np.shape(t_ms)) if np.ndim(t_ms) else 0.0
        raw = sam.companion_logg(t_ms, r_ms, mode=self.ms_logg_mode, fixed=self.ms_logg_fixed)
        out = np.clip(raw, *self.ms_logg_bounds)
        return out if np.ndim(out) else float(out)

    # ── Band fluxes [erg/s/cm²/Å, surface, unreddened] ───────────────────
    def wd_band_flux(self, fk, t_wd, logg_wd):
        return interp_bilinear_lnf(self._wd_ln[fk], self.wd_teff_nodes,
                                   self.wd_logg_nodes, t_wd, logg_wd)

    def ms_band_flux(self, fk, t_ms, logg_ms):
        if self._ms_blackbody:
            t = np.atleast_1d(np.asarray(t_ms, float))
            idx, w = self._ms_weights[fk]
            out = np.array([float(sam.blackbody_flux_lambda(self._ms_wave[idx], tt) @ w)
                            if np.isfinite(tt) and tt > 0 else np.nan for tt in t])
            return out if np.ndim(t_ms) else float(out[0])
        return interp_bilinear_lnf(self._ms_ln[fk], self.ms_teff_nodes,
                                   self.ms_logg_nodes, t_ms, logg_ms)

    # ── Depth and its inversion ───────────────────────────────────────────
    def depth(self, fk, r_wd, f_wd, r_ms, f_ms):
        """Model depth 1/(1+q) from radii [R_sun] and band fluxes."""
        with np.errstate(invalid="ignore", divide="ignore"):
            q = (np.asarray(r_ms, float) / np.asarray(r_wd, float)) ** 2 \
                * np.asarray(f_ms, float) / np.asarray(f_wd, float)
            return 1.0 / (1.0 + q)

    def required_radius(self, fk, depth, r_wd, f_wd, t_ms, n_iter=6):
        """R_MS reproducing `depth` at t_ms, with the shared companion gravity.

        When the gravity depends on the radius (ms_logg_mode="from_radius")
        the inversion is iterated from the empirical main-sequence radius;
        otherwise a single evaluation is exact. Returns NaN where the companion
        band flux is undefined (outside the grid).
        """
        t = np.asarray(t_ms, float)
        if self._ms_blackbody or self.ms_logg_mode != "from_radius":
            g = self.ms_logg(t, None)
            return eclipse_required_radius(depth, r_wd, f_wd, self.ms_band_flux(fk, t, g))
        r = np.asarray(sam.radius_main_sequence(t), float)
        for _ in range(int(n_iter)):
            g = self.ms_logg(t, r)
            r_new = eclipse_required_radius(depth, r_wd, f_wd, self.ms_band_flux(fk, t, g))
            r = np.where(np.isfinite(r_new), r_new, np.nan)
        return r

    def describe(self):
        """Settings recorded in the stages' JSON outputs."""
        return {
            "ms_model": self.ms_model,
            "ms_label": self.ms_label,
            "ms_grid": self.ms_grid_label,
            "ms_logg_mode": self.ms_logg_mode,
            "ms_logg_fixed": self.ms_logg_fixed,
            "wd_model": "Koester DA",
            "wd_logg_policy": self.wd_logg_policy,
            "interpolation": "bilinear in ln F over (Teff, logg) nodes; no extrapolation",
            "band_weighting": "photon" if self.photon_counting else "energy",
            "reddening": "none",
        }
