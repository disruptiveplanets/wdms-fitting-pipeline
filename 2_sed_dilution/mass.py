#!/usr/bin/env python3
"""mass.py — Stellar masses, white-dwarf core composition and orbital configuration.

The joint fit measures (T_WD, R_WD, T_MS, R_MS, d, A_V, ϖ) but never a mass.
This module turns those posteriors into masses by folding each posterior sample
through a mass-radius relation, so the reported mass uncertainties carry the
full (and strongly non-Gaussian) shape of the fitted radii.

Method — following Brown et al. 2022, MNRAS 513, 3050
-----------------------------------------------------
White dwarf
    M_WD is read off cooling tracks at the measured (T_WD, R_WD). Two core
    compositions are evaluated independently:

        CO  Bédard et al. 2020 (Montreal), thick hydrogen layer q_H = 1e-4
        He  Althaus et al. 2013 (La Plata), tabulated by Camisassa et al.

    Brown used Fontaine et al. 2001 for CO and Panei et al. 2007 for He; both
    are the direct predecessors of the grids above, computed by the same two
    groups. Thick hydrogen layers are kept because they describe the majority
    of PCEB white dwarfs with measured layer masses (Parsons et al. 2017).

    A sample only receives a mass if (T_WD, R_WD) falls inside the domain
    actually covered by that core composition. Hot compact white dwarfs lie
    outside the He-core domain entirely — no He-core model is simultaneously
    that hot and that small — which is what makes the core diagnosis a real
    measurement rather than a restatement of the assumed relation.

M dwarf
    M_MS is obtained by inverting the semi-empirical mass-radius relation of
    Brown et al. 2022 (their table 1), a fifth-order polynomial calibrated on
    15 279 M dwarfs with Gaia parallaxes and 2MASS radii, with masses assigned
    through the K_abs–M relation of Mann et al. 2019. Being empirical it
    absorbs the 5-15 per cent radius inflation of M dwarfs relative to
    theoretical tracks. Below M = 0.121 M_sun, where the semi-empirical fit
    crosses the models, the 1 Gyr BHAC15 isochrone takes over.

Uncertainties
    Statistical errors come from the posterior itself. On top of them Brown
    adds a floor for the relations: 5 per cent on the secondary mass (3.5 per
    cent intrinsic scatter of the M dwarf relation, ~2 per cent from Mann
    et al. 2019, ~2 per cent systematic offset) and 1 per cent on the white
    dwarf mass. Both are reported separately from the statistical width.

Orbital configuration
    Given an orbital period, Kepler's third law gives the separation, and the
    Eggleton (1983) approximation gives the Roche lobe of each component. The
    radial-velocity semi-amplitudes assume i = 90 deg, which costs nothing here
    since every target is eclipsing.

Inputs
    The joint posterior (first four columns T_WD [K], R_WD [R_sun], T_MS [K],
    R_MS [R_sun]), the Bédard 2020 DA table (CO cores), the La Plata DA table
    (He cores), the BHAC15 tracks, and optionally the orbital period [d].
    main_all.WD_MASS_TRACK chooses the adopted white-dwarf mass: "auto"
    (Brown et al. rule) or "co" (always the CO-core mass).

Outputs (<output_base>/TIC{tic_id}_Gaia{gaia_id}/mass/, plus
mass_summary.csv for the whole run, written by main_all.py to
<WORKSPACE>/SED-dilution summaries/)
-------------------------------------------------
    posterior_samples_mass.npy  [N, 5]  M_WD_CO M_WD_He M_MS a_Rsun f_Roche_MS
    params_mass.json            p16/p50/p84, core diagnosis, orbital summary
    plot_mass.pdf               core diagnostic, mass posteriors, Roche filling

Variable naming: all exported keys carry the _mass suffix.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd
from scipy.interpolate import LinearNDInterpolator

from wdms_common import M_SUN_G

# ── Physical constants (CGS) ──────────────────────────────────────────────────
_G_CGS    = 6.67430e-8     # cm³/(g·s²)
_M_SUN_G  = M_SUN_G        # g, astropy.constants.M_sun (wdms_common)
_R_SUN_CM = 6.957e10       # cm
_DAY_S    = 86400.0        # s

# ── Brown et al. 2022, table 1 ────────────────────────────────────────────────
# R/R_sun = sum_{n=0}^{5} a_n (M/M_sun)^n, coefficients ordered a_0 … a_5.
BROWN2022_COEFFS = (-0.124, 3.53, -14.9, 41.1, -54.1, 27.4)
BROWN2022_SIGMA  = ( 0.010, 0.16,   0.9,  2.7,   3.6,   1.8)

# Below this mass the semi-empirical fit crosses the theoretical tracks and
# Brown et al. switch to the 1 Gyr BHAC15 isochrone. Above the upper bound the
# calibration sample is truncated in temperature and the fit is not usable.
BROWN2022_M_MIN = 0.121
BROWN2022_M_MAX = 0.65

# Standard deviation of the fractional residuals of the relation taken as a
# function of radius (Brown et al. 2022, sect. 2.1). Used to widen the radius
# prior so that the fit is not told the relation is exact.
BROWN2022_RADIUS_SCATTER = 0.035

# Relation floors added in quadrature to the statistical width (Brown, sect. 2.1).
# Defaults when this module is used alone; main_all.py sets its own values
# (see its "Fit settings" section, function apply_fit_settings).
MS_MASS_REL_SYST = 0.05    # secondary
WD_MASS_REL_SYST = 0.01    # primary

# ── Core-composition diagnosis ────────────────────────────────────────────────
# Single-star evolution cannot strip a red giant below the helium flash, so a
# white dwarf under this mass in a close binary must have a helium core.
HE_CORE_MASS_MAX = 0.45
# Prada Moroni & Straniero (2009) reach CO cores down to here with an extreme
# progenitor; below it a CO core is not a credible option.
CO_CORE_MASS_MIN = 0.33
# Posterior probability required before a composition is declared, rather than
# left ambiguous. Deliberately strict: with 30 per cent mass uncertainties most
# of this sample cannot be classified, and saying so is the honest answer.
CORE_PROB_MIN = 0.85

# ── Brown et al. decision rule ────────────────────────────────────────────────
# Brown et al. (2022, 2023) fit each system with CO-core models first, repeat
# the fit with He-core models when the CO mass lands below BROWN_CO_TRIAL_MASS,
# and keep CO only if that second fit is pushed against the upper mass limit of
# the He grid. The rule always answers CO or He, which is what makes their
# published tables directly comparable, and that is why it is reproduced here.
#
# Note that the deciding limit belongs to the models, not to the physics: Panei
# et al. (2007), the He grid Brown uses, stops at 0.5 M_sun. A He grid with a
# lower ceiling makes the test stricter than Brown's and manufactures CO cores,
# so classify_core_brown reports the ceiling it actually used.
BROWN_CO_TRIAL_MASS = 0.50   # M_sun, below which the He models get their turn
BROWN_HE_PIN_TOL    = 0.01   # M_sun from the ceiling still counted as pinned
BROWN_HE_MIN_FRAC   = 0.50   # posterior fraction needing a He solution at all


# ══════════════════════════════════════════════════════════════════════════════
# M dwarf mass-radius relation
# ══════════════════════════════════════════════════════════════════════════════

def brown2022_radius(mass_Msun, coeffs=BROWN2022_COEFFS):
    """Semi-empirical M dwarf radius [R_sun] from mass [M_sun]."""
    m = np.asarray(mass_Msun, dtype=float)
    r = np.zeros_like(m)
    for n, a in enumerate(coeffs):
        r += a * m**n
    return r


def bhac15_isochrone(csv_path, log_age_yr=9.0):
    """Interpolate the BHAC15 evolutionary tracks onto a single isochrone.

    Returns (mass, Teff, radius) arrays sorted by mass. The distributed file is
    a set of tracks, not isochrones, so each mass is interpolated in log-age.
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"M/Ms": "mass", "log t(yr)": "logt",
                            "Teff": "teff", "R/Rs": "radius"})
    for col in ("mass", "logt", "teff", "radius"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=("mass", "logt", "teff", "radius"))

    rows = []
    for mass, track in df.groupby("mass"):
        track = track.sort_values("logt")
        if track["logt"].min() <= log_age_yr <= track["logt"].max():
            rows.append((
                float(mass),
                float(np.interp(log_age_yr, track["logt"], track["teff"])),
                float(np.interp(log_age_yr, track["logt"], track["radius"])),
            ))
    if len(rows) < 5:
        raise ValueError(
            f"BHAC15: only {len(rows)} masses reach log(t/yr) = {log_age_yr} in {csv_path}"
        )
    arr = np.array(sorted(rows), dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2]


class MDwarfRelation:
    """Invertible M dwarf mass-radius relation.

    With relation="brown2022" the semi-empirical fit of Brown et al. 2022 is
    extended by BHAC15 at low mass. The two branches are joined at
    BROWN2022_M_MIN, where they agree to better than a per cent, giving a single
    monotonic curve that can be inverted.

    With relation="bhac15" the theoretical isochrone is used on its own, which
    is the comparison point: the difference between the two masses derived from
    the same radius is the 5-15 per cent inflation M dwarfs show relative to the
    models, read as a mass rather than a radius.
    """

    def __init__(self, bhac15_csv, log_age_yr=9.0, n_grid=2000,
                 relation="brown2022"):
        relation = str(relation).strip().lower()
        if relation not in {"bhac15", "brown2022"}:
            raise ValueError(
                f"relation must be 'bhac15' or 'brown2022', got {relation!r}"
            )

        m_bhac, _, r_bhac = bhac15_isochrone(bhac15_csv, log_age_yr)

        if relation == "bhac15":
            mass, radius = m_bhac, r_bhac
        else:
            low = m_bhac < BROWN2022_M_MIN
            if low.sum() < 2:
                raise ValueError("BHAC15 isochrone does not extend below the Brown 2022 floor")

            m_low = np.linspace(float(m_bhac[low].min()), BROWN2022_M_MIN, n_grid // 4)
            r_low = np.interp(m_low, m_bhac, r_bhac)

            m_high = np.linspace(BROWN2022_M_MIN, BROWN2022_M_MAX, n_grid)
            r_high = brown2022_radius(m_high)

            # Remove the seam duplicate so the concatenated curve has no
            # repeated node.
            mass   = np.concatenate([m_low[:-1], m_high])
            radius = np.concatenate([r_low[:-1], r_high])

        # Enforce strict monotonicity so that the inverse interpolation is
        # single-valued.
        keep = np.concatenate([[True], np.diff(radius) > 0])

        self.relation     = relation
        self.mass_grid    = mass[keep]
        self.radius_grid  = radius[keep]
        # Kept for plotting: the theoretical relation the semi-empirical one
        # replaces, so the amount of inflation being applied stays visible.
        self.bhac_mass    = m_bhac
        self.bhac_radius  = r_bhac
        self.mass_min     = float(self.mass_grid[0])
        self.mass_max     = float(self.mass_grid[-1])
        self.radius_min   = float(self.radius_grid[0])
        self.radius_max   = float(self.radius_grid[-1])
        self.log_age_yr   = float(log_age_yr)
        age_gyr           = 10**(self.log_age_yr - 9)
        if relation == "bhac15":
            self.label         = f"BHAC15 {age_gyr:.0f} Gyr isochrone"
            self.seam_mass     = float("nan")
            self.seam_mismatch = 0.0
        else:
            self.label = (f"Brown et al. 2022 above {BROWN2022_M_MIN} M_sun, "
                          f"BHAC15 {age_gyr:.0f} Gyr below")
            self.seam_mass = BROWN2022_M_MIN
            self.seam_mismatch = float(
                abs(brown2022_radius(BROWN2022_M_MIN) - np.interp(BROWN2022_M_MIN, m_bhac, r_bhac))
                / brown2022_radius(BROWN2022_M_MIN)
            )

    def radius(self, mass_Msun):
        """Radius [R_sun] at a given mass, NaN outside the calibrated range."""
        m = np.asarray(mass_Msun, dtype=float)
        r = np.interp(m, self.mass_grid, self.radius_grid,
                      left=np.nan, right=np.nan)
        return r

    def mass(self, radius_Rsun):
        """Mass [M_sun] at a given radius, NaN outside the calibrated range."""
        r = np.asarray(radius_Rsun, dtype=float)
        m = np.interp(r, self.radius_grid, self.mass_grid,
                      left=np.nan, right=np.nan)
        return m

    def inflation_factor(self, mass_Msun):
        """Semi-empirical over theoretical radius at the same mass.

        Multiplying a BHAC15 radius by this factor applies the observed
        inflation without disturbing anything else the theoretical grid
        provides. It is 1 by construction below the seam, where the relation is
        the BHAC15 isochrone itself, and 1 again outside the calibrated range
        so that no radius is silently extrapolated.
        """
        m = np.asarray(mass_Msun, dtype=float)
        if self.relation == "bhac15":
            return np.ones_like(m)
        r_semi   = np.interp(m, self.mass_grid, self.radius_grid,
                             left=np.nan, right=np.nan)
        r_theory = np.interp(m, self.bhac_mass, self.bhac_radius,
                             left=np.nan, right=np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            factor = r_semi / r_theory
        return np.where(np.isfinite(factor) & (r_theory > 0), factor, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# White dwarf cooling grids
# ══════════════════════════════════════════════════════════════════════════════

class WDCoolingGrid:
    """Mass and cooling age interpolated over the (Teff, R) plane of a track set.

    Interpolation is linear on a Delaunay triangulation of (log Teff, log R),
    which returns NaN outside the convex hull of the models. That NaN is the
    useful part: it says no model of this core composition reproduces the
    measured temperature and radius together.
    """

    def __init__(self, teff_K, radius_Rsun, mass_Msun,
                 cooling_age_Gyr=None, label=""):
        teff   = np.asarray(teff_K,      dtype=float)
        radius = np.asarray(radius_Rsun, dtype=float)
        mass   = np.asarray(mass_Msun,   dtype=float)

        good = (np.isfinite(teff) & np.isfinite(radius) & np.isfinite(mass)
                & (teff > 0) & (radius > 0) & (mass > 0))
        if cooling_age_Gyr is not None:
            age = np.asarray(cooling_age_Gyr, dtype=float)
            good &= np.isfinite(age)
        if good.sum() < 50:
            raise ValueError(f"WD grid '{label}': only {good.sum()} usable points")

        x = np.column_stack([np.log10(teff[good]), np.log10(radius[good])])
        # Duplicate coordinates make the triangulation degenerate.
        _, uniq = np.unique(np.round(x, 9), axis=0, return_index=True)
        x = x[uniq]

        self.label       = label
        self.teff_K      = teff[good][uniq]
        self.radius_Rsun = radius[good][uniq]
        self.mass_Msun   = mass[good][uniq]

        self._interp_mass = LinearNDInterpolator(x, self.mass_Msun)
        if cooling_age_Gyr is not None:
            self.age_Gyr = age[good][uniq]
            self._interp_age = LinearNDInterpolator(x, self.age_Gyr)
        else:
            self.age_Gyr = None
            self._interp_age = None

    @property
    def mass_max(self):
        """Highest mass in the track set, i.e. the ceiling a fit can be pinned to."""
        return float(self.mass_Msun.max())

    @property
    def teff_max(self):
        """Hottest model in the track set."""
        return float(self.teff_K.max())

    def _coords(self, teff_K, radius_Rsun):
        teff   = np.atleast_1d(np.asarray(teff_K,      dtype=float))
        radius = np.atleast_1d(np.asarray(radius_Rsun, dtype=float))
        ok = (teff > 0) & (radius > 0) & np.isfinite(teff) & np.isfinite(radius)
        x = np.full((teff.size, 2), np.nan)
        x[ok, 0] = np.log10(teff[ok])
        x[ok, 1] = np.log10(radius[ok])
        return x

    def mass(self, teff_K, radius_Rsun):
        """Mass [M_sun] at (Teff, R), NaN where no model of this core fits."""
        return self._interp_mass(self._coords(teff_K, radius_Rsun))

    def cooling_age(self, teff_K, radius_Rsun):
        """Cooling age [Gyr] at (Teff, R), NaN outside the model domain."""
        if self._interp_age is None:
            return np.full(np.atleast_1d(teff_K).shape, np.nan)
        return self._interp_age(self._coords(teff_K, radius_Rsun))

    def radius_envelope(self, teff_K, rel_tol=0.03):
        """Smallest and largest model radius near a temperature.

        Reported alongside a rejected core composition so the reason is legible:
        a 46 000 K helium-core model cannot be smaller than about 0.03 R_sun.
        """
        teff = float(teff_K)
        near = np.abs(self.teff_K - teff) <= rel_tol * teff
        if not near.any():
            return np.nan, np.nan
        return float(self.radius_Rsun[near].min()), float(self.radius_Rsun[near].max())


def load_bedard_grid(csv_path, hypothesis="thick"):
    """CO-core grid from a Bédard et al. 2020 table (Montreal cooling sequences)."""
    df = pd.read_csv(csv_path)
    if "hypothesis" in df.columns and hypothesis is not None:
        sub = df[df["hypothesis"].astype(str).str.strip().str.lower() == hypothesis]
        if len(sub) > 50:
            df = sub

    age = df["CoolingAgeGyr"] if "CoolingAgeGyr" in df.columns else None
    grid = WDCoolingGrid(
        teff_K          = df["Teff"],
        radius_Rsun     = df["RadiusRsun"],
        mass_Msun       = df["MassMsun"],
        cooling_age_Gyr = age,
        label           = f"CO (Bédard 2020, {hypothesis} H layer)",
    )
    print(f"[MASS] {grid.label}: {grid.mass_Msun.size} points, "
          f"M = {grid.mass_Msun.min():.3f}–{grid.mass_Msun.max():.3f} M_sun")
    return grid


def load_laplata_grid(dat_path, core="He"):
    """Grid of one core composition from the La Plata DA table.

    Columns are log g, log Teff, M, cooling age, R, core, then photometry. He
    cores come from Althaus et al. 2013, CO from Camisassa et al. 2016 and ONe
    from Camisassa et al. 2019.
    """
    rows = []
    with open(dat_path, encoding="utf-8") as fp:
        for line in fp:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[5] != core:
                continue
            rows.append((10.0**float(fields[1]), float(fields[4]),
                         float(fields[2]), float(fields[3])))
    if not rows:
        raise ValueError(f"La Plata table {dat_path}: no '{core}' entries")

    arr = np.array(rows, dtype=float)
    source = {"He": "Althaus 2013", "CO": "Camisassa 2016", "ONe": "Camisassa 2019"}
    grid = WDCoolingGrid(
        teff_K          = arr[:, 0],
        radius_Rsun     = arr[:, 1],
        mass_Msun       = arr[:, 2],
        cooling_age_Gyr = arr[:, 3],
        label           = f"{core} ({source.get(core, 'La Plata')})",
    )
    print(f"[MASS] {grid.label}: {grid.mass_Msun.size} points, "
          f"M = {grid.mass_Msun.min():.3f}–{grid.mass_Msun.max():.3f} M_sun, "
          f"Teff ≤ {grid.teff_K.max():.0f} K")
    return grid


# ══════════════════════════════════════════════════════════════════════════════
# Orbital configuration
# ══════════════════════════════════════════════════════════════════════════════

def semi_major_axis(period_d, m1_Msun, m2_Msun):
    """Orbital separation [R_sun] from Kepler's third law."""
    p    = np.asarray(period_d, dtype=float) * _DAY_S
    mtot = (np.asarray(m1_Msun, dtype=float) + np.asarray(m2_Msun, dtype=float)) * _M_SUN_G
    a_cm = (_G_CGS * mtot * p**2 / (4.0 * np.pi**2))**(1.0 / 3.0)
    return a_cm / _R_SUN_CM


def roche_lobe_fraction(q):
    """Eggleton (1983) Roche-lobe radius in units of the separation.

    q is the mass ratio of the star whose lobe is wanted to its companion.
    """
    q = np.asarray(q, dtype=float)
    q13 = q**(1.0 / 3.0)
    q23 = q13 * q13
    return 0.49 * q23 / (0.6 * q23 + np.log1p(q13))


def orbital_configuration(period_d, m_wd, m_ms, r_ms):
    """Separation, mass ratio, Roche lobes and RV semi-amplitudes at i = 90 deg.

    The inclination assumption is harmless for an eclipsing sample and makes the
    semi-amplitudes directly usable when planning spectroscopic follow-up.
    """
    m_wd = np.asarray(m_wd, dtype=float)
    m_ms = np.asarray(m_ms, dtype=float)
    r_ms = np.asarray(r_ms, dtype=float)

    a = semi_major_axis(period_d, m_wd, m_ms)
    q = m_ms / m_wd

    rl_ms = roche_lobe_fraction(q) * a
    rl_wd = roche_lobe_fraction(1.0 / q) * a

    v_orb = 2.0 * np.pi * a * _R_SUN_CM / (float(period_d) * _DAY_S)  # cm/s
    k_wd  = v_orb * m_ms / (m_wd + m_ms) / 1.0e5                      # km/s
    k_ms  = v_orb * m_wd / (m_wd + m_ms) / 1.0e5

    return {
        "a_Rsun":        a,
        "q_MS_over_WD":  q,
        "RL_MS_Rsun":    rl_ms,
        "RL_WD_Rsun":    rl_wd,
        "f_Roche_MS":    r_ms / rl_ms,
        "K_WD_kms":      k_wd,
        "K_MS_kms":      k_ms,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Posterior propagation
# ══════════════════════════════════════════════════════════════════════════════

def _summary(values, rel_syst=None):
    """p16/p50/p84 of a sample, optionally with a relative systematic floor."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 10:
        return None
    p16, p50, p84 = (float(x) for x in np.percentile(v, [16, 50, 84]))
    out = {
        "p16": p16, "p50": p50, "p84": p84,
        "sigma_stat": 0.5 * (p84 - p16),
        "n_valid": int(v.size),
    }
    if rel_syst:
        sigma_syst = float(rel_syst) * abs(p50)
        out["sigma_syst"]  = sigma_syst
        out["sigma_total"] = float(np.hypot(out["sigma_stat"], sigma_syst))
    return out


def diagnose_core(m_co_samples, m_he_samples):
    """Classify the core composition one posterior sample at a time.

    Each sample is a (T_WD, R_WD) pair, and each core composition either has a
    model that reproduces it or does not. That test is the non-circular part:
    it uses temperature and radius jointly, the way Parsons et al. (2017)
    separate the two families, instead of reading a mass off an assumed
    composition and then cutting on it.

    A composition is allowed for a sample when a model of that family matches
    and the resulting mass is physically reachable for that family: a CO core
    cannot form below CO_CORE_MASS_MIN, and a helium core cannot exceed the
    mass at which the helium flash would have occurred.

    Returns the label, a readable reason, and the four posterior fractions.
    """
    m_co = np.asarray(m_co_samples, dtype=float)
    m_he = np.asarray(m_he_samples, dtype=float)

    co_ok = np.isfinite(m_co) & (m_co >= CO_CORE_MASS_MIN)
    he_ok = np.isfinite(m_he) & (m_he <= HE_CORE_MASS_MAX)

    fractions = {
        "CO_only": float(np.mean(co_ok & ~he_ok)),
        "He_only": float(np.mean(he_ok & ~co_ok)),
        "both":    float(np.mean(co_ok & he_ok)),
        "neither": float(np.mean(~co_ok & ~he_ok)),
    }

    if fractions["CO_only"] >= CORE_PROB_MIN:
        return "CO", (f"{100*fractions['CO_only']:.0f} per cent of the posterior is "
                      f"reproduced only by CO-core models"), fractions
    if fractions["He_only"] >= CORE_PROB_MIN:
        return "He", (f"{100*fractions['He_only']:.0f} per cent of the posterior is "
                      f"reproduced only by He-core models"), fractions
    if fractions["neither"] >= CORE_PROB_MIN:
        return "undetermined", "outside the domain of both model families", fractions

    leaning = "CO" if fractions["CO_only"] > fractions["He_only"] else "He"
    return "ambiguous", (f"both compositions remain possible, leaning {leaning} "
                         f"(CO only {100*fractions['CO_only']:.0f} per cent, "
                         f"He only {100*fractions['He_only']:.0f} per cent)"), fractions


def classify_core_brown(m_co_samples, m_he_samples, he_grid, teff_K):
    """Binary He/CO label following the decision rule of Brown et al. (2022, 2023).

    A CO core is the default. When the CO mass falls below BROWN_CO_TRIAL_MASS
    the helium models get their turn, and helium is adopted unless they cannot
    reach the measurement.

    Brown refit the light curve with He models and asked whether the fit was
    driven against the upper mass limit of the grid. Here the fit is not
    repeated, because (Teff, R) was measured without assuming a composition, so
    the equivalent question is whether the He models can reproduce that
    (Teff, R) at all. They fail when the star is more compact than any He model
    of the same temperature, which is the same statement as being pinned at the
    mass ceiling.

    Samples hotter than the He grid are excluded from that count rather than
    held against helium: a missing model is not a failed fit, and folding the
    two together would let the extent of the grid decide the composition.

    Unlike `diagnose_core` this never answers "ambiguous". That is the point:
    it is what makes the result comparable with the published samples. The two
    are meant to be reported side by side, the label in the table and the
    posterior fractions in the text.
    """
    m_co = np.asarray(m_co_samples, dtype=float)
    m_he = np.asarray(m_he_samples, dtype=float)
    teff = np.asarray(teff_K, dtype=float)

    co_p50 = float(np.nanmedian(m_co)) if np.isfinite(m_co).any() else float("nan")
    he_p50 = float(np.nanmedian(m_he)) if np.isfinite(m_he).any() else float("nan")

    testable = teff <= he_grid.teff_max
    too_hot  = float(np.mean(~testable))
    coverage = float(np.mean(np.isfinite(m_he[testable]))) if testable.any() else 0.0
    caveat   = ("" if too_hot < 0.01 else
                f"; a further {100*too_hot:.0f} per cent of the posterior is hotter "
                f"than the He grid and could not be tested")

    if not np.isfinite(co_p50):
        return "CO", ("neither family reproduces this (Teff, R); reported as CO by "
                      "default, and the label should not be trusted")
    if co_p50 >= BROWN_CO_TRIAL_MASS:
        return "CO", (f"the CO-core mass {co_p50:.3f} M_sun is at or above "
                      f"{BROWN_CO_TRIAL_MASS:.2f} M_sun, so the He models are not tried")
    if coverage < BROWN_HE_MIN_FRAC:
        return "CO", (f"the CO-core mass {co_p50:.3f} M_sun is low, but the star is more "
                      f"compact than any He model over {100*(1-coverage):.0f} per cent of "
                      f"the testable posterior, the equivalent of a He fit pinned at its "
                      f"mass ceiling{caveat}")
    if he_p50 >= he_grid.mass_max - BROWN_HE_PIN_TOL:
        return "CO", (f"the He-core mass {he_p50:.3f} M_sun sits at the ceiling of the "
                      f"He grid ({he_grid.mass_max:.3f} M_sun), so He cannot "
                      f"accommodate it{caveat}")

    support = ("" if coverage > 0.99 else
               f"; the He mass rests on {100*coverage:.0f} per cent of the testable "
               f"posterior")
    return "He", (f"the CO-core mass {co_p50:.3f} M_sun is below "
                  f"{BROWN_CO_TRIAL_MASS:.2f} M_sun and the He models reach "
                  f"{he_p50:.3f} M_sun without hitting their "
                  f"{he_grid.mass_max:.3f} M_sun ceiling{support}{caveat}")


def derive_masses(samples, co_grid, he_grid, ms_relation,
                  period_d=None, rng=None, max_samples=20000,
                  wd_mass_track="auto"):
    """Fold a joint posterior through the mass-radius relations, sample by sample.

    `samples` is the [N, 7] array written by joint_fit_ideal: T_WD, R_WD, T_MS,
    R_MS, d, A_V, ϖ (the [N, 6] legacy array also works: only the first four
    columns are read).

    wd_mass_track
        "auto"  adopt CO or He according to the Brown et al. rule (default)
        "co"    always adopt the Bédard 2020 CO-core mass, regardless of that
                classification. He masses stay in the diagnostic columns.
    """
    wd_mass_track = str(wd_mass_track).strip().lower()
    if wd_mass_track not in {"auto", "co"}:
        raise ValueError(
            f"wd_mass_track must be 'auto' or 'co', got {wd_mass_track!r}"
        )
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"Expected a [N, >=4] posterior array, got {arr.shape}")

    if max_samples and arr.shape[0] > max_samples:
        rng = np.random.default_rng() if rng is None else rng
        arr = arr[rng.choice(arr.shape[0], size=max_samples, replace=False)]

    t_wd, r_wd, _, r_ms = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    m_co = np.asarray(co_grid.mass(t_wd, r_wd), dtype=float)
    m_he = np.asarray(he_grid.mass(t_wd, r_wd), dtype=float)
    m_ms = np.asarray(ms_relation.mass(r_ms),   dtype=float)

    frac_co = float(np.mean(np.isfinite(m_co)))
    frac_he = float(np.mean(np.isfinite(m_he)))
    frac_ms = float(np.mean(np.isfinite(m_ms)))
    # Brown et al. hand over to BHAC15 below 0.121 M_sun; tracking how much of
    # the posterior sits there says how much of the answer is semi-empirical.
    if ms_relation.relation == "bhac15":
        frac_ms_bhac = 1.0
    else:
        frac_ms_bhac = float(np.mean(m_ms[np.isfinite(m_ms)] < BROWN2022_M_MIN)) \
            if np.isfinite(m_ms).any() else float("nan")

    sum_co = _summary(m_co, WD_MASS_REL_SYST)
    sum_he = _summary(m_he, WD_MASS_REL_SYST)
    sum_ms = _summary(m_ms, MS_MASS_REL_SYST)

    # Two readings of the same posterior. The Brown rule gives the binary label
    # that goes in the table and can be set beside the published samples; the
    # probabilistic diagnosis says how firm that label is, and belongs in the
    # text. Reporting only the first would overstate the case.
    core_prob, reason_prob, core_fractions = diagnose_core(m_co, m_he)
    core, reason = classify_core_brown(m_co, m_he, he_grid, t_wd)

    # Composition-marginalised mass: each sample takes the mass of whichever
    # core can reproduce it, and draws between them when both can. That is a
    # flat prior on the composition, and it is the only defensible single
    # number for a target whose core is not settled.
    rng_pick = np.random.default_rng() if rng is None else rng
    co_ok = np.isfinite(m_co) & (m_co >= CO_CORE_MASS_MIN)
    he_ok = np.isfinite(m_he) & (m_he <= HE_CORE_MASS_MAX)
    take_he = (he_ok & ~co_ok) | (he_ok & co_ok & (rng_pick.random(m_co.size) < 0.5))
    m_marg = np.where(take_he, m_he, np.where(co_ok, m_co, np.nan))

    if wd_mass_track == "co":
        m_wd = m_co
        core_used = "CO"
        core_rule = "forced Bédard 2020 CO-core"
        take_co_age = np.ones(m_co.size, dtype=bool)
    else:
        m_wd = {"CO": m_co, "He": m_he}.get(core, m_marg)
        core_used = core if core in ("CO", "He") else "marginalised"
        core_rule = "Brown et al. (2022, 2023)"
        take_co_age = ~take_he

    out = {
        "n_samples":       int(arr.shape[0]),
        "core":            core_used if wd_mass_track == "co" else core,
        "core_reason":     (
            "forced: M_WD read off the Bédard et al. 2020 CO-core sequences "
            "only; the He-core grid is not used for the adopted mass"
            if wd_mass_track == "co" else reason
        ),
        "core_rule":       core_rule,
        "wd_mass_track":   wd_mass_track,
        "core_auto":       core,
        "core_auto_reason": reason,
        "core_he_ceiling_Msun": he_grid.mass_max,
        "core_probabilistic":        core_prob,
        "core_probabilistic_reason": reason_prob,
        "core_fractions":  core_fractions,
        "frac_in_domain":  {"CO": frac_co, "He": frac_he},
        "frac_MS_in_range": frac_ms,
        "frac_MS_on_bhac15_branch": frac_ms_bhac,
        "M_WD_CO": sum_co,
        "M_WD_He": sum_he,
        "M_WD_marginalised": _summary(m_marg, WD_MASS_REL_SYST),
        "M_WD":         _summary(m_wd, WD_MASS_REL_SYST),
        "M_WD_core_used": core_used,
        "M_MS":         sum_ms,
        "cooling_age_Gyr": _summary(np.where(
            take_co_age,
            np.asarray(co_grid.cooling_age(t_wd, r_wd), dtype=float),
            np.asarray(he_grid.cooling_age(t_wd, r_wd), dtype=float),
        )),
        "_samples": {"M_WD_CO": m_co, "M_WD_He": m_he, "M_MS": m_ms,
                     "M_WD": m_wd, "M_WD_marginalised": m_marg},
    }

    if period_d is not None and np.isfinite(period_d) and period_d > 0:
        orb = orbital_configuration(period_d, m_wd, m_ms, r_ms)
        out["period_d"] = float(period_d)
        out["orbital"]  = {k: _summary(v) for k, v in orb.items()}
        out["_samples"].update({
            "a_Rsun":     orb["a_Rsun"],
            "f_Roche_MS": orb["f_Roche_MS"],
        })
        f50 = out["orbital"]["f_Roche_MS"]
        if f50 is not None:
            out["roche_state"] = (
                "Roche-lobe filling"  if f50["p50"] >= 0.95 else
                "close to filling"    if f50["p50"] >= 0.80 else
                "detached"
            )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════════════════

def _plot_mass(result, samples, co_grid, he_grid, ms_relation, label, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.4))
    t_wd, r_wd, r_ms = samples[:, 0], samples[:, 1], samples[:, 3]

    # ── Core diagnostic: where the measurement sits among the model domains ──
    ax = axes[0, 0]
    for grid, colour in ((co_grid, "#1f77b4"), (he_grid, "#d62728")):
        for mass in np.unique(np.round(grid.mass_Msun, 3)):
            on = np.abs(grid.mass_Msun - mass) < 1e-6
            if on.sum() < 5:
                continue
            order = np.argsort(grid.teff_K[on])
            ax.plot(grid.teff_K[on][order] / 1e3, grid.radius_Rsun[on][order],
                    color=colour, lw=0.5, alpha=0.35, zorder=1)
        ax.plot([], [], color=colour, lw=1.4, label=grid.label)

    ax.scatter(t_wd / 1e3, r_wd, s=4, color="k", alpha=0.25, zorder=3,
               label="joint posterior")
    ax.set_xscale("log")
    ax.set_xlabel(r"$T_{\rm WD}$ [kK]")
    ax.set_ylabel(r"$R_{\rm WD}$ [$R_\odot$]")
    # Bound on the bulk of the posterior: a handful of far-tail samples would
    # otherwise stretch the axes and flatten the region of interest.
    t_lo, t_hi = np.percentile(t_wd, [0.5, 99.5]) / 1e3
    r_lo, r_hi = np.percentile(r_wd, [0.5, 99.5])
    ax.set_xlim(0.6 * t_lo, 1.7 * t_hi)
    ax.set_ylim(0.6 * r_lo, max(1.7 * r_hi, 0.035))
    ax.set_title(f"core: {result['core']}", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")

    # ── White dwarf mass under each core hypothesis ──────────────────────────
    ax = axes[0, 1]
    for key, colour, name in (("M_WD_CO", "#1f77b4", "CO core"),
                              ("M_WD_He", "#d62728", "He core")):
        values = result["_samples"][key]
        values = values[np.isfinite(values)]
        if values.size < 10:
            continue
        ax.hist(values, bins=50, histtype="step", color=colour, lw=1.4,
                density=True, label=f"{name} ({100*values.size/samples.shape[0]:.0f}%)")
        ax.axvline(np.median(values), color=colour, ls="--", lw=1.0)
    if result["M_WD_core_used"] == "marginalised":
        values = result["_samples"]["M_WD_marginalised"]
        values = values[np.isfinite(values)]
        if values.size >= 10:
            ax.hist(values, bins=50, histtype="stepfilled", color="k", alpha=0.18,
                    density=True, label="core marginalised", zorder=0)
    ax.axvline(HE_CORE_MASS_MAX, color="grey", ls=":", lw=1.0)
    ax.text(HE_CORE_MASS_MAX, ax.get_ylim()[1] * 0.95, " He-flash limit",
            fontsize=7, color="grey", va="top")
    ax.set_xlabel(r"$M_{\rm WD}$ [$M_\odot$]")
    ax.set_ylabel("posterior density")
    ax.legend(fontsize=8)

    # ── M dwarf: relation used and resulting mass ────────────────────────────
    ax = axes[1, 0]
    ax.plot(ms_relation.bhac_mass, ms_relation.bhac_radius, color="grey",
            ls="--", lw=1.2, label="BHAC15 1 Gyr (theoretical)")
    ax.plot(ms_relation.mass_grid, ms_relation.radius_grid, "k-", lw=1.4,
            label=("Brown 2022 (semi-empirical)"
                   if ms_relation.relation == "brown2022"
                   else "BHAC15 (theoretical)"))
    if np.isfinite(ms_relation.seam_mass):
        ax.axvline(ms_relation.seam_mass, color="grey", ls=":", lw=1.0)
    m_ms = result["_samples"]["M_MS"]
    ok = np.isfinite(m_ms)
    ax.scatter(m_ms[ok], r_ms[ok], s=4, color="#2ca02c", alpha=0.3,
               label="joint posterior")
    ax.set_xlabel(r"$M_{\rm MS}$ [$M_\odot$]")
    ax.set_ylabel(r"$R_{\rm MS}$ [$R_\odot$]")
    if ok.any():
        ax.set_xlim(max(0.0, 0.6 * np.nanmin(m_ms)), 1.4 * np.nanmax(m_ms))
        ax.set_ylim(0.6 * r_ms.min(), 1.4 * r_ms.max())
    ax.legend(fontsize=8)

    # ── Orbital configuration, when a period is available ────────────────────
    ax = axes[1, 1]
    if "f_Roche_MS" in result["_samples"]:
        values = result["_samples"]["f_Roche_MS"]
        values = values[np.isfinite(values)]
        ax.hist(values, bins=50, histtype="stepfilled", color="#9467bd",
                alpha=0.5, density=True)
        ax.axvline(1.0, color="k", ls="--", lw=1.2)
        ax.text(1.0, ax.get_ylim()[1] * 0.95, " Roche lobe", fontsize=7, va="top")
        ax.set_xlabel(r"$R_{\rm MS} / R_{\rm L,MS}$")
        ax.set_ylabel("posterior density")
        orb = result["orbital"]
        ax.set_title(
            f"P = {result['period_d']:.5f} d   "
            f"a = {orb['a_Rsun']['p50']:.3f} $R_\\odot$   "
            f"q = {orb['q_MS_over_WD']['p50']:.2f}",
            fontsize=9,
        )
    else:
        ax.text(0.5, 0.5, "no orbital period provided", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="grey")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(label, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_mass_analysis(
    gaia_id,
    tic_id,
    output_base,
    posterior_samples_joint_path,
    bedard_da_csv,
    laplata_dat,
    bhac15_csv,
    period_d=None,
    seed=42,
    max_samples=20000,
    co_grid=None,
    he_grid=None,
    ms_relation=None,
    wd_mass_track="auto",
):
    """Derive masses, core composition and orbital configuration for one target.

    The three model objects are reusable across targets; pass them in to avoid
    rebuilding the triangulations for every star.

    wd_mass_track is "auto" (Brown He/CO choice) or "co" (always Bédard 2020).
    """
    out_dir = Path(output_base) / f"TIC{tic_id}_Gaia{gaia_id}" / "mass"
    out_dir.mkdir(parents=True, exist_ok=True)
    label = f"Gaia {gaia_id} / TIC {tic_id}"

    if co_grid is None:
        co_grid = load_bedard_grid(bedard_da_csv)
    if he_grid is None:
        he_grid = load_laplata_grid(laplata_dat, core="He")
    if ms_relation is None:
        ms_relation = MDwarfRelation(bhac15_csv)

    samples = np.load(posterior_samples_joint_path)
    rng     = np.random.default_rng(seed)
    if max_samples and samples.shape[0] > max_samples:
        samples = samples[rng.choice(samples.shape[0], size=max_samples, replace=False)]

    result = derive_masses(samples, co_grid, he_grid, ms_relation,
                           period_d=period_d, rng=rng, max_samples=None,
                           wd_mass_track=wd_mass_track)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"[MASS] {label}")
    print(f"[MASS]   WD track: {result['wd_mass_track']}  "
          f"({result['core_rule']})")
    print(f"[MASS]   adopted core: {result['core']} — {result['core_reason']}")
    print(f"[MASS]   core (Brown rule): {result['core_auto']} — "
          f"{result['core_auto_reason']}")
    print(f"[MASS]   core (posterior):  {result['core_probabilistic']} — "
          f"{result['core_probabilistic_reason']}")
    print(f"[MASS]   inside model domain: CO {100*result['frac_in_domain']['CO']:.0f}%, "
          f"He {100*result['frac_in_domain']['He']:.0f}%")
    if result["core_he_ceiling_Msun"] < BROWN_CO_TRIAL_MASS - 1e-6:
        print(f"[MASS]   warning: the He grid stops at "
              f"{result['core_he_ceiling_Msun']:.3f} M_sun, below the "
              f"{BROWN_CO_TRIAL_MASS:.2f} M_sun ceiling of the Panei models Brown "
              f"uses, so this test is stricter than theirs and favours CO")
    for key, name in (("M_WD", f"M_WD [{result['M_WD_core_used']}]"), ("M_MS", "M_MS")):
        entry = result[key]
        if entry is None:
            print(f"[MASS]   {name}: undetermined")
            continue
        print(f"[MASS]   {name} = {entry['p50']:.3f} "
              f"-{entry['p50']-entry['p16']:.3f}/+{entry['p84']-entry['p50']:.3f} M_sun "
              f"(stat {entry['sigma_stat']:.3f}, relation {entry['sigma_syst']:.3f})")
    if "orbital" in result:
        orb = result["orbital"]
        print(f"[MASS]   a = {orb['a_Rsun']['p50']:.3f} R_sun   "
              f"q = {orb['q_MS_over_WD']['p50']:.3f}   "
              f"R_MS/R_L = {orb['f_Roche_MS']['p50']:.3f} ({result['roche_state']})")
        print(f"[MASS]   K_WD = {orb['K_WD_kms']['p50']:.1f} km/s   "
              f"K_MS = {orb['K_MS_kms']['p50']:.1f} km/s  (i = 90 deg)")

    # ── Posterior samples ─────────────────────────────────────────────────────
    columns = [result["_samples"]["M_WD_CO"], result["_samples"]["M_WD_He"],
               result["_samples"]["M_MS"]]
    columns.append(result["_samples"].get("a_Rsun", np.full(samples.shape[0], np.nan)))
    columns.append(result["_samples"].get("f_Roche_MS", np.full(samples.shape[0], np.nan)))
    path_samp = out_dir / "posterior_samples_mass.npy"
    np.save(path_samp, np.column_stack(columns).astype(np.float64))

    # ── JSON ──────────────────────────────────────────────────────────────────
    params = {k: v for k, v in result.items() if k != "_samples"}
    params.update({
        "gaia_id": gaia_id,
        "tic_id":  tic_id,
        "models_mass": {
            "CO": co_grid.label,
            "He": he_grid.label,
            "MS": ms_relation.label,
        },
        "relation_systematics_mass": {
            "M_WD_rel": WD_MASS_REL_SYST,
            "M_MS_rel": MS_MASS_REL_SYST,
        },
    })
    path_params = out_dir / "params_mass.json"
    with open(path_params, "w", encoding="utf-8") as fp:
        json.dump(params, fp, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────────────
    path_plot = out_dir / "plot_mass.pdf"
    _plot_mass(result, samples, co_grid, he_grid, ms_relation, label, path_plot)
    print(f"[MASS]   Saved → {out_dir}")

    out = {f"{k}_mass": v for k, v in result.items() if k != "_samples"}
    out.update({
        "gaia_id": gaia_id,
        "tic_id":  tic_id,
        "posterior_samples_mass_path": str(path_samp),
        "params_mass_path":            str(path_params),
        "plot_mass_path":              str(path_plot),
    })
    return out


def write_mass_summary(results, output_base):
    """One CSV row per target with the numbers meant for the paper tables."""
    import csv

    rows = []
    for res in results:
        if res is None:
            continue
        fractions = res.get("core_fractions_mass") or {}
        row = {
            "gaia_id": res.get("gaia_id"),
            "tic_id":  res.get("tic_id"),
            "run_mode":         res.get("run_mode"),
            "run_status":       res.get("run_status"),
            "run_finished_utc": res.get("run_finished_utc"),
            "core":    res.get("core_mass"),
            "M_WD_core_used":   res.get("M_WD_core_used_mass"),
            "wd_mass_track":    res.get("wd_mass_track_mass"),
            "core_rule":        res.get("core_rule_mass"),
            "core_auto":        res.get("core_auto_mass"),
            "core_reason":      res.get("core_reason_mass"),
            "core_he_ceiling_Msun":       res.get("core_he_ceiling_Msun_mass"),
            "core_probabilistic":         res.get("core_probabilistic_mass"),
            "core_probabilistic_reason":  res.get("core_probabilistic_reason_mass"),
            "frac_CO_only":     fractions.get("CO_only"),
            "frac_He_only":     fractions.get("He_only"),
            "frac_both":        fractions.get("both"),
            "frac_neither":     fractions.get("neither"),
        }
        for key, prefix in (("M_WD_mass", "M_WD"), ("M_WD_CO_mass", "M_WD_CO"),
                            ("M_WD_He_mass", "M_WD_He"),
                            ("M_WD_marginalised_mass", "M_WD_marg"),
                            ("M_MS_mass", "M_MS")):
            entry = res.get(key)
            if not entry:
                continue
            row[f"{prefix}_p50"]   = entry["p50"]
            row[f"{prefix}_err_lo"] = entry["p50"] - entry["p16"]
            row[f"{prefix}_err_hi"] = entry["p84"] - entry["p50"]
            if "sigma_total" in entry:
                row[f"{prefix}_sigma_stat"]  = entry["sigma_stat"]
                row[f"{prefix}_sigma_syst"]  = entry["sigma_syst"]
                row[f"{prefix}_sigma_total"] = entry["sigma_total"]

        age = res.get("cooling_age_Gyr_mass")
        if age:
            row["cooling_age_Gyr_p50"] = age["p50"]

        orb = res.get("orbital_mass")
        if orb:
            row["period_d"] = res.get("period_d_mass")
            row["roche_state"] = res.get("roche_state_mass")
            for key in ("a_Rsun", "q_MS_over_WD", "RL_MS_Rsun",
                        "f_Roche_MS", "K_WD_kms", "K_MS_kms"):
                entry = orb.get(key)
                if entry:
                    row[f"{key}_p50"]    = entry["p50"]
                    row[f"{key}_err_lo"] = entry["p50"] - entry["p16"]
                    row[f"{key}_err_hi"] = entry["p84"] - entry["p50"]
        rows.append(row)

    if not rows:
        print("[MASS] Nothing to summarise.")
        return None

    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    path = Path(output_base) / "mass_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[MASS] Summary → {path}")

    print(f"\n{'target':<16}{'core':<12}{'from':<14}{'M_WD':>20}{'M_MS':>20}")
    for row in rows:
        wd = (f"{row.get('M_WD_p50', float('nan')):.3f} "
              f"-{row.get('M_WD_err_lo', float('nan')):.3f}"
              f"/+{row.get('M_WD_err_hi', float('nan')):.3f}")
        ms = (f"{row.get('M_MS_p50', float('nan')):.3f} "
              f"-{row.get('M_MS_err_lo', float('nan')):.3f}"
              f"/+{row.get('M_MS_err_hi', float('nan')):.3f}")
        print(f"TIC {str(row['tic_id']):<12}{str(row['core']):<12}"
              f"{str(row['M_WD_core_used']):<14}{wd:>20}{ms:>20}")
    return path
