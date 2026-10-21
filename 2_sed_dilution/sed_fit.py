#!/usr/bin/env python3
"""sed_fit.py — 1-comp and 2-comp SED fitting for WD+MS eclipsing systems.

Step 1 of the pipeline. Downloads the Gaia DR3 astrometry and the VizieR SED
photometry of one target, then fits it twice with UltraNest:

    1-component : white dwarf only            [T_WD, R_WD, ϖ, A_V]
    2-component : white dwarf + M dwarf       [T_WD, R_WD, T_MS, R_MS, ϖ, A_V]

Units: temperatures in K, radii in R_sun, distances in pc, A_V in mag, fluxes
F_lambda in erg/s/cm²/Å, wavelengths in Å inside the model (μm for the
wavelength cuts given by the user).

Called by main_all.py:
    run_sed_fit(gaia_id, tic_id, output_base=<per-target analysis folder>, mode=MODE, ...)

Directory layout written by this module (main_all.py uses
<WORKSPACE>/TIC<id>/SED-dilution analysis/ as <output_base>):
    <output_base>/
    ├── downloads/
    │   └── {gaia_id}/
    │       └── {gaia_id}_vizier_sed_*.vot      ← VizieR cache (re-used on re-run)
    └── TIC{tic_id}_Gaia{gaia_id}/
        └── sed/
            ├── posterior_samples_1c_sed.npy    ← [N, 4]: T_WD, R_WD, d, AV
            ├── posterior_samples_sed.npy        ← [N, 6]: T_WD, R_WD, T_MS, R_MS, d, AV
            ├── params_sed.json                  ← p16 / p50 / p84 + Bayesian evidence
            ├── plot_sed.pdf                     ← SED fit figure
            ├── plot_sed_residuals.pdf           ← same figure + 2-comp residual panel
            ├── ultranest_1c/                    ← UltraNest run logs (1-comp)
            └── ultranest_2c/                    ← UltraNest run logs (2-comp)

Photometry delivered by VizieR without an uncertainty is not dropped straight
away: the relative uncertainty σ/F is looked up band by band in the catalogue
that published the measurement (see SIGMA_LOOKUP_CATALOGS) and applied to the
VizieR flux. Photometry whose σ is still unknown afterwards, and photometry
whose σ/F exceeds sed_max_rel_err, are excluded from the fit; the former is
not plotted either. Surviving points enter the likelihood with σ floored at
FRAC_FLOOR × F, since the published uncertainties are far smaller than the
actual catalogue-to-catalogue disagreement.

Variable naming: all exported keys carry the _sed (or _1c_sed) suffix to avoid
call collisions with dilution_fit.py and joint_fit.py.

Sampling is done on the Gaia parallax ϖ [mas] with a Gaussian prior; distances
are obtained as d = 1000/ϖ. Saved posteriors and returned values are in pc.

Extinction is constrained in two stages. The 1-component fit takes A_V flat on
[AV_MIN, AV_MAX]; the 2-component fit keeps that interval but weights it by a
Gaussian centred on the 1-component A_V posterior, with σ = half its 16–84
interval (floored at AV_2C_SIGMA_MIN). The white dwarf alone drives the
reddening — it dominates the blue end of the SED where the CCM89 law bites —
so a flat A_V in the 2-component fit lets extinction trade against companion
flux and settles on values that are far too low.

The companion's spectrum can be a blackbody or a synthetic atmosphere (PHOENIX
or BT-Settl); see SED_ATMO_VARIANTS and the ms_atmosphere / compare_atmospheres
arguments of run_sed_fit. Running several variants on the same photometry turns
the choice of companion model into a measurable systematic, reported next to
the nested-sampling (statistical) uncertainties in params_sed.json.

The white dwarf's spectrum is a blackbody or a Koester DA model atmosphere; see
WD_ATMO_VARIANTS and the wd_atmosphere argument. The same white-dwarf model is
used for every companion variant of a run.

Network and caches: the Gaia archive is queried on every call; the VizieR
VOTable is cached in <output_base>/downloads/<gaia_id>/; atmosphere grids and
SVO filter curves are cached by sed_atmosphere_models.py (DEFAULT_CACHE_DIR).
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus
from scipy.stats import gaussian_kde

import certifi
import astropy.units as u
from astropy.table import Table
from astropy.utils.data import download_file
from ultranest import ReactiveNestedSampler

os.environ["SSL_CERT_FILE"] = certifi.where()

# ── Caching astropy constants (avoid repeated unit overhead) ───────────────
_PC_TO_CM   = (1.0 * u.pc).to(u.cm).value          # 3.085677581e18 cm/pc
_RSUN_TO_CM = (1.0 * u.R_sun).to(u.cm).value        # 6.957e10 cm

# ── Physical constants (CGS) ───────────────────────────────────────────────
from astropy.constants import h as _h_cgs, c as _c_cgs, k_B as _kB_cgs
from wdms_common import M_SUN_G, T_MS_MIN_K, T_MS_MAX_K, derive_seed
_H   = _h_cgs.cgs.value    # erg·s
_C   = _c_cgs.cgs.value    # cm/s
_KB  = _kB_cgs.cgs.value   # erg/K

# ── Prior bounds ───────────────────────────────────────────────────────────
# Prior, filtering and sampling defaults when this module is used alone; main_all.py sets its own values
# (see its "Fit settings" section, function apply_fit_settings).
R_V_EXT          = 3.1
T_WD_MIN         = 5_000.0     # K
T_WD_MAX         = 100_000.0   # K
R_WD_MIN_RSUN    = 0.002
R_WD_MAX_RSUN    = 0.08
# Companion temperature interval, shared with the dilution and joint stages
# (wdms_common.T_MS_MIN_K / T_MS_MAX_K = 2300–4000 K). This is the fitting
# bound; a companion grid only narrows it when it does not cover it.
T_MS_MIN         = T_MS_MIN_K  # K
T_MS_MAX         = T_MS_MAX_K  # K
R_MS_MIN_RSUN    = 0.05
R_MS_MAX_RSUN    = 0.80
AV_MIN           = 0.0
AV_MAX           = 0.6
AV_2C_SIGMA_MIN  = 0.01        # floor on the width of the 2-comp A_V prior
N_SIGMA_PLX      = 5.0         # truncation on parallax Gaussian prior
MIN_PLX_SIGMA    = 1e-3        # mas, guard on tiny parallax uncertainty
PLX_MIN_MAS      = 0.05        # mas, hard floor (d ≤ 20 kpc) keeping 1000/ϖ finite

# ── SED filtering config ───────────────────────────────────────────────────
ABS_FLOOR           = 1e-20    # absolute guard, keeps σ strictly positive
FRAC_FLOOR          = 0.18    # σ floor as a fraction of the flux
# Reject photometry with σ/F above this value. Default of both run_sed_fit and
# joint_fit_ideal.run_joint_fit_ideal; main_all.py passes the same value (0.25)
# explicitly to both, so the two stages apply the same cut.
SED_MAX_REL_ERR_DEFAULT = 0.25
# Optional outlier clipping (off by default: main_all.SED_OUTLIER_CLIPPING = False).
# A point is a candidate when |residual - median| > SED_MAD_SIGMA × σ_robust,
# with σ_robust = 1.4826 × MAD, i.e. 3σ clipping in robust units.
SED_MAD_SIGMA       = 3.0
SED_MAX_REJECT_FRAC = 0.2     # at most this fraction of the targeted points rejected
SED_OUTLIER_MAX_ITERS = 1     # clipping passes; the continuum is rebuilt after each
SED_REJECT_BOTH     = True    # reject points above and below the continuum
OPTICAL_MAX_UM      = 1.1     # optical / infrared boundary for the sampling guard [μm]
MIN_POINTS_OPTICAL  = 3       # clipping never leaves fewer optical points than this
MIN_POINTS_IR       = 2       # ... nor fewer infrared points than this
VIZIER_RADIUS_ARCSEC_DEFAULT = 3   # default VizieR cone-search radius [arcsec]

# UltraNest settings of the 1- and 2-component SED runs, per sampling mode.
SED_SAMPLING = {
    "test":  dict(min_num_live_points=16, max_iters=20_000),
    "final": dict(min_num_live_points=500, dlogz=0.2),
}

# ── Gaia TAP endpoints (tried in order; 30 s timeout each) ─────────────────
# Put the preferred mirror first. ARI is often more responsive than ESA.
GAIA_TAP_ENDPOINTS = [
    #("ARI", "https://gaia.ari.uni-heidelberg.de/tap"),
    ("ESA", "https://gea.esac.esa.int/tap-server/tap"),
]
GAIA_TAP_TIMEOUT_S = 30.0

# ── Companion spectral model ───────────────────────────────────────────────
# The white-dwarf model (wd_atmosphere: blackbody or Koester DA, see
# WD_ATMO_VARIANTS below) is the same in every variant. An M dwarf's spectrum
# is carved by TiO, VO, H2O and H- opacity, so only the companion model changes
# from one variant to the next, and any difference in the recovered parameters
# is attributable to that single choice.
#
#   bb       reproduces the historical fit exactly: both components evaluated
#            monochromatically at the effective wavelength of each filter.
#   bb-int   same blackbodies, but integrated through the real filter
#            transmissions. Differencing bb-int against bb isolates the cost of
#            the monochromatic approximation, and differencing an atmosphere
#            variant against bb-int isolates the cost of the blackbody
#            companion. Without this intermediate step the two effects are
#            entangled.
#   phoenix  PHOENIX (Allard) grid distributed by STScI.
#   btsettl  BT-Settl CIFIST via the SVO theoretical spectra server. Best
#            reference below ~3000 K, but each (Teff, logg) node is a ~66 MB
#            download, so the grid is deliberately narrow (see
#            ATMO_GRID_RANGES) and the first run is slow.
SED_ATMO_VARIANTS = {
    "bb": {
        "label":           "Blackbody (monochromatic)",
        "ms_model":        "blackbody",
        "band_integrated": False,
    },
    "bb-int": {
        "label":           "Blackbody (band-integrated)",
        "ms_model":        "blackbody-exact",
        "band_integrated": True,
    },
    "phoenix": {
        "label":           "PHOENIX (Allard/STScI)",
        "ms_model":        "phoenix-stsci",
        "band_integrated": True,
    },
    "btsettl": {
        "label":           "BT-Settl CIFIST",
        "ms_model":        "bt-settl-cifist",
        "band_integrated": True,
    },
}

# Spellings accepted by ms_atmosphere / compare_atmospheres.
ATMO_ALIASES = {
    "bb": "bb", "blackbody": "bb", "planck": "bb",
    "bb-int": "bb-int", "bb_int": "bb-int", "blackbody-bandint": "bb-int",
    "blackbody-exact": "bb-int",
    "phoenix": "phoenix", "phoenix-stsci": "phoenix",
    "btsettl": "btsettl", "bt-settl": "btsettl", "bt-settl-cifist": "btsettl",
}

# Extent of the atmosphere grid loaded for the companion. Kept wider than the
# T_MS prior so that a run reuses the cached grid instead of rebuilding one;
# the prior is clipped to the grid afterwards.
ATMO_GRID_RANGES = {
    "phoenix-stsci":   {"teff": (2300.0, 5900.0), "logg": (4.0, 5.5)},
    "bt-settl-cifist": {"teff": (2300.0, 4600.0), "logg": (4.5, 5.5)},
}
ATMO_AV_NODES        = 9       # A_V nodes of the synthetic-photometry table
ATMO_MS_LOGG_MODE    = "ms_relation"   # "ms_relation" | "from_radius" | "fixed"
ATMO_MS_LOGG_FIXED   = 5.0
ATMO_CACHE_DIR_DEFAULT = None  # None → sed_atmosphere_models.DEFAULT_CACHE_DIR

# Spectral model of the white dwarf itself.
#
#   blackbody  historical behaviour, kept so that earlier runs remain
#              reproducible and so that the cost of the approximation can be
#              measured by differencing the two variants.
#   koester    Koester DA model atmospheres. This is the physically correct
#              choice: the optical continuum of a hot DA forms at depths where
#              the local temperature exceeds Teff, and a large part of the flux
#              escapes below the Lyman limit. Fitting a blackbody to optical
#              photometry therefore returns a Teff that is too high and a
#              radius that is too small, and both errors push the mass read off
#              a cooling track upward.
WD_ATMO_VARIANTS = {
    "blackbody": {"label": "Blackbody",  "wd_model": "blackbody-exact"},
    "koester":   {"label": "Koester DA", "wd_model": "koester-da"},
}
WD_ATMO_ALIASES = {
    "bb": "blackbody", "blackbody": "blackbody", "planck": "blackbody",
    "blackbody-exact": "blackbody", "none": "blackbody",
    "koester": "koester", "koester2": "koester", "koester-da": "koester",
    "da": "koester", "atmosphere": "koester",
}
WD_ATMO_GRID_RANGE = {"teff": (5_000.0, 80_000.0), "logg": (6.5, 9.5)}

# A DA atmosphere needs a gravity, unlike a blackbody. Between logg 7.0 and
# 8.5 the spectral shape moves by 0.4 per cent at 80 000 K but by 24 per cent
# at 10 000 K, so a single fixed value will not do across the sample.
# "cooling_track" reads the mass off a WD cooling track at the (Teff, R_WD)
# the sampler is proposing and forms logg = log(G M / R**2), which keeps the
# atmosphere consistent with the star being fitted.
ATMO_WD_LOGG_MODE  = "cooling_track"   # "cooling_track" | "fixed"
ATMO_WD_LOGG_FIXED = 7.75

# Parameters of the 2-component fit, in the order of the posterior columns.
PARAM_NAMES_2C = ["T_WD_sed", "R_WD_sed", "T_MS_sed", "R_MS_sed", "d_sed", "AV_sed"]

_G_CGS_LOCAL = 6.67430e-8
_MSUN_G      = M_SUN_G        # g, astropy.constants.M_sun (wdms_common)

# ── Catalogue lookup for photometry delivered without an uncertainty ───────
# The VizieR SED service returns several photometric points with no σ. Each
# entry below tells where to read the missing uncertainty, band by band, keyed
# by the sed_filter string of the SED table. Bands present in several
# catalogues are taken from the first entry listed. Two keys tune the lookup:
#   match    : "gaia_id"  → row selected on id_column = Gaia DR3 source_id
#              "position" → cone search, nearest source kept
#   quantity : "flux"      → the columns hold a flux and its error
#              "magnitude" → the columns hold a magnitude and its error, and
#                            σ_F/F = 0.4·ln(10)·σ_mag
SIGMA_LOOKUP_RADIUS_ARCSEC = 3.0    # cone search radius for "position" entries
_MAG_TO_REL_FLUX = 0.4 * np.log(10.0)

SIGMA_LOOKUP_CATALOGS = (
    {
        "catalog":   "I/360/syntphot",
        "match":     "gaia_id",
        "quantity":  "flux",
        "id_column": "Source",
        "bands": {
            "SDSS:u": ("Fu", "e_Fu"),
            "SDSS:g": ("Fg", "e_Fg"),
            "SDSS:r": ("Fr", "e_Fr"),
            "SDSS:i": ("Fi", "e_Fi"),
            "SDSS:z": ("Fz", "e_Fz"),
        },
    },
    {
        "catalog":   "J/A+A/674/A33/gspc-wd",
        "match":     "gaia_id",
        "quantity":  "flux",
        "id_column": "GaiaDR3",
        "bands": {
            "Johnson:U": ("FUJHN",  "e_FUJHN"),
            "Johnson:B": ("FBJHN",  "e_FBJHN"),
            "Johnson:V": ("FVJHN",  "e_FVJHN"),
            "Johnson:R": ("FRJHN",  "e_FRJHN"),
            "Johnson:I": ("FIJHN",  "e_FIJHN"),
            "SDSS:z":    ("FzSDSS", "e_FzSDSS"),
        },
    },
    # PPMXL and the 2MASS PSC carry no Gaia identifier: both are matched on
    # sky position. They republish the same 2MASS K photometry, so a target
    # whose Ks is flagged as an upper limit (e_Kmag empty) stays without σ.
    {
        "catalog":   "I/317/sample",
        "match":     "position",
        "quantity":  "magnitude",
        "bands": {
            "2MASS:Ks": ("Kmag", "e_Kmag"),
        },
    },
    {
        "catalog":   "II/246/out",
        "match":     "position",
        "quantity":  "magnitude",
        "bands": {
            "Johnson:K": ("Kmag", "e_Kmag"),
        },
    },
)
_SIGMA_LOOKUP_BANDS = {b for entry in SIGMA_LOOKUP_CATALOGS for b in entry["bands"]}


# ══════════════════════════════════════════════════════════════════════════════
# Data retrieval
# ══════════════════════════════════════════════════════════════════════════════

def fetch_gaia_astrometry(source_id, timeout_s=None):
    """Query Gaia DR3 (ra, dec, parallax, parallax_error).

    Endpoints are tried in the order of ``GAIA_TAP_ENDPOINTS`` (top of file).
    Each is aborted after ``timeout_s`` seconds (default ``GAIA_TAP_TIMEOUT_S``).
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
    from astroquery.utils.tap.core import TapPlus

    if timeout_s is None:
        timeout_s = float(GAIA_TAP_TIMEOUT_S)

    query = f"""
        SELECT ra, dec, parallax, parallax_error
        FROM gaiadr3.gaia_source
        WHERE source_id = {int(source_id)}
    """

    def _query(url):
        return TapPlus(url=url).launch_job(query).get_results()

    for label, url in GAIA_TAP_ENDPOINTS:
        try:
            print(f"[Gaia] Trying {label}…")
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_query, url)
                tbl = fut.result(timeout=float(timeout_s))
            if len(tbl) == 0:
                raise ValueError(f"Source {source_id} not found on {label}")
            row = tbl[0]
            print(f"[Gaia] OK ({label})")
            return (float(row["ra"]), float(row["dec"]),
                    float(row["parallax"]), float(row["parallax_error"]))
        except FuturesTimeoutError:
            print(f"[Gaia] {label} timed out after {timeout_s:.0f}s")
        except Exception as exc:
            print(f"[Gaia] {label} failed: {exc}")
    raise RuntimeError(f"All Gaia endpoints failed for source_id={source_id}")


def fetch_vizier_sed(ra_deg, dec_deg, gaia_id, download_root,
                     vizier_radius_arcsec=VIZIER_RADIUS_ARCSEC_DEFAULT,
                     use_local=True, force_download=False,
                     max_tries=3, retry_sleep_s=1.5):
    """Download or reuse cached VizieR SED VOTable.

    Returns (path, Table) on success, (None, None) on failure.
    """
    from astropy.io.votable import parse_single_table
    import time

    def _read(path):
        try:
            return Table.read(path, format="votable")
        except Exception:
            return parse_single_table(path).to_table()

    url = (
        "https://vizier.cds.unistra.fr/viz-bin/sed"
        f"?-c={quote_plus(f'{ra_deg} {dec_deg}')}"
        f"&-c.rs={vizier_radius_arcsec}&-out.max=999999&-mime=VOTable"
    )
    out_dir = Path(download_root) / str(gaia_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    if use_local and not force_download:
        for local_path in sorted(out_dir.glob(f"{gaia_id}_vizier_sed_*.vot")):
            try:
                tbl = _read(local_path)
                print(f"[VizieR] Using local: {local_path.name}")
                return local_path, tbl
            except Exception as exc:
                print(f"[VizieR] Local unreadable ({exc}), will re-download.")
                try:
                    local_path.rename(local_path.with_suffix(".vot.bad"))
                except Exception:
                    pass

    last_exc = None
    for attempt in range(max_tries):
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        raw = out_dir / f"{gaia_id}_vizier_sed_{stamp}.vot"
        try:
            Path(download_file(url, cache=False, allow_insecure=True)).replace(raw)
            tbl = _read(raw)
            return raw, tbl
        except Exception as exc:
            last_exc = exc
            print(f"[VizieR] Attempt {attempt + 1}/{max_tries} failed: {exc}")
            try:
                raw.rename(raw.with_suffix(".vot.bad"))
            except Exception:
                pass
            if attempt + 1 < max_tries:
                time.sleep(retry_sleep_s)

    print(f"[VizieR] SKIP {gaia_id}: all attempts failed. Last: {last_exc}")
    return None, None


def _table_value(row, column):
    """Table cell as float; NaN when the column is absent, masked or invalid."""
    try:
        value = row[column]
    except KeyError:
        return np.nan
    if np.ma.is_masked(value):
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def fetch_catalog_rel_errors(gaia_id, ra_deg=None, dec_deg=None,
                             wanted_bands=None,
                             radius_arcsec=SIGMA_LOOKUP_RADIUS_ARCSEC):
    """Read σ/F band by band from the catalogues of SIGMA_LOOKUP_CATALOGS.

    The *relative* uncertainty is returned rather than the published σ: the
    catalogues mix F_nu and F_lambda conventions (and J/A+A/674/A33 labels its
    SDSS z flux with the wrong unit), and their absolute flux scale differs by
    a few percent from the VizieR SED value. σ/F is immune to both problems,
    and it is also what a magnitude error natively measures.

    Parameters
    ----------
    gaia_id : int
        Gaia DR3 source_id, used by the "gaia_id" entries.
    ra_deg, dec_deg : float, optional
        ICRS position of the target. Required by the "position" entries, which
        are skipped when it is missing.
    wanted_bands : set of str, optional
        Restrict the query to these sed_filter names. Default: every band
        declared in SIGMA_LOOKUP_CATALOGS.
    radius_arcsec : float, optional
        Cone search radius of the "position" entries.

    Returns
    -------
    dict  — {band_name: sigma_over_flux}; empty when nothing could be read.
    """
    from astroquery.vizier import Vizier

    if wanted_bands is None:
        wanted_bands = _SIGMA_LOOKUP_BANDS

    rel_errors = {}
    for entry in SIGMA_LOOKUP_CATALOGS:
        bands = {b: cols for b, cols in entry["bands"].items()
                 if b in wanted_bands and b not in rel_errors}
        if not bands:
            continue

        match   = entry.get("match", "gaia_id")
        columns = [c for pair in bands.values() for c in pair]
        has_position = (ra_deg is not None and dec_deg is not None
                        and np.isfinite(ra_deg) and np.isfinite(dec_deg))
        if match == "position" and not has_position:
            print(f"[SED] σ lookup: no position for {entry['catalog']}")
            continue

        try:
            if match == "position":
                from astropy.coordinates import SkyCoord
                # "+_r" sorts by increasing separation: row_limit=1 keeps the
                # nearest source of the cone.
                vizier = Vizier(columns=columns + ["+_r"], row_limit=1)
                tables = vizier.query_region(
                    SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg,
                             frame="icrs"),
                    radius=radius_arcsec * u.arcsec,
                    catalog=entry["catalog"],
                )
            else:
                vizier = Vizier(columns=[entry["id_column"]] + columns,
                                row_limit=1)
                tables = vizier.query_constraints(
                    catalog=entry["catalog"],
                    **{entry["id_column"]: str(int(gaia_id))},
                )
            if len(tables) == 0 or len(tables[0]) == 0:
                print(f"[SED] σ lookup: no entry in {entry['catalog']}")
                continue
            row = tables[0][0]
        except Exception as exc:
            print(f"[SED] σ lookup failed on {entry['catalog']}: {exc}")
            continue

        is_mag = entry.get("quantity", "flux") == "magnitude"
        for band, (value_col, err_col) in bands.items():
            value = _table_value(row, value_col)
            err   = _table_value(row, err_col)
            if not err > 0:
                continue
            if is_mag and np.isfinite(value):
                rel_errors[band] = _MAG_TO_REL_FLUX * err
            elif not is_mag and value > 0:
                rel_errors[band] = err / value

    return rel_errors


# ══════════════════════════════════════════════════════════════════════════════
# SED table parsing helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_wave(tbl):
    if "sed_wave" in tbl.colnames:
        return u.Quantity(tbl["sed_wave"], tbl["sed_wave"].unit, copy=False)
    nu = u.Quantity(tbl["sed_freq"], tbl["sed_freq"].unit, copy=False)
    return (_c_cgs / nu).to(u.m)


def _get_flux_err(tbl, wave):
    f = u.Quantity(tbl["sed_flux"], tbl["sed_flux"].unit, copy=False)
    e = (u.Quantity(tbl["sed_eflux"], tbl["sed_eflux"].unit, copy=False)
         if "sed_eflux" in tbl.colnames else None)
    equiv = u.spectral_density(wave)
    cgs   = u.erg / u.s / u.cm**2 / u.AA
    f_cgs = f.to(cgs, equivalencies=equiv)
    e_cgs = None if e is None else e.to(cgs, equivalencies=equiv)
    return f_cgs, e_cgs


def _get_bandpass(tbl):
    for col in ("sed_filter", "sed_band", "filter"):
        if col in tbl.colnames:
            return np.array(tbl[col].astype(str))
    wl_um = _get_wave(tbl).to(u.um).value
    return np.array([f"{wl:.4f} um" for wl in wl_um])


def _get_catalog(tbl):
    if "sed_cat" in tbl.colnames:
        cat = np.array(tbl["sed_cat"].astype(str))
        out = []
        for x in cat:
            if x.startswith(("II/", "I/", "V/", "J/")):
                out.append("/".join(x.split("/")[:2]))
            else:
                out.append(x.split("/")[-1] or x)
        return np.array(out)
    bp = _get_bandpass(tbl)
    return np.array([b.split(".")[0] for b in bp])


def _deduplicate(filt, cat, err):
    best = {}
    for idx, (f, c, e) in enumerate(zip(filt, cat, err)):
        key = (c, f)
        if key not in best:
            best[key] = idx
        else:
            prev_e = err[best[key]]
            if np.isfinite(e) and (not np.isfinite(prev_e) or e < prev_e):
                best[key] = idx
    return np.array(sorted(best.values()), dtype=int)


def _is_gaia(cat_name, band_name):
    return "GAIA" in f"{cat_name} {band_name}".upper()


def _append_reason(reasons, mask, text):
    for i in np.where(mask)[0]:
        reasons[i] = f"{reasons[i]};{text}" if reasons[i] else text


def _keep_min_sampling(mask_used, wl_um):
    n_opt = int(np.sum(mask_used & (wl_um <= OPTICAL_MAX_UM)))
    n_ir  = int(np.sum(mask_used & (wl_um >  OPTICAL_MAX_UM)))
    return n_opt >= MIN_POINTS_OPTICAL and n_ir >= MIN_POINTS_IR


def _robust_outlier_mask(wl_um, flux_cgs, target_mask, active_mask,
                         max_reject=None):
    """
    One pass of continuum-relative 3σ (MAD) clipping on normalized residuals.

    The continuum is a leave-one-out log-log interpolation through the currently
    active photometry. Residuals are therefore measured relative to that
    continuum model, which also flags points dominated by strong absorption
    features that a smooth continuum (and later a blackbody) cannot reproduce.
    """
    idx_active = np.where(active_mask & target_mask)[0]
    if idx_active.size < 3:
        return np.zeros_like(active_mask, dtype=bool), np.full(active_mask.size, np.nan)

    logw = np.log10(wl_um)
    logf = np.log10(flux_cgs)
    residuals = np.full(active_mask.size, np.nan)

    for i in idx_active:
        train = active_mask.copy(); train[i] = False
        tidx  = np.where(train)[0]
        if tidx.size < 2:
            continue
        x = logw[tidx]; y = logf[tidx]
        order = np.argsort(x); x, y = x[order], y[order]
        if logw[i] < x[0] or logw[i] > x[-1]:
            continue
        residuals[i] = logf[i] - np.interp(logw[i], x, y)

    valid = np.isfinite(residuals[idx_active])
    if not np.any(valid):
        return np.zeros_like(active_mask, dtype=bool), residuals

    rvals     = residuals[idx_active][valid]
    med       = np.median(rvals)
    sigma_rob = 1.4826 * np.median(np.abs(rvals - med))
    if not np.isfinite(sigma_rob) or sigma_rob <= 0:
        return np.zeros_like(active_mask, dtype=bool), residuals

    threshold = SED_MAD_SIGMA * sigma_rob
    cand = np.zeros_like(active_mask, dtype=bool)
    if SED_REJECT_BOTH:
        cand[idx_active] = np.abs(residuals[idx_active] - med) > threshold
    else:
        cand[idx_active] = (residuals[idx_active] - med) > threshold

    cand_idx = np.where(cand)[0]
    if cand_idx.size == 0:
        return np.zeros_like(active_mask, dtype=bool), residuals

    n_target = int(np.sum(active_mask & target_mask))
    if max_reject is None:
        max_reject = max(1, int(np.floor(SED_MAX_REJECT_FRAC * n_target)))
    max_reject = int(max(0, max_reject))
    if max_reject == 0:
        return np.zeros_like(active_mask, dtype=bool), residuals

    score_order = np.argsort(np.abs(residuals[cand_idx] - med))[::-1]

    keep_reject = np.zeros_like(active_mask, dtype=bool)
    for j in cand_idx[score_order]:
        if int(np.sum(keep_reject)) >= max_reject:
            break
        trial = active_mask.copy(); trial[j] = False
        if _keep_min_sampling(trial, wl_um):
            keep_reject[j] = True

    return keep_reject, residuals


def _iterative_robust_outlier_mask(
    wl_um,
    flux_cgs,
    target_mask,
    active_mask,
    max_iters=None,
):
    """
    Iterative continuum-relative 3σ clipping.

    After each rejection pass the continuum is rebuilt on the surviving points
    and the clipping is repeated, until no additional points are rejected or
    ``max_iters`` (default SED_OUTLIER_MAX_ITERS = 1) is reached. The threshold
    is SED_MAD_SIGMA robust sigmas (see the constant). A global cap
    ``SED_MAX_REJECT_FRAC`` still limits the total fraction rejected relative
    to the initial active+target set.
    """
    if max_iters is None:
        max_iters = SED_OUTLIER_MAX_ITERS
    max_iters = int(max(1, max_iters))
    mask_work = np.asarray(active_mask, dtype=bool).copy()
    rejected_total = np.zeros_like(mask_work, dtype=bool)
    residuals_last = np.full(mask_work.size, np.nan)

    n0_target = int(np.sum(active_mask & target_mask))
    global_max_reject = max(1, int(np.floor(SED_MAX_REJECT_FRAC * n0_target))) if n0_target > 0 else 0

    print(
        f"[SED] Iterative {SED_MAD_SIGMA}σ outlier clipping "
        f"(max {max_iters} iterations, max reject frac={SED_MAX_REJECT_FRAC:.0%})"
    )

    for it in range(1, max_iters + 1):
        n_already = int(np.sum(rejected_total))
        remaining_budget = global_max_reject - n_already
        if remaining_budget <= 0:
            print(f"[SED]   iter {it}: global reject budget exhausted "
                  f"({n_already}/{global_max_reject})")
            break

        out_mask, residuals_last = _robust_outlier_mask(
            wl_um, flux_cgs, target_mask, mask_work, max_reject=remaining_budget
        )
        n_new = int(np.sum(out_mask))
        if n_new == 0:
            print(f"[SED]   iter {it}: no additional outliers — converged")
            break

        rejected_total |= out_mask
        mask_work &= ~out_mask
        print(
            f"[SED]   iter {it}: rejected {n_new} point(s) "
            f"(cumulative {int(np.sum(rejected_total))})"
        )

        if not _keep_min_sampling(mask_work, wl_um):
            # Roll back this iteration's rejections and stop.
            rejected_total &= ~out_mask
            mask_work = np.asarray(active_mask, dtype=bool) & ~rejected_total
            print(
                f"[SED]   iter {it}: sampling guard tripped — "
                f"reverted this iteration and stopped"
            )
            break
    else:
        print(f"[SED]   reached maximum of {max_iters} clipping iterations")

    return rejected_total, residuals_last


# ══════════════════════════════════════════════════════════════════════════════
# Physics
# ══════════════════════════════════════════════════════════════════════════════

def _planck_F_lambda(w_AA, T):
    """Planck spectral flux [erg/s/cm²/Å] at wavelength w_AA [Å], temperature T [K]."""
    wl_cm = np.asarray(w_AA, dtype=float) * 1e-8
    x = np.clip((_H * _C) / (wl_cm * _KB * T), 1e-6, 700.0)
    B = 2.0 * _H * _C**2 / wl_cm**5 / np.expm1(x)
    return np.pi * B * 1e-8   # convert per cm to per Å


def _ccm89_axav(wave_A, Rv=R_V_EXT):
    """CCM89 + O'Donnell 1994: A(λ)/A(V) as a function of wavelength [Å]."""
    x = 1.0 / (np.asarray(wave_A, float) * 1e-4)   # μm⁻¹
    a = np.zeros_like(x); b = np.zeros_like(x)

    ir  = x < 1.1
    a[ir] =  0.574 * x[ir]**1.61
    b[ir] = -0.527 * x[ir]**1.61

    opt = (x >= 1.1) & (x <= 3.3)
    y   = x[opt] - 1.82
    a[opt] = (1
              + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4
              + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7)
    b[opt] = (  1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4
              - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7)

    return a + b / Rv


def _redden(w_AA, A_V, Rv=R_V_EXT):
    w = np.asarray(w_AA, dtype=float)
    if A_V <= 0:
        return np.ones_like(w)
    return 10.0**(-0.4 * A_V * _ccm89_axav(w, Rv))


def _model_1comp(params, w_AA):
    T_WD, R_WD_Rsun, d_pc, A_V = params
    d_cm  = d_pc  * _PC_TO_CM
    R_cm  = R_WD_Rsun * _RSUN_TO_CM
    s     = (R_cm / d_cm)**2
    return s * _planck_F_lambda(w_AA, T_WD) * _redden(w_AA, A_V)


def _model_2comp(params, w_AA):
    T_WD, R_WD_Rsun, T_MS, R_MS_Rsun, d_pc, A_V = params
    d_cm     = d_pc      * _PC_TO_CM
    R_WD_cm  = R_WD_Rsun * _RSUN_TO_CM
    R_MS_cm  = R_MS_Rsun * _RSUN_TO_CM
    s_WD = (R_WD_cm / d_cm)**2
    s_MS = (R_MS_cm / d_cm)**2
    f = (s_WD * _planck_F_lambda(w_AA, T_WD)
       + s_MS * _planck_F_lambda(w_AA, T_MS))
    return f * _redden(w_AA, A_V)


def _lnlike_gauss(model, data, sigma):
    r = (data - model) / sigma
    return -0.5 * float(np.sum(r * r + np.log(2.0 * np.pi * sigma * sigma)))


# ══════════════════════════════════════════════════════════════════════════════
# Companion spectral model variants
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_atmo_key(name):
    key = ATMO_ALIASES.get(str(name).strip().lower())
    if key is None:
        raise KeyError(
            f"Unknown companion model {name!r}; expected one of "
            f"{sorted(set(ATMO_ALIASES))}"
        )
    return key


def _resolve_wd_atmo_key(name):
    key = WD_ATMO_ALIASES.get(str(name).strip().lower())
    if key is None:
        raise KeyError(
            f"Unknown white-dwarf model {name!r}; expected one of "
            f"{sorted(set(WD_ATMO_ALIASES))}"
        )
    return key


class _WDCoolingLogg:
    """logg = log(G M / R**2) with M read off a WD cooling track at (Teff, R).

    Falls back to a fixed gravity outside the track, and memoises on a coarse
    (Teff, R) lattice because the caller hits it once per likelihood
    evaluation while the gravity itself barely moves between neighbours.
    """

    def __init__(self, track_path, fallback=ATMO_WD_LOGG_FIXED, verbose=True):
        from mass import load_bedard_grid

        self._grid = load_bedard_grid(track_path)
        self._fallback = float(fallback)
        self._cache = {}
        if verbose:
            print(f"[SED] WD gravity from the cooling track: {Path(track_path).name}")

    def __call__(self, t_wd, r_wd):
        key = (round(float(t_wd) / 100.0), round(float(r_wd), 5))
        value = self._cache.get(key)
        if value is None:
            mass = np.atleast_1d(self._grid.mass(float(t_wd), float(r_wd)))[0]
            if np.isfinite(mass):
                value = float(np.log10(
                    _G_CGS_LOCAL * float(mass) * _MSUN_G / (float(r_wd) * _RSUN_TO_CM) ** 2))
            else:
                value = self._fallback
            self._cache[key] = value
        return value


class _SedModel:
    """Predicted band fluxes for one choice of companion spectral model.

    Exposes the same two entry points whatever the variant — ``bands_1comp``
    and ``bands_2comp`` return the model flux at each fitted photometric point,
    in erg/s/cm²/Å — so the samplers below are written once. ``spectrum_*``
    returns the underlying high-resolution spectrum and is used only for
    plotting.
    """

    def __init__(self, key, w_fit, band_fit, av_nodes=None, cache_dir=None,
                 grid_ranges=None, ms_logg_mode=ATMO_MS_LOGG_MODE,
                 ms_logg_fixed=ATMO_MS_LOGG_FIXED, wd_atmosphere="blackbody",
                 wd_logg_mode=ATMO_WD_LOGG_MODE, wd_logg_fixed=ATMO_WD_LOGG_FIXED,
                 wd_cooling_track=None, t_ms_min=T_MS_MIN, t_ms_max=T_MS_MAX,
                 verbose=True):
        spec = SED_ATMO_VARIANTS[key]
        self.wd_key          = _resolve_wd_atmo_key(wd_atmosphere)
        self.wd_model        = WD_ATMO_VARIANTS[self.wd_key]["wd_model"]
        self.wd_label        = WD_ATMO_VARIANTS[self.wd_key]["label"]
        self.key             = key
        self.label           = spec["label"]
        self.ms_model        = spec["ms_model"]
        # A model atmosphere only exists as a band-integrated component, so
        # asking for a Koester white dwarf promotes the monochromatic variant.
        self.band_integrated = bool(spec["band_integrated"]) or self.wd_key != "blackbody"
        self.w_fit           = np.asarray(w_fit, dtype=float)
        # Requested companion fitting bound [K]; narrowed below only when the
        # companion grid does not cover it (never widened).
        self.t_ms_requested  = (float(t_ms_min), float(t_ms_max))
        self.t_ms_bounds     = self.t_ms_requested
        self.t_wd_bounds     = (T_WD_MIN, T_WD_MAX)
        self._ms_logg_mode   = str(ms_logg_mode)
        self._ms_logg_fixed  = float(ms_logg_fixed)
        self._wd_logg_fixed  = float(wd_logg_fixed)
        self._wd_logg_fn     = None
        self._sam = None
        self._wd = None
        self._ms = None

        if not self.band_integrated:
            return

        if self.band_integrated and not spec["band_integrated"]:
            print(f"[SED] Variant '{key}' promoted to band integration: a "
                  f"{self.wd_label} white dwarf has no monochromatic form")

        import sed_atmosphere_models as sam
        self._sam = sam

        kw = {} if cache_dir is None else {"cache_dir": Path(cache_dir)}
        bandpasses = sam.resolve_bandpasses(band_fit, self.w_fit, verbose=verbose, **kw)

        if self.wd_key == "blackbody":
            self._wd = sam.make_component(
                "blackbody-exact", bandpasses, R_V=R_V_EXT,
                blackbody_teff_range=(T_WD_MIN, T_WD_MAX), verbose=verbose, **kw)
        else:
            ranges = dict(WD_ATMO_GRID_RANGE)
            ranges.update((grid_ranges or {}).get(self.wd_model, {}))
            if av_nodes is None:
                av_nodes = np.linspace(AV_MIN, AV_MAX, ATMO_AV_NODES)
            self._wd = sam.make_component(
                self.wd_model, bandpasses, av_nodes=np.asarray(av_nodes, dtype=float),
                R_V=R_V_EXT,
                teff_range=ranges["teff"], logg_range=ranges["logg"],
                verbose=verbose, **kw)
            lo, hi = self._wd.teff_bounds
            self.t_wd_bounds = (max(T_WD_MIN, lo), min(T_WD_MAX, hi))
            if self.t_wd_bounds[0] >= self.t_wd_bounds[1]:
                raise ValueError(
                    f"[SED] Empty T_WD interval for grid {self.wd_model}: "
                    f"prior [{T_WD_MIN:.0f}, {T_WD_MAX:.0f}] K vs grid "
                    f"[{lo:.0f}, {hi:.0f}] K")
            if str(wd_logg_mode) == "cooling_track" and wd_cooling_track is not None:
                self._wd_logg_fn = _WDCoolingLogg(
                    wd_cooling_track, fallback=self._wd_logg_fixed, verbose=verbose)
            elif verbose:
                print(f"[SED] WD gravity held at logg = {self._wd_logg_fixed:.2f}")

        if self.ms_model == "blackbody-exact":
            self._ms = sam.make_component(
                "blackbody-exact", bandpasses, R_V=R_V_EXT,
                blackbody_teff_range=self.t_ms_requested, verbose=verbose, **kw)
        else:
            ranges = dict(ATMO_GRID_RANGES.get(self.ms_model, {}))
            ranges.update((grid_ranges or {}).get(self.ms_model, {}))
            if av_nodes is None:
                av_nodes = np.linspace(AV_MIN, AV_MAX, ATMO_AV_NODES)
            if self.ms_model == "bt-settl-cifist":
                print("[SED] BT-Settl: the SVO server sends ~66 MB per (Teff, logg) "
                      f"node, so building {ranges.get('teff')} K x "
                      f"{ranges.get('logg')} dex costs a few GB the first time. "
                      "Subsequent runs read the cached grid.")
            self._ms = sam.make_component(
                self.ms_model, bandpasses, av_nodes=np.asarray(av_nodes, dtype=float),
                R_V=R_V_EXT,
                teff_range=ranges.get("teff", (2300.0, 5900.0)),
                logg_range=ranges.get("logg", (4.0, 5.5)),
                verbose=verbose, **kw)
            lo, hi = self._ms.teff_bounds
            req_lo, req_hi = self.t_ms_requested
            self.t_ms_bounds = (max(req_lo, lo), min(req_hi, hi))
            if self.t_ms_bounds[0] >= self.t_ms_bounds[1]:
                raise ValueError(
                    f"[SED] Empty T_MS interval for grid {self.ms_model}: "
                    f"prior [{req_lo:.0f}, {req_hi:.0f}] K vs grid "
                    f"[{lo:.0f}, {hi:.0f}] K")
            if self.t_ms_bounds != self.t_ms_requested:
                print(f"[SED] WARNING: the {self.ms_model} grid covers "
                      f"[{lo:.0f}, {hi:.0f}] K only; the requested T_MS bound "
                      f"[{req_lo:.0f}, {req_hi:.0f}] K is narrowed to "
                      f"[{self.t_ms_bounds[0]:.0f}, {self.t_ms_bounds[1]:.0f}] K "
                      f"(no extrapolation)")

    def _ms_logg(self, t_ms, r_ms):
        # Shared prescription (sed_atmosphere_models.companion_logg), also used
        # by the eclipse-depth term (depth_model), then clipped to this grid.
        if self._ms is None or not self._ms.uses_logg:
            return 0.0
        value = float(self._sam.companion_logg(
            t_ms, r_ms, mode=self._ms_logg_mode, fixed=self._ms_logg_fixed))
        lo, hi = self._ms.logg_bounds
        return float(np.clip(value, lo, hi))

    def _wd_logg(self, t_wd, r_wd):
        if self._wd is None or not getattr(self._wd, "uses_logg", False):
            return 0.0
        value = (self._wd_logg_fn(t_wd, r_wd) if self._wd_logg_fn is not None
                 else self._wd_logg_fixed)
        lo, hi = self._wd.logg_bounds
        return float(np.clip(value, lo, hi))

    def bands_1comp(self, params):
        """Model flux at each fitted point for [T_WD, R_WD, d_pc, A_V]."""
        if not self.band_integrated:
            return _model_1comp(params, self.w_fit)
        T_WD, R_WD, d_pc, A_V = params
        scale = (R_WD * _RSUN_TO_CM / (d_pc * _PC_TO_CM)) ** 2
        return scale * self._wd.band_flux(T_WD, logg=self._wd_logg(T_WD, R_WD), a_v=A_V)

    def bands_2comp(self, params, wd_logg=None, ms_logg=None):
        """Model flux for [T_WD, R_WD, T_MS, R_MS, d_pc, A_V].

        wd_logg / ms_logg: optional gravities already computed by the caller
        (the joint fit computes them once per sample and shares them with the
        eclipse-depth term). They are clipped to this model's grid range, the
        same policy as the gravities computed here when they are omitted.
        Ignored for a component that has no gravity (blackbody).
        """
        if not self.band_integrated:
            return _model_2comp(params, self.w_fit)
        T_WD, R_WD, T_MS, R_MS, d_pc, A_V = params
        d_cm = d_pc * _PC_TO_CM
        s_wd = (R_WD * _RSUN_TO_CM / d_cm) ** 2
        s_ms = (R_MS * _RSUN_TO_CM / d_cm) ** 2
        g_wd = self._wd_logg(T_WD, R_WD) if wd_logg is None else self._clip_logg(self._wd, wd_logg)
        g_ms = self._ms_logg(T_MS, R_MS) if ms_logg is None else self._clip_logg(self._ms, ms_logg)
        return (s_wd * self._wd.band_flux(T_WD, logg=g_wd, a_v=A_V)
                + s_ms * self._ms.band_flux(T_MS, logg=g_ms, a_v=A_V))

    @staticmethod
    def _clip_logg(component, value):
        if component is None or not getattr(component, "uses_logg", False):
            return 0.0
        lo, hi = component.logg_bounds
        return float(np.clip(float(value), lo, hi))

    def spectrum_1comp(self, params, wave_AA):
        if not self.band_integrated:
            return _model_1comp(params, wave_AA)
        T_WD, R_WD, d_pc, A_V = params
        scale = (R_WD * _RSUN_TO_CM / (d_pc * _PC_TO_CM)) ** 2
        _, flux = self._wd.spectrum(
            T_WD, logg=self._wd_logg(T_WD, R_WD), a_v=A_V, wave_out=wave_AA)
        return scale * flux

    def spectrum_components(self, params, wave_AA):
        """(WD, companion) spectra at the observer, for the 2-component model."""
        T_WD, R_WD, T_MS, R_MS, d_pc, A_V = params
        if not self.band_integrated:
            wd = _model_1comp([T_WD, R_WD, d_pc, A_V], wave_AA)
            return wd, _model_2comp(params, wave_AA) - wd
        d_cm = d_pc * _PC_TO_CM
        s_wd = (R_WD * _RSUN_TO_CM / d_cm) ** 2
        s_ms = (R_MS * _RSUN_TO_CM / d_cm) ** 2
        _, f_wd = self._wd.spectrum(
            T_WD, logg=self._wd_logg(T_WD, R_WD), a_v=A_V, wave_out=wave_AA)
        _, f_ms = self._ms.spectrum(T_MS, logg=self._ms_logg(T_MS, R_MS),
                                    a_v=A_V, wave_out=wave_AA)
        return s_wd * f_wd, s_ms * f_ms

    def spectrum_2comp(self, params, wave_AA):
        wd, ms = self.spectrum_components(params, wave_AA)
        return wd + ms

    def __repr__(self):
        return f"<_SedModel {self.key}: {self.label}>"


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_sed_fit(
    gaia_id,
    tic_id,
    output_base,
    mode="final",
    sed_exclude_windows_um=None,
    sed_exclude_below_um=0.35,
    sed_exclude_above_um=10.0,
    enable_outlier_filter=False,
    filter_all_sed_points=False,
    outlier_max_iters=None,
    show_excluded_sed_points=False,
    download_root=None,
    vizier_radius_arcsec=VIZIER_RADIUS_ARCSEC_DEFAULT,
    sed_max_rel_err=SED_MAX_REL_ERR_DEFAULT,
    recover_sigma_from_catalogs=True,
    ms_atmosphere="blackbody",
    compare_atmospheres=None,
    atmo_cache_dir=ATMO_CACHE_DIR_DEFAULT,
    atmo_grid_ranges=None,
    ms_logg_mode=ATMO_MS_LOGG_MODE,
    ms_logg_fixed=ATMO_MS_LOGG_FIXED,
    wd_atmosphere="blackbody",
    wd_logg_mode=ATMO_WD_LOGG_MODE,
    wd_logg_fixed=ATMO_WD_LOGG_FIXED,
    wd_cooling_track=None,
    t_ms_min=T_MS_MIN,
    t_ms_max=T_MS_MAX,
    seed=42,
):
    """Run 1-comp and 2-comp SED fits for a WD+MS eclipsing system.

    Parameters
    ----------
    gaia_id : int
        Gaia DR3 source_id.
    tic_id : int
        TESS Input Catalog ID.
    output_base : str or Path
        Root of the results directory tree.
    mode : {"test", "final"}
        "test"  → min_num_live_points=16, max_iters=20 000 (fast, for debugging)
        "final" → min_num_live_points=500, dlogz=0.2
    sed_exclude_windows_um : list of (lo, hi) tuples in micron, optional
        Wavelength windows to exclude from the SED fit.
    sed_exclude_below_um : float
        Exclude all SED points below this wavelength [μm].
    sed_exclude_above_um : float
        Exclude all SED points above this wavelength [μm].
    enable_outlier_filter : bool
        Apply iterative continuum-relative 3σ (MAD) clipping on normalized
        residuals. The continuum is rebuilt after each rejection pass until no
        further points are rejected, up to ``outlier_max_iters``.
    filter_all_sed_points : bool
        If True, the outlier filter targets all photometry; if False, Gaia only.
    outlier_max_iters : int or None
        Maximum number of sigma-clipping iterations (None:
        ``SED_OUTLIER_MAX_ITERS``, 1 by default).
    show_excluded_sed_points : bool
        If True, plot excluded SED points (crimson crosses) and show them in
        the legend. If False, omit them from both the figure and the legend.
    download_root : str or Path, optional
        Where to cache VizieR downloads. Default: <output_base>/downloads.
    sed_max_rel_err : float
        Reject any photometric point with σ/F above this value. Points without
        a usable uncertainty are always rejected, whatever this setting.
    recover_sigma_from_catalogs : bool
        Look up σ/F in the source catalogue (SIGMA_LOOKUP_CATALOGS) for the
        photometry VizieR returns without an uncertainty, so those points can
        be fitted with their own σ instead of being discarded.
    ms_atmosphere : str
        Companion spectral model of the headline fit, whose posterior feeds the
        dilution and joint stages: "blackbody" (default, historical behaviour),
        "bb-int", "phoenix" or "btsettl". See SED_ATMO_VARIANTS.
    compare_atmospheres : sequence of str, "all", or None
        Extra companion models refitted on the same photometry, with the same
        priors, to measure how much the choice of model moves the parameters.
        The reference is added automatically if missing. None disables the
        comparison and leaves the runtime unchanged.
    atmo_cache_dir : str or Path, optional
        Where atmosphere grids and filter transmissions are cached. Default:
        sed_atmosphere_models.DEFAULT_CACHE_DIR.
    atmo_grid_ranges : dict, optional
        Per-model override of ATMO_GRID_RANGES, e.g.
        {"bt-settl-cifist": {"teff": (2600, 4000), "logg": (5.0, 5.5)}}.
    ms_logg_mode : {"ms_relation", "from_radius", "fixed"}
        Gravity used when querying the companion's atmosphere grid.
    ms_logg_fixed : float
        Gravity used when ms_logg_mode="fixed".
    wd_atmosphere : {"blackbody", "koester"}
        White-dwarf spectral model, shared by every companion variant. See
        WD_ATMO_VARIANTS. "koester" reads the local Koester DA grid and forces
        band integration.
    wd_logg_mode : {"cooling_track", "fixed"}
        Gravity used when reading the Koester grid. "cooling_track" needs
        wd_cooling_track; otherwise wd_logg_fixed is used.
    wd_logg_fixed : float
        White-dwarf gravity [log10 cgs] for "fixed", and the fallback outside
        the cooling track.
    wd_cooling_track : str or Path, optional
        Bédard 2020 CSV used by wd_logg_mode="cooling_track".
    t_ms_min, t_ms_max : float
        Companion temperature fitting bound [K], the same for every companion
        variant (default 2300–4000 K, wdms_common). A companion grid that does
        not cover it narrows it, with a warning; it is never widened.
    seed : int
        Master seed. NumPy's global random state is seeded before each UltraNest
        run with derive_seed(seed, "sed:1comp:<ref>") or
        derive_seed(seed, "sed:2comp:<variant>"); the seeds are stored in
        params_sed.json under "seeds_sed".

    Returns
    -------
    dict with keys (all _sed-suffixed):
        T_WD_sed, R_WD_sed, T_MS_sed, R_MS_sed, d_sed, AV_sed  — 2-comp p50
        T_WD_1c_sed, R_WD_1c_sed, d_1c_sed, AV_1c_sed           — 1-comp p50
        logZ1_sed, logZ2_sed, dlogZ_sed, K_sed
        model_preferred_sed
        samp1_sed, samp2_sed                                      — numpy arrays
        kde_sed                                                   — gaussian_kde on samp2
        posterior_samples_sed_path, posterior_samples_1c_sed_path
        params_sed_path, plot_sed_path, plot_sed_residuals_path
        ms_atmosphere_sed, atmosphere_variants_sed
        systematics_atmosphere_sed, systematics_atmosphere_sed_path
        uncertainty_budget_sed, seeds_sed
    """
    if sed_exclude_windows_um is None:
        sed_exclude_windows_um = []

    gaia_id = int(gaia_id)
    tic_id  = int(tic_id)

    out_dir = Path(output_base) / f"TIC{tic_id}_Gaia{gaia_id}" / "sed"
    out_dir.mkdir(parents=True, exist_ok=True)

    if download_root is None:
        download_root = Path(output_base) / "downloads"

    print(f"\n{'='*60}")
    print(f"[SED] Gaia DR3 {gaia_id}  /  TIC {tic_id}  /  mode={mode}")
    print(f"{'='*60}")

    # ── Step 1: Fetch Gaia astrometry & VizieR SED ────────────────────────
    ra, dec, plx_mas, e_plx_mas = fetch_gaia_astrometry(gaia_id)
    _, sed_table = fetch_vizier_sed(ra, dec, gaia_id, download_root,
                                    vizier_radius_arcsec=vizier_radius_arcsec)
    if sed_table is None:
        raise RuntimeError(f"[SED] VizieR retrieval failed for Gaia {gaia_id}")

    # ── Step 2: Parse SED table ───────────────────────────────────────────
    wave_q    = _get_wave(sed_table)
    flux_q, eflux_q = _get_flux_err(sed_table, wave_q)

    wl_um_all    = wave_q.to(u.um).value.astype(float)
    flux_cgs_all = flux_q.to_value(u.erg / u.s / u.cm**2 / u.AA).astype(float)
    err_cgs_all  = (np.full_like(flux_cgs_all, np.nan) if eflux_q is None
                    else eflux_q.to_value(u.erg / u.s / u.cm**2 / u.AA).astype(float))
    band_all = _get_bandpass(sed_table)
    cat_all  = _get_catalog(sed_table)

    valid = np.isfinite(wl_um_all) & np.isfinite(flux_cgs_all) & (flux_cgs_all > 0)
    wl_um    = wl_um_all[valid].copy()
    flux_cgs = flux_cgs_all[valid].copy()
    err_cgs  = err_cgs_all[valid].copy()
    band     = band_all[valid].copy()
    cat      = cat_all[valid].copy()

    keep = _deduplicate(band, cat, err_cgs)
    wl_um, flux_cgs, err_cgs, band, cat = (a[keep] for a in (wl_um, flux_cgs, err_cgs, band, cat))

    # ── Step 3: SED filtering ─────────────────────────────────────────────
    n          = len(wl_um)
    reasons    = np.array([""] * n, dtype=object)
    gaia_mask  = np.array([_is_gaia(c, b) for c, b in zip(cat, band)], dtype=bool)
    target_msk = np.ones(n, dtype=bool) if filter_all_sed_points else gaia_mask

    mask_base = np.ones(n, dtype=bool)

    for lo, hi in sed_exclude_windows_um:
        m = (wl_um >= lo) & (wl_um <= hi)
        _append_reason(reasons, m, "exclude_window"); mask_base &= ~m

    if sed_exclude_below_um is not None:
        m = wl_um < sed_exclude_below_um
        _append_reason(reasons, m, "below_um"); mask_base &= ~m

    if sed_exclude_above_um is not None:
        m = wl_um > sed_exclude_above_um
        _append_reason(reasons, m, "above_um"); mask_base &= ~m

    # Points VizieR delivers without σ are recovered when the catalogue that
    # published them quotes an uncertainty: σ/F is read there and applied to
    # the VizieR flux, so the point enters the fit with its own uncertainty.
    sigma_recovered_sed = {}
    if recover_sigma_from_catalogs:
        missing   = ~(np.isfinite(err_cgs) & (err_cgs > 0))
        candidate = {str(b).strip() for b in band[missing]} & _SIGMA_LOOKUP_BANDS
        if candidate:
            rel_from_cat = fetch_catalog_rel_errors(
                gaia_id, ra_deg=ra, dec_deg=dec, wanted_bands=candidate,
                radius_arcsec=vizier_radius_arcsec)
            for i in np.where(missing)[0]:
                rel = rel_from_cat.get(str(band[i]).strip())
                if rel is not None and rel > 0 and flux_cgs[i] > 0:
                    err_cgs[i] = rel * flux_cgs[i]
                    sigma_recovered_sed[str(band[i]).strip()] = float(rel)
        if sigma_recovered_sed:
            detail = ", ".join(f"{b} ({r:.1%})"
                               for b, r in sorted(sigma_recovered_sed.items()))
            print(f"[SED] σ recovered from source catalogues: {detail}")

    # Photometry without a usable σ is dropped rather than assigned a
    # placeholder uncertainty; the σ/F cut then applies to every catalogue.
    has_sigma = np.isfinite(err_cgs) & (err_cgs > 0)
    _append_reason(reasons, ~has_sigma, "no_sigma"); mask_base &= has_sigma

    rel_err = np.full(n, np.nan)
    rel_err[has_sigma] = err_cgs[has_sigma] / flux_cgs[has_sigma]
    bad_rel = has_sigma & (rel_err > sed_max_rel_err)
    _append_reason(reasons, bad_rel, "rel_err"); mask_base &= ~bad_rel

    mask_used = mask_base.copy()
    if enable_outlier_filter:
        out_mask, _ = _iterative_robust_outlier_mask(
            wl_um,
            flux_cgs,
            target_msk,
            mask_used,
            max_iters=outlier_max_iters,
        )
        _append_reason(reasons, out_mask, "outlier")
        mask_used &= ~out_mask

        if not _keep_min_sampling(mask_used, wl_um):
            print("[SED] Sampling guard: reverting to no outlier rejection")
            mask_used = mask_base.copy()

    n_used = int(np.sum(mask_used))
    n_outlier = int(np.sum(np.array(["outlier" in r for r in reasons])))
    print(f"[SED] Points used={n_used}/{n}, rejected={n - n_used}  "
          f"(no σ: {int(np.sum(~has_sigma))}, "
          f"σ/F > {sed_max_rel_err:.0%}: {int(np.sum(bad_rel))}, "
          f"outlier: {n_outlier})")

    # Build fit arrays (wavelength in Å for planck function)
    # σ is floored at FRAC_FLOOR × F. Catalogues quote σ/F down to ~0.5 %, far
    # below the level at which they actually agree with one another: the same
    # star can differ by tens of percent from one survey to the next, and the
    # blackbody + CCM89 model is itself an approximation. Without the floor a
    # handful of over-precise, mutually inconsistent points dominate the
    # likelihood and drive A_V to zero.
    w_fit = wl_um[mask_used] * 1e4           # μm → Å
    f_fit = flux_cgs[mask_used].astype(float)
    e_fit = np.maximum(err_cgs[mask_used].astype(float),
                       np.maximum(FRAC_FLOOR * f_fit, ABS_FLOOR))
    band_fit = band[mask_used]

    if w_fit.size < 6:
        raise RuntimeError(f"[SED] Only {w_fit.size} points after filtering (need ≥ 6)")

    # ── Step 3b: Companion spectral models to fit ─────────────────────────
    ref_key = _resolve_atmo_key(ms_atmosphere)
    if compare_atmospheres is None:
        variant_keys = [ref_key]
    elif isinstance(compare_atmospheres, str) and compare_atmospheres.lower() == "all":
        variant_keys = list(SED_ATMO_VARIANTS)
    else:
        variant_keys = [_resolve_atmo_key(k) for k in compare_atmospheres]
    if ref_key not in variant_keys:
        variant_keys.insert(0, ref_key)
    variant_keys = list(dict.fromkeys(variant_keys))

    av_nodes = np.linspace(AV_MIN, AV_MAX, ATMO_AV_NODES)
    wd_key = _resolve_wd_atmo_key(wd_atmosphere)
    print(f"[SED] White-dwarf model '{wd_key}': {WD_ATMO_VARIANTS[wd_key]['label']}")
    models = {}
    for key in variant_keys:
        print(f"[SED] Companion model '{key}': {SED_ATMO_VARIANTS[key]['label']}")
        models[key] = _SedModel(
            key, w_fit, band_fit, av_nodes=av_nodes, cache_dir=atmo_cache_dir,
            grid_ranges=atmo_grid_ranges, ms_logg_mode=ms_logg_mode,
            ms_logg_fixed=ms_logg_fixed, wd_atmosphere=wd_key,
            wd_logg_mode=wd_logg_mode, wd_logg_fixed=wd_logg_fixed,
            wd_cooling_track=wd_cooling_track, t_ms_min=t_ms_min,
            t_ms_max=t_ms_max, verbose=True)
    ref_model = models[ref_key]

    # Every variant is sampled with the same T_MS prior. An atmosphere grid
    # that stops at 2300 K would otherwise fit a narrower prior than the
    # blackbody, and part of the parameter shift between them would be the
    # prior rather than the physics under test.
    t_ms_lo = max(m.t_ms_bounds[0] for m in models.values())
    t_ms_hi = min(m.t_ms_bounds[1] for m in models.values())
    if t_ms_hi <= t_ms_lo:
        raise RuntimeError("[SED] The requested companion models have no common T_MS range")
    if (t_ms_lo, t_ms_hi) != (float(t_ms_min), float(t_ms_max)):
        print(f"[SED] WARNING: T_MS prior narrowed from the requested "
              f"[{float(t_ms_min):.0f}, {float(t_ms_max):.0f}] K to "
              f"[{t_ms_lo:.0f}, {t_ms_hi:.0f}] K (common to all companion grids)")
    else:
        print(f"[SED] T_MS prior: [{t_ms_lo:.0f}, {t_ms_hi:.0f}] K")

    t_wd_lo = max(m.t_wd_bounds[0] for m in models.values())
    t_wd_hi = min(m.t_wd_bounds[1] for m in models.values())
    if t_wd_hi <= t_wd_lo:
        raise RuntimeError("[SED] The requested white-dwarf model has no usable T_WD range")
    if (t_wd_lo, t_wd_hi) != (T_WD_MIN, T_WD_MAX):
        print(f"[SED] T_WD prior clipped to [{t_wd_lo:.0f}, {t_wd_hi:.0f}] K "
              f"(extent of the {WD_ATMO_VARIANTS[wd_key]['label']} grid)")

    # ── Step 4: Gaussian prior on the Gaia parallax ───────────────────────
    # The published Gaia DR3 parallax and uncertainty are used as they are:
    # no zero-point correction and no uncertainty inflation is applied. Both
    # may be considered in a future improvement; their effect on this sample
    # has not been evaluated.
    # The prior is applied on ϖ itself, not on d = 1000/ϖ: the transformation
    # is non-linear, so a symmetric Gaussian in d misrepresents the likelihood
    # as soon as σ_ϖ/ϖ is not negligible.
    if np.isfinite(plx_mas) and plx_mas > 0 and np.isfinite(e_plx_mas) and e_plx_mas > 0:
        plx0_mas  = plx_mas
        splx_mas  = max(e_plx_mas, MIN_PLX_SIGMA)
    else:
        print("[SED] Invalid parallax — falling back to ϖ ~ N(5.0, 3.75) mas")
        plx0_mas, splx_mas = 5.0, 3.75

    plx_min_mas = max(PLX_MIN_MAS, plx0_mas - N_SIGMA_PLX * splx_mas)
    plx_max_mas = plx0_mas + N_SIGMA_PLX * splx_mas
    _ln_plx_norm = -0.5 * np.log(2.0 * np.pi * splx_mas**2)
    print(f"[SED] Parallax prior: ϖ ~ N({plx0_mas:.4f}, {splx_mas:.4f}) mas  "
          f"[{plx_min_mas:.4f}, {plx_max_mas:.4f}]  "
          f"→ d ∈ [{1000.0 / plx_max_mas:.1f}, {1000.0 / plx_min_mas:.1f}] pc")

    def _ln_plx_prior(plx):
        return _ln_plx_norm - 0.5 * ((plx - plx0_mas) / splx_mas)**2

    # ── Step 5: UltraNest 1-component ─────────────────────────────────────
    print("\n[SED] ── 1-component fit ──")

    def prior_1c(uvec):
        uT, uR, uP, uA = uvec
        T   = t_wd_lo + uT * (t_wd_hi - t_wd_lo)
        R   = np.exp(np.log(R_WD_MIN_RSUN) + uR * np.log(R_WD_MAX_RSUN / R_WD_MIN_RSUN))
        plx = plx_min_mas + uP * (plx_max_mas - plx_min_mas)
        A   = AV_MIN  + uA * (AV_MAX   - AV_MIN)
        return np.array([T, R, plx, A])

    def loglike_1c(params):
        T_WD, R_WD, plx, A_V = params
        d_pc = 1000.0 / plx
        return (_ln_plx_prior(plx)
                + _lnlike_gauss(ref_model.bands_1comp([T_WD, R_WD, d_pc, A_V]), f_fit, e_fit))

    # A variant-specific log_dir keeps runs made under different models apart.
    # (With UltraNest's default resume="subfolder", checked on UltraNest 4.5.0,
    # each run writes a new run<N>/ subfolder rather than resuming an old one.)
    suffix_1c = "" if ref_key == "bb" else f"_{ref_key}"
    # UltraNest draws from NumPy's global random state: seed it right before
    # each sampler is built and run (see the seed argument).
    seeds_sed = {"master": int(seed), "1comp": derive_seed(seed, f"sed:1comp:{ref_key}"),
                 "2comp": {}}
    np.random.seed(seeds_sed["1comp"])
    sampler_1c = ReactiveNestedSampler(
        ["T_WD", "R_WD_Rsun", "plx_mas", "AV"],
        loglike_1c, prior_1c,
        log_dir=str(out_dir / f"ultranest_1c{suffix_1c}"),
    )
    run_kw = dict(max_iters=1_000_000, frac_remain=0.01,
                  Lepsilon=0.1, max_num_improvement_loops=3)
    run_kw.update(SED_SAMPLING["test" if mode == "test" else "final"])

    sampler_1c.run(**run_kw)
    sampler_1c.print_results()

    post1 = sampler_1c.results
    samp1 = np.array(post1["samples"], dtype=float)      # [N, 4]: T_WD, R_WD, ϖ, AV
    samp1[:, 2] = 1000.0 / samp1[:, 2]                   # ϖ [mas] → d [pc]
    logZ1 = float(post1["logz"])
    elogZ1 = float(post1["logzerr"])

    p16_1c = np.percentile(samp1, 16, axis=0)
    p50_1c = np.percentile(samp1, 50, axis=0)
    p84_1c = np.percentile(samp1, 84, axis=0)
    T_WD_1c_p50, R_WD_1c_p50, d_1c_p50, AV_1c_p50 = p50_1c

    # ── Step 6: UltraNest 2-component ─────────────────────────────────────
    # ϖ keeps its own Gaia prior, but A_V is anchored on the 1-component
    # result. Reddening is set by the white dwarf: it dominates the blue end
    # of the SED, which is where the CCM89 law bites, while the cool companion
    # only shows up in the red. Left flat over [AV_MIN, AV_MAX], the 2-comp fit
    # trades extinction against companion flux and drifts to an unphysically
    # low A_V. The 1-comp A_V posterior is therefore reused as a Gaussian,
    # truncated to [AV_MIN, AV_MAX] by the uniform prior transform below.
    av2_mu    = float(AV_1c_p50)
    av2_sigma = max(AV_2C_SIGMA_MIN, 0.5 * float(p84_1c[3] - p16_1c[3]))
    _ln_av2_norm = -0.5 * np.log(2.0 * np.pi * av2_sigma**2)

    def _ln_av2_prior(A_V):
        return _ln_av2_norm - 0.5 * ((A_V - av2_mu) / av2_sigma)**2

    print(f"\n[SED] ── 2-component fits (ϖ free, "
          f"A_V ~ N({av2_mu:.4f}, {av2_sigma:.4f}) from 1-comp) ──")

    def prior_2c(uvec):
        uT1, uR1, uT2, uR2, uP, uA = uvec
        T1  = t_wd_lo     + uT1 * (t_wd_hi     - t_wd_lo)
        R1  = np.exp(np.log(R_WD_MIN_RSUN) + uR1 * np.log(R_WD_MAX_RSUN / R_WD_MIN_RSUN))
        T2  = t_ms_lo     + uT2 * (t_ms_hi     - t_ms_lo)
        R2  = R_MS_MIN_RSUN + uR2 * (R_MS_MAX_RSUN - R_MS_MIN_RSUN)
        plx = plx_min_mas + uP * (plx_max_mas - plx_min_mas)
        A   = AV_MIN  + uA * (AV_MAX  - AV_MIN)
        return np.array([T1, R1, T2, R2, plx, A])

    def _make_loglike_2c(model):
        def loglike_2c(params):
            T_WD, R_WD, T_MS, R_MS, plx, A_V = params
            d_pc = 1000.0 / plx
            return (_ln_plx_prior(plx)
                    + _ln_av2_prior(A_V)
                    + _lnlike_gauss(
                        model.bands_2comp([T_WD, R_WD, T_MS, R_MS, d_pc, A_V]),
                        f_fit, e_fit))
        return loglike_2c

    runs = {}
    for key in variant_keys:
        model = models[key]
        print(f"\n[SED] 2-component fit — {model.label}")
        suffix_2c = "" if key == "bb" else f"_{key}"
        seeds_sed["2comp"][key] = derive_seed(seed, f"sed:2comp:{key}")
        np.random.seed(seeds_sed["2comp"][key])
        sampler_2c = ReactiveNestedSampler(
            ["T_WD", "R_WD_Rsun", "T_MS", "R_MS_Rsun", "plx_mas", "AV"],
            _make_loglike_2c(model), prior_2c,
            log_dir=str(out_dir / f"ultranest_2c{suffix_2c}"),
        )
        sampler_2c.run(**run_kw)
        sampler_2c.print_results()

        post = sampler_2c.results
        samples = np.array(post["samples"], dtype=float)   # [N, 6]: …, ϖ, AV
        samples[:, 4] = 1000.0 / samples[:, 4]             # ϖ [mas] → d [pc]
        p16 = np.percentile(samples, 16, axis=0)
        p50 = np.percentile(samples, 50, axis=0)
        p84 = np.percentile(samples, 84, axis=0)
        chi = (f_fit - model.bands_2comp(p50)) / e_fit
        dof = int(len(f_fit) - 6)
        runs[key] = {
            "label":     model.label,
            "ms_model":  model.ms_model,
            "band_integrated": model.band_integrated,
            "samples":   samples,
            "p16": p16, "p50": p50, "p84": p84,
            "logZ":  float(post["logz"]),
            "elogZ": float(post["logzerr"]),
            "chi2":  float(np.sum(chi ** 2)),
            "dof":   dof,
            "chi2_red": float(np.sum(chi ** 2) / dof) if dof > 0 else None,
        }
        print(f"[SED]   {model.label}: T_WD={p50[0]:.0f} K  T_MS={p50[2]:.0f} K  "
              f"R_MS={p50[3]:.4f} R_sun  logZ={runs[key]['logZ']:.3f}  "
              f"χ²_red={runs[key]['chi2_red']:.2f}")

    ref_run = runs[ref_key]
    samp2   = ref_run["samples"]
    logZ2   = ref_run["logZ"]
    elogZ2  = ref_run["elogZ"]
    p16_2c, p50_2c, p84_2c = ref_run["p16"], ref_run["p50"], ref_run["p84"]
    T_WD_p50, R_WD_p50, T_MS_p50, R_MS_p50, d_p50, AV_p50 = p50_2c

    dlogZ_sed = logZ2 - logZ1
    K_sed     = float(np.exp(dlogZ_sed))
    if   dlogZ_sed > 5.0:   strength = "very strong"
    elif dlogZ_sed > 2.5:   strength = "strong"
    elif dlogZ_sed > 1.0:   strength = "substantial"
    else:                   strength = "weak"
    model_preferred_sed = "2comp" if dlogZ_sed > 0 else "1comp"

    print(f"\n[SED] logZ1={logZ1:.3f}±{elogZ1:.3f}  "
          f"logZ2={logZ2:.3f}±{elogZ2:.3f}  "
          f"ΔlogZ={dlogZ_sed:.3f}  K={K_sed:.2e}  → {model_preferred_sed} ({strength})")

    # ── Step 7: Save posteriors ───────────────────────────────────────────
    path_samp1 = out_dir / "posterior_samples_1c_sed.npy"
    path_samp2 = out_dir / "posterior_samples_sed.npy"
    np.save(path_samp1, samp1)
    np.save(path_samp2, samp2)
    variant_posterior_paths = {}
    for key, run in runs.items():
        # The reference posterior keeps its historical filename: the dilution
        # and joint stages read it by name.
        path = (path_samp2 if key == ref_key
                else out_dir / f"posterior_samples_sed_{key}.npy")
        np.save(path, run["samples"])
        variant_posterior_paths[key] = str(path)
    print(f"[SED] Saved posteriors → {out_dir}/")

    # ── Step 7b: Atmosphere-model systematics ─────────────────────────────
    systematics_sed, uncertainty_budget_sed = _atmosphere_systematics(runs, ref_key)
    systematics_path = None
    if len(runs) > 1:
        systematics_path = out_dir / "systematics_atmosphere_sed.json"
        with open(systematics_path, "w", encoding="utf-8") as fp:
            json.dump(systematics_sed, fp, indent=2)
        print(f"[SED] Saved atmosphere systematics → {systematics_path}")
        _print_systematics_table(systematics_sed, uncertainty_budget_sed)

    # ── Step 8: Save params_sed.json ──────────────────────────────────────
    def _pd(p16, p50, p84, names):
        return {name: {"p16": float(p16[i]), "p50": float(p50[i]), "p84": float(p84[i])}
                for i, name in enumerate(names)}

    # χ² diagnostics at the posterior median, using the same points and
    # effective uncertainties as the SED likelihood (including FRAC_FLOOR /
    # ABS_FLOOR already applied in e_fit).
    resid_1c = (f_fit - ref_model.bands_1comp(p50_1c)) / e_fit
    chi2_1c = float(np.sum(resid_1c ** 2))
    n_data_1c = int(len(f_fit))
    dof_1c = int(n_data_1c - 4)
    chi2_red_1c = float(chi2_1c / dof_1c) if dof_1c > 0 else None

    chi2_2c     = ref_run["chi2"]
    n_data_2c   = int(len(f_fit))
    dof_2c      = ref_run["dof"]
    chi2_red_2c = ref_run["chi2_red"]

    params_sed = {
        "gaia_id": gaia_id,
        "tic_id":  tic_id,
        "1comp": {
            **_pd(p16_1c, p50_1c, p84_1c,
                  ["T_WD_1c_sed", "R_WD_1c_sed", "d_1c_sed", "AV_1c_sed"]),
            "logZ_sed":  logZ1,
            "elogZ_sed": elogZ1,
            "n_sed_points": int(n_data_1c),
            "chi2_1c_sed": float(chi2_1c),
            "dof_1c_sed": int(dof_1c),
            "chi2_red_1c_sed": chi2_red_1c,
        },
        "2comp": {
            **_pd(p16_2c, p50_2c, p84_2c,
                  ["T_WD_sed", "R_WD_sed", "T_MS_sed", "R_MS_sed", "d_sed", "AV_sed"]),
            "logZ_sed":  logZ2,
            "elogZ_sed": elogZ2,
            "n_sed_points": int(n_data_2c),
            "chi2_2c_sed": float(chi2_2c),
            "dof_2c_sed": int(dof_2c),
            "chi2_red_2c_sed": chi2_red_2c,
        },
        "dlogZ_sed":          float(dlogZ_sed),
        "K_sed":              K_sed,
        "evidence_strength":  strength,
        "model_preferred_sed": model_preferred_sed,
        "sigma_from_catalog_sed": sigma_recovered_sed,
        "parallax_prior_sed": {
            "plx0_mas":    float(plx0_mas),
            "e_plx_mas":   float(splx_mas),
            "plx_min_mas": float(plx_min_mas),
            "plx_max_mas": float(plx_max_mas),
        },
        "av_prior_sed": {
            "1comp": {"kind": "uniform",
                      "min": float(AV_MIN), "max": float(AV_MAX)},
            "2comp": {"kind": "gaussian_from_1comp",
                      "mu": float(av2_mu), "sigma": float(av2_sigma),
                      "min": float(AV_MIN), "max": float(AV_MAX)},
        },
        "wd_model_sed": {
            "reference": wd_key,
            "label": WD_ATMO_VARIANTS[wd_key]["label"],
            "wd_model": ref_model.wd_model,
            "wd_logg_mode": (str(wd_logg_mode) if wd_key != "blackbody" else "not applicable"),
            "wd_logg_fixed": float(wd_logg_fixed),
            "t_wd_prior_K": [float(t_wd_lo), float(t_wd_hi)],
        },
        "companion_model_sed": {
            "reference": ref_key,
            "label": ref_run["label"],
            "ms_model": ref_run["ms_model"],
            "band_integrated": ref_run["band_integrated"],
            "ms_logg_mode": str(ms_logg_mode),
            "t_ms_requested_K": [float(t_ms_min), float(t_ms_max)],
            "t_ms_prior_K": [float(t_ms_lo), float(t_ms_hi)],
            "variants_compared": list(runs),
        },
        "uncertainty_budget_sed": uncertainty_budget_sed,
        "sed_max_rel_err_sed": float(sed_max_rel_err),
        "seeds_sed": seeds_sed,
    }
    params_path = out_dir / "params_sed.json"
    with open(params_path, "w", encoding="utf-8") as fp:
        json.dump(params_sed, fp, indent=2)
    print(f"[SED] Saved params → {params_path}")

    # ── Step 9: Plots ─────────────────────────────────────────────────────
    # Points without σ take no part in the fit and are not displayed at all.
    plot_kw = dict(
        wl_um=wl_um[has_sigma], flux_cgs=flux_cgs[has_sigma],
        err_cgs=err_cgs[has_sigma], mask_used=mask_used[has_sigma],
        w_fit=w_fit, f_fit=f_fit, e_fit=e_fit,
        samp1=samp1, samp2=samp2,
        p50_1c=p50_1c, p50_2c=p50_2c,
        model=ref_model,
        show_excluded_sed_points=show_excluded_sed_points,
    )
    plot_path = out_dir / "plot_sed.pdf"
    _plot_sed(out_path=plot_path, with_residuals=False, **plot_kw)

    plot_resid_path = out_dir / "plot_sed_residuals.pdf"
    _plot_sed(out_path=plot_resid_path, with_residuals=True, **plot_kw)

    plot_compare_path = None
    if len(runs) > 1:
        plot_compare_path = out_dir / "plot_sed_atmosphere_comparison.pdf"
        _plot_atmosphere_comparison(
            wl_um=wl_um[has_sigma], flux_cgs=flux_cgs[has_sigma],
            err_cgs=err_cgs[has_sigma], mask_used=mask_used[has_sigma],
            w_fit=w_fit, f_fit=f_fit, e_fit=e_fit,
            models=models, runs=runs, ref_key=ref_key,
            systematics=systematics_sed, out_path=plot_compare_path,
        )

    # ── Step 10: Build KDE on 2-comp posterior (for joint_fit.py) ─────────
    # Columns: [T_WD, R_WD, T_MS, R_MS, d, AV] — all 6 dimensions
    kde_sed = gaussian_kde(samp2.T)

    # ── Return ─────────────────────────────────────────────────────────────
    return {
        "gaia_id": gaia_id,
        "tic_id":  tic_id,
        # 1-comp medians
        "T_WD_1c_sed": float(T_WD_1c_p50),
        "R_WD_1c_sed": float(R_WD_1c_p50),
        "d_1c_sed":    float(d_1c_p50),
        "AV_1c_sed":   float(AV_1c_p50),
        # 2-comp medians
        "T_WD_sed": float(T_WD_p50),
        "R_WD_sed": float(R_WD_p50),
        "T_MS_sed": float(T_MS_p50),
        "R_MS_sed": float(R_MS_p50),
        "d_sed":    float(d_p50),
        "AV_sed":   float(AV_p50),
        # Bayesian evidence
        "logZ1_sed":  logZ1,
        "logZ2_sed":  logZ2,
        "dlogZ_sed":  float(dlogZ_sed),
        "K_sed":      K_sed,
        "model_preferred_sed": model_preferred_sed,
        # Full posteriors (numpy arrays, shape [N,4] and [N,6])
        "samp1_sed": samp1,
        "samp2_sed": samp2,
        # KDE on samp2 for joint_fit.py
        "kde_sed": kde_sed,
        # Companion-model systematics
        "ms_atmosphere_sed":          ref_key,
        "atmosphere_variants_sed":    {k: {kk: vv for kk, vv in r.items()
                                           if kk not in ("samples", "p16", "p50", "p84")}
                                       for k, r in runs.items()},
        "systematics_atmosphere_sed": systematics_sed,
        "uncertainty_budget_sed":     uncertainty_budget_sed,
        "seeds_sed":                  seeds_sed,
        # File paths
        "posterior_samples_sed_path":    str(path_samp2),
        "posterior_samples_1c_sed_path": str(path_samp1),
        "posterior_samples_by_variant_sed": variant_posterior_paths,
        "params_sed_path":               str(params_path),
        "plot_sed_path":                 str(plot_path),
        "plot_sed_residuals_path":       str(plot_resid_path),
        "plot_sed_atmosphere_comparison_path": (
            None if plot_compare_path is None else str(plot_compare_path)),
        "systematics_atmosphere_sed_path": (
            None if systematics_path is None else str(systematics_path)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Companion-model systematics
# ══════════════════════════════════════════════════════════════════════════════

def _atmosphere_systematics(runs, ref_key):
    """Turn a set of same-data fits into a stat + syst uncertainty budget.

    The nested sampler returns the width of the posterior under one assumed
    spectral model; it says nothing about that assumption being right. Refitting
    the same photometry under a different companion model and taking the largest
    median-to-median shift gives a systematic term that can be quoted next to
    the statistical one.
    """
    ref = runs[ref_key]
    variants = {}
    for key, run in runs.items():
        variants[key] = {
            "label": run["label"],
            "ms_model": run["ms_model"],
            "band_integrated": run["band_integrated"],
            "logZ": run["logZ"],
            "elogZ": run["elogZ"],
            "chi2": run["chi2"],
            "dof": run["dof"],
            "chi2_red": run["chi2_red"],
            "params": {
                name: {"p16": float(run["p16"][i]),
                       "p50": float(run["p50"][i]),
                       "p84": float(run["p84"][i])}
                for i, name in enumerate(PARAM_NAMES_2C)
            },
        }

    shifts = {}
    for key, run in runs.items():
        if key == ref_key:
            continue
        entry = {}
        for i, name in enumerate(PARAM_NAMES_2C):
            sigma_stat = 0.5 * float(ref["p84"][i] - ref["p16"][i])
            delta = float(run["p50"][i] - ref["p50"][i])
            entry[name] = {
                "delta": delta,
                "delta_rel": (delta / float(ref["p50"][i])
                              if ref["p50"][i] != 0 else None),
                "delta_over_sigma_stat": (delta / sigma_stat
                                          if sigma_stat > 0 else None),
            }
        entry["dlogZ_vs_reference"] = float(run["logZ"] - ref["logZ"])
        shifts[key] = entry

    budget = {
        "reference_variant": ref_key,
        "reference_label": ref["label"],
        "variants_compared": list(runs),
        "definition": (
            "sigma_stat = half the 16-84 credible interval of the reference "
            "nested-sampling posterior; sigma_syst = largest shift of the "
            "median across companion spectral models; sigma_total = quadrature "
            "sum of the two."
        ),
        "params": {},
    }
    for i, name in enumerate(PARAM_NAMES_2C):
        sigma_stat = 0.5 * float(ref["p84"][i] - ref["p16"][i])
        deltas = [abs(float(run["p50"][i] - ref["p50"][i]))
                  for key, run in runs.items() if key != ref_key]
        sigma_syst = max(deltas) if deltas else 0.0
        budget["params"][name] = {
            "p50": float(ref["p50"][i]),
            "p16": float(ref["p16"][i]),
            "p84": float(ref["p84"][i]),
            "sigma_stat": sigma_stat,
            "sigma_syst": float(sigma_syst),
            "sigma_total": float(np.hypot(sigma_stat, sigma_syst)),
            "syst_over_stat": (float(sigma_syst / sigma_stat)
                               if sigma_stat > 0 else None),
        }

    systematics = {
        "reference": ref_key,
        "variants": variants,
        "shifts_vs_reference": shifts,
        "budget": budget,
    }
    return systematics, budget


def _print_systematics_table(systematics, budget):
    """Console summary of the companion-model comparison."""
    ref = systematics["reference"]
    print(f"\n[SED] ── Companion-model systematics (reference: {ref}) ──")

    header = f"  {'variant':<10s} {'logZ':>10s} {'chi2_red':>9s}  " + \
             "  ".join(f"{n.replace('_sed',''):>10s}" for n in PARAM_NAMES_2C)
    print(header)
    for key, info in systematics["variants"].items():
        values = "  ".join(f"{info['params'][n]['p50']:>10.4g}" for n in PARAM_NAMES_2C)
        chi2_red = info["chi2_red"]
        print(f"  {key:<10s} {info['logZ']:>10.2f} "
              f"{(chi2_red if chi2_red is not None else float('nan')):>9.2f}  {values}")

    print(f"  {'-' * (len(header) - 2)}")
    print("  shift vs reference, in units of the statistical sigma:")
    for key, entry in systematics["shifts_vs_reference"].items():
        values = "  ".join(
            f"{(entry[n]['delta_over_sigma_stat'] or 0.0):>+10.2f}" for n in PARAM_NAMES_2C)
        print(f"  {key:<10s} {'':>10s} {'':>9s}  {values}")

    print("  systematic / statistical uncertainty ratio:")
    ratios = "  ".join(
        f"{(budget['params'][n]['syst_over_stat'] or 0.0):>10.2f}" for n in PARAM_NAMES_2C)
    print(f"  {'':<10s} {'':>10s} {'':>9s}  {ratios}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot helper
# ══════════════════════════════════════════════════════════════════════════════

def _plot_sed(wl_um, flux_cgs, err_cgs, mask_used,
              w_fit, f_fit, e_fit, samp1, samp2, p50_1c, p50_2c, out_path,
              model=None, with_residuals=False, show_excluded_sed_points=True):
    """SED figure. With with_residuals=True, add a 2-comp residual panel."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    rng   = np.random.default_rng(42)
    wplot = np.logspace(np.log10(np.nanmin(w_fit) * 0.8),
                        np.log10(np.nanmax(w_fit) * 1.2), 900)

    spec_1comp = _model_1comp if model is None else model.spectrum_1comp
    spec_2comp = _model_2comp if model is None else model.spectrum_2comp

    def _posterior_band(samples, model_fn, n_draw=400):
        idx = rng.choice(len(samples), size=min(n_draw, len(samples)), replace=False)
        M   = np.array([model_fn(samples[i], wplot) for i in idx])
        return np.nanpercentile(M, 16, axis=0), np.nanpercentile(M, 84, axis=0)

    # Median models
    m1_med = spec_1comp(p50_1c, wplot)
    m2_med = spec_2comp(p50_2c, wplot)

    # Decompose 2-comp into WD-only and MS-only at medians
    if model is None:
        T_WD_2c, R_WD_2c, _, _, d_2c, AV_2c = p50_2c
        wd_only_med = _model_1comp([T_WD_2c, R_WD_2c, d_2c, AV_2c], wplot)
        ms_only_med = m2_med - wd_only_med
    else:
        wd_only_med, ms_only_med = model.spectrum_components(p50_2c, wplot)

    lo1, hi1 = _posterior_band(samp1, spec_1comp)
    lo2, hi2 = _posterior_band(samp2, spec_2comp)

    if with_residuals:
        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(7.2, 5.9), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)
    else:
        fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        axr = None

    ax_bottom = axr if axr is not None else ax
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax_bottom.set_xlabel(r"Wavelength ($\mu$m)", fontsize=14)
    ax.set_ylabel(r"Flux (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)", fontsize=14)

    ticks_A = np.array([0.1, 0.3, 1.0, 3.0]) * 1e4
    ax_bottom.xaxis.set_major_locator(FixedLocator(ticks_A))
    ax_bottom.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e4:g}"))
    ax_bottom.xaxis.set_minor_formatter(NullFormatter())

    # Used photometry (all displayed points have a measured σ)
    w_used = wl_um[mask_used] * 1e4
    f_used = flux_cgs[mask_used]
    e_used = err_cgs[mask_used]

    ax.errorbar(w_used, f_used, yerr=e_used,
                fmt="o", ms=4, mfc="white", mec="k", ecolor="k",
                elinewidth=0.8, capsize=2, alpha=0.9,
                label="SED", zorder=10)

    # Rejected photometry (optional display + legend entry)
    if show_excluded_sed_points and (~mask_used).any():
        ax.scatter(wl_um[~mask_used] * 1e4, flux_cgs[~mask_used],
                   marker="x", s=35, linewidths=1.0, color="crimson",
                   alpha=0.8, label="Excluded", zorder=12)

    # Model curves
    T1_1c = p50_1c[0]; T1_2c = p50_2c[0]; T2_2c = p50_2c[2]

    ax.plot(wplot, m1_med, lw=1.8, color="cornflowerblue",
            label=f"1-comp WD: {T1_1c:.0f} K")
    ax.fill_between(wplot, lo1, hi1, alpha=0.20, color="cornflowerblue", lw=0)

    ax.plot(wplot, m2_med, lw=1.9, color="#ff7f0e",
            label=f"2-comp: {T1_2c:.0f} K + {T2_2c:.0f} K")
    ax.fill_between(wplot, lo2, hi2, alpha=0.20, color="darkorange", lw=0)

    ax.plot(wplot, ms_only_med, lw=1.1, ls="--", color="darkorange",
            alpha=0.85, label="MS component")

    ax.legend(frameon=False, ncol=2, fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(3_200, 60_000)
    ax.set_ylim(3e-19, 2e-15)

    if axr is not None:
        # Residuals use the band-integrated prediction, i.e. the quantity the
        # likelihood actually compares against, not the plotted spectrum.
        model_at_fit = (_model_2comp(p50_2c, w_fit) if model is None
                        else model.bands_2comp(p50_2c))
        chi_2c = (f_fit - model_at_fit) / e_fit

        axr.axhspan(-1.0, 1.0, color="0.85", zorder=0, lw=0)
        axr.axhline(0.0, color="k", lw=0.8, zorder=1)
        axr.plot(w_fit, chi_2c, "o", ms=4, mfc="white",
                 mec="#ff7f0e", mew=1.1, label="2-comp", zorder=5)

        span = max(float(np.nanmax(np.abs(chi_2c))), 3.0)
        axr.set_ylim(-1.15 * span, 1.15 * span)
        axr.set_ylabel(r"$(F - F_{\rm mod})\,/\,\sigma$", fontsize=13)
        axr.legend(frameon=False, fontsize=11)
        axr.grid(True, alpha=0.25)

    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[SED] Saved plot → {out_path}")


def _plot_atmosphere_comparison(wl_um, flux_cgs, err_cgs, mask_used,
                                w_fit, f_fit, e_fit, models, runs, ref_key,
                                systematics, out_path):
    """Three panels: SED overlay, residuals per model, parameter shifts.

    The bottom panel is the one that answers "how much does the companion model
    matter": each bar is the median shift relative to the reference, expressed
    in units of that reference's statistical uncertainty, so a bar shorter than
    1 means the systematic is buried in the nested-sampling error bar.
    """
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    keys = list(runs)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(keys)))
    wplot = np.logspace(np.log10(np.nanmin(w_fit) * 0.8),
                        np.log10(np.nanmax(w_fit) * 1.2), 900)

    fig, (ax, axr, axb) = plt.subplots(
        3, 1, figsize=(7.6, 8.6),
        gridspec_kw={"height_ratios": [3, 1.2, 1.8]}, constrained_layout=True)

    # ── Panel 1: SED and models ────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"Flux (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)", fontsize=14)
    ax.errorbar(wl_um[mask_used] * 1e4, flux_cgs[mask_used], yerr=err_cgs[mask_used],
                fmt="o", ms=4, mfc="white", mec="k", ecolor="k",
                elinewidth=0.8, capsize=2, alpha=0.9, label="Photometry", zorder=10)

    for color, key in zip(colors, keys):
        run, model = runs[key], models[key]
        total = model.spectrum_2comp(run["p50"], wplot)
        _, companion = model.spectrum_components(run["p50"], wplot)
        ax.plot(wplot, total, lw=1.7, color=color,
                label=f"{run['label']}: {run['p50'][0]:.0f} K + {run['p50'][2]:.0f} K")
        ax.plot(wplot, companion, lw=1.0, ls="--", color=color, alpha=0.8)

    ax.legend(frameon=False, fontsize=11, ncol=1, loc="lower left")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0.8 * float(np.nanmin(w_fit)), 1.3 * float(np.nanmax(w_fit)))
    # An atmosphere companion collapses by many decades in the blue; without an
    # explicit range the photometry is squeezed into the top of the panel.
    ax.set_ylim(0.02 * float(np.nanmin(f_fit)), 20.0 * float(np.nanmax(f_fit)))

    # ── Panel 2: residuals ─────────────────────────────────────────────────
    axr.set_xscale("log")
    axr.axhspan(-1.0, 1.0, color="0.88", zorder=0, lw=0)
    axr.axhline(0.0, color="k", lw=0.8, zorder=1)
    for color, key in zip(colors, keys):
        chi = (f_fit - models[key].bands_2comp(runs[key]["p50"])) / e_fit
        axr.plot(w_fit, chi, "o", ms=4, mfc="none", mec=color, mew=1.2,
                 label=f"{key} (χ²/ν = {runs[key]['chi2_red']:.2f})", zorder=5)
    axr.set_ylabel(r"$(F-F_{\rm mod})/\sigma$", fontsize=13)
    axr.set_xlabel(r"Wavelength ($\mu$m)", fontsize=14)
    axr.legend(frameon=False, fontsize=10.5, ncol=len(keys),
               loc="lower center", bbox_to_anchor=(0.5, 1.0))
    axr.grid(True, alpha=0.25)
    ticks_A = np.array([0.1, 0.3, 1.0, 3.0]) * 1e4
    for a in (ax, axr):
        a.xaxis.set_major_locator(FixedLocator(ticks_A))
        a.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e4:g}"))
        a.xaxis.set_minor_formatter(NullFormatter())
    axr.set_xlim(ax.get_xlim())

    # ── Panel 3: parameter shifts, in units of the statistical sigma ───────
    others = [k for k in keys if k != ref_key]
    labels = [n.replace("_sed", "") for n in PARAM_NAMES_2C]
    x = np.arange(len(PARAM_NAMES_2C))
    width = 0.8 / max(1, len(others))
    for i, key in enumerate(others):
        entry = systematics["shifts_vs_reference"][key]
        values = [entry[n]["delta_over_sigma_stat"] or 0.0 for n in PARAM_NAMES_2C]
        color = colors[keys.index(key)]
        axb.bar(x + (i - (len(others) - 1) / 2) * width, values, width * 0.92,
                color=color, alpha=0.85, label=f"{key} − {ref_key}")

    axb.axhspan(-1.0, 1.0, color="0.88", zorder=0, lw=0)
    axb.axhline(0.0, color="k", lw=0.8)
    axb.set_xticks(x)
    axb.set_xticklabels(labels, fontsize=9)
    axb.set_ylabel(r"$\Delta$ median / $\sigma_{\rm stat}$", fontsize=13)
    axb.legend(frameon=False, fontsize=11)
    axb.grid(True, axis="y", alpha=0.25)

    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[SED] Saved atmosphere comparison plot → {out_path}")
