"""Synthetic stellar atmospheres and band-integrated photometry for SED fitting.

Replaces the blackbody approximation with PHOENIX / BT-Settl grids. For cool
stars (M, K) the spectrum is dominated by molecular absorption (TiO, VO, H2O)
and H- opacity: a blackbody biases Teff by hundreds of K and underestimates the
radius.

Three grid sources are available:

  "phoenix-stsci"   PHOENIX (Allard) grid distributed by STScI.
                    Teff 2000-6500 K in 100 K steps, logg 0.0-5.5 in 0.5 dex
                    steps, solar metallicity. One FITS file per Teff (~12 MB)
                    containing all logg values. Downloaded on first use.

  "bt-settl-cifist" BT-Settl CIFIST via the SVO Theoretical Spectra Server.
                    Best reference for late M dwarfs, but ~66 MB per
                    (Teff, logg) pair: a full download is slow.

  "koester-da"      Koester DA white-dwarf spectra, read from the local
                    directory KOESTER_DIR_DEFAULT (never downloaded).

All grids provide surface flux densities F_lambda in erg/s/cm2/A. The observed
flux is therefore F_surface * (R/d)**2, exactly as for pi*B_lambda in the
previous blackbody model: the calling code's scale factor is unchanged.

Note that this STScI PHOENIX grid (SED term) is not the same file set as the
local PHOENIX SpecInt spectra in models/phoenix2021/ that dilution_fit.py uses
for the eclipse-depth term.

Cache (DEFAULT_CACHE_DIR, overridable with the SED_MODEL_CACHE environment
variable or the cache_dir arguments):

    <cache>/raw/<source>/    downloaded spectra (PHOENIX FITS, BT-Settl npz)
    <cache>/grids/           rebinned grids, one .npz per (source, Teff range,
                             logg range, resolution)
    <cache>/filters/         SVO filter transmissions, one .npz per filter

A cached grid is reused only when its file name matches exactly; changing a
grid range builds (and possibly downloads) a new one.

Typical usage:

    import sed_atmosphere_models as sam

    grid  = sam.SpectralGrid.load("phoenix-stsci", teff_range=(2300, 5900))
    bands = [sam.resolve_bandpass(name, lambda_eff_AA=w) for name, w in zip(filters, waves)]
    table = sam.BandPhotometry(grid, bands, av_nodes=np.linspace(0.0, 0.6, 9))

    flux_per_band = table(teff=3200.0, logg=5.0, a_v=0.12)   # erg/s/cm2/A, surface
"""

from __future__ import annotations

import io
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wdms_common import M_SUN_G

__all__ = [
    "DEFAULT_CACHE_DIR",
    "Bandpass",
    "SpectralGrid",
    "BandPhotometry",
    "BlackbodyBandPhotometry",
    "BlackbodyComponent",
    "Component",
    "band_weights",
    "make_component",
    "blackbody_flux_lambda",
    "resolve_bandpass",
    "resolve_bandpasses",
    "resolve_svo_id",
    "tophat_bandpass",
    "logg_main_sequence",
    "radius_main_sequence",
    "mass_main_sequence",
    "companion_logg",
    "ccm89_odonnell_axav",
    "redden_factor",
]


DEFAULT_CACHE_DIR = Path(
    os.environ.get("SED_MODEL_CACHE", Path.home() / "Documents" / "SED" / "SEDs_fit" / "model_cache")
)

# Common rebinning wavelength grid for all sources.
GRID_WAVE_MIN_AA = 2_000.0
GRID_WAVE_MAX_AA = 400_000.0
GRID_RESOLUTION = 2000.0  # lambda / dlambda; ample for broad-band photometry

_PRINT_LOCK = threading.Lock()


def _log(msg):
    with _PRINT_LOCK:
        print(msg, flush=True)


# --------------------------------------------------------------------------
# Extinction (CCM89 + O'Donnell 1994 optical coefficients)
# --------------------------------------------------------------------------

def ccm89_odonnell_axav(wave_AA, Rv=3.1):
    """A(lambda) / A(V) for the CCM89 law with O'Donnell (1994) optical coefficients."""
    lam_um = np.asarray(wave_AA, dtype=float) * 1e-4
    x = 1.0 / lam_um
    a = np.zeros_like(x)
    b = np.zeros_like(x)

    ir = x < 1.1
    a[ir] = 0.574 * x[ir] ** 1.61
    b[ir] = -0.527 * x[ir] ** 1.61

    opt = (x >= 1.1) & (x <= 3.3)
    y = x[opt] - 1.82
    a[opt] = (
        1.0
        + 0.17699 * y
        - 0.50447 * y**2
        - 0.02427 * y**3
        + 0.72085 * y**4
        + 0.01979 * y**5
        - 0.77530 * y**6
        + 0.32999 * y**7
    )
    b[opt] = (
        1.41338 * y
        + 2.28305 * y**2
        + 1.07233 * y**3
        - 5.38434 * y**4
        - 0.62251 * y**5
        + 5.30260 * y**6
        - 2.09002 * y**7
    )

    # Near-UV (3.3 < x <= 8): CCM89 extrapolation without the 2175 A bump term
    uv = x > 3.3
    if np.any(uv):
        xu = np.clip(x[uv], 3.3, 8.0)
        fa = np.where(xu >= 5.9, -0.04473 * (xu - 5.9) ** 2 - 0.009779 * (xu - 5.9) ** 3, 0.0)
        fb = np.where(xu >= 5.9, 0.2130 * (xu - 5.9) ** 2 + 0.1207 * (xu - 5.9) ** 3, 0.0)
        a[uv] = 1.752 - 0.316 * xu - 0.104 / ((xu - 4.67) ** 2 + 0.341) + fa
        b[uv] = -3.090 + 1.825 * xu + 1.206 / ((xu - 4.62) ** 2 + 0.263) + fb

    return a + b / Rv


def redden_factor(wave_AA, A_V, Rv=3.1):
    """Multiplicative attenuation factor 10**(-0.4 A_lambda)."""
    wave_AA = np.asarray(wave_AA, dtype=float)
    if A_V <= 0:
        return np.ones_like(wave_AA)
    return 10.0 ** (-0.4 * A_V * ccm89_odonnell_axav(wave_AA, Rv))


# --------------------------------------------------------------------------
# Empirical main-sequence relations
# --------------------------------------------------------------------------

# Empirical dwarf sequence from the Pecaut & Mamajek (2013 and revisions) EEM
# table, supplemented by Mann et al. (2015) for M types. Covers M9V (~2400 K)
# to G0V (~5930 K). Radius and mass come from the same table, so the radius and
# gravity derived below stay mutually consistent.
_MS_TEFF_K = np.array(
    [
        2400.0, 2500.0, 2650.0, 2850.0, 3030.0, 3200.0, 3410.0,
        3550.0, 3700.0, 3870.0, 4070.0, 4400.0, 4600.0, 4830.0,
        4960.0, 5080.0, 5280.0, 5490.0, 5660.0, 5770.0, 5930.0,
    ]
)
_MS_RADIUS_RSUN = np.array(
    [
        0.102, 0.114, 0.120, 0.137, 0.196, 0.274, 0.361,
        0.446, 0.501, 0.588, 0.628, 0.680, 0.706, 0.739,
        0.755, 0.770, 0.813, 0.855, 0.911, 0.975, 1.055,
    ]
)
_MS_MASS_MSUN = np.array(
    [
        0.079, 0.085, 0.090, 0.102, 0.162, 0.230, 0.360,
        0.440, 0.500, 0.570, 0.620, 0.680, 0.710, 0.750,
        0.780, 0.800, 0.850, 0.890, 0.940, 1.000, 1.060,
    ]
)

_G_CGS = 6.67430e-8
_M_SUN_G = M_SUN_G   # g, astropy.constants.M_sun (wdms_common)
_R_SUN_CM = 6.957e10


def radius_main_sequence(teff):
    """Main-sequence dwarf radius at the given Teff, in R_sun.

    Interpolation is done in log(R), which varies more smoothly than R along the
    sequence. Outside the table bounds the value is clamped to the extreme
    point: extrapolating a stellar radius would not be meaningful.
    """
    return np.exp(
        np.interp(np.asarray(teff, dtype=float), _MS_TEFF_K, np.log(_MS_RADIUS_RSUN))
    )


def mass_main_sequence(teff):
    """Main-sequence dwarf mass at the given Teff, in M_sun."""
    return np.exp(
        np.interp(np.asarray(teff, dtype=float), _MS_TEFF_K, np.log(_MS_MASS_MSUN))
    )


def logg_main_sequence(teff):
    """Surface gravity of a main-sequence dwarf.

    Derived from mass and radius from the same table rather than tabulated
    separately, to avoid mutually inconsistent (Teff, R, logg) triplets.
    Broad-band SED fitting is only weakly sensitive to logg anyway.
    """
    mass_g = mass_main_sequence(teff) * _M_SUN_G
    radius_cm = radius_main_sequence(teff) * _R_SUN_CM
    return np.log10(_G_CGS * mass_g / radius_cm**2)


def companion_logg(teff, radius_rsun, mode="ms_relation", fixed=5.0):
    """Companion (M dwarf) surface gravity [log10 cgs] used to read its grid.

    The single gravity prescription of the pipeline, used by the SED term
    (sed_fit._SedModel) and by every eclipse-depth calculation (depth_model):

      "ms_relation" : logg_main_sequence(teff), the empirical dwarf sequence
                      (radius_rsun is not used)
      "from_radius" : log10(G M / R^2) with M = mass_main_sequence(teff) and
                      R = radius_rsun
      "fixed"       : the constant `fixed`

    Not clipped: each caller clips to the logg range of its own grid.
    Accepts scalars or arrays.
    """
    mode = str(mode)
    teff = np.asarray(teff, dtype=float)
    if mode == "fixed":
        return np.full(teff.shape, float(fixed)) if teff.ndim else float(fixed)
    if mode == "from_radius":
        mass_g = mass_main_sequence(teff) * _M_SUN_G
        radius_cm = np.asarray(radius_rsun, dtype=float) * _R_SUN_CM
        return np.log10(_G_CGS * mass_g / radius_cm**2)
    return logg_main_sequence(teff)


# --------------------------------------------------------------------------
# Bandpasses
# --------------------------------------------------------------------------

# VizieR SEDs name filters "CATALOGUE:band"; the SVO Filter Profile Service
# uses "FACILITY/INSTRUMENT.band". Mapping table for filters commonly found in
# VizieR SEDs of stellar targets.
VIZIER_TO_SVO = {
    "2MASS:J": "2MASS/2MASS.J",
    "2MASS:H": "2MASS/2MASS.H",
    "2MASS:Ks": "2MASS/2MASS.Ks",
    "WISE:W1": "WISE/WISE.W1",
    "WISE:W2": "WISE/WISE.W2",
    "WISE:W3": "WISE/WISE.W3",
    "WISE:W4": "WISE/WISE.W4",
    "GAIA/GAIA3:G": "GAIA/GAIA3.G",
    "GAIA/GAIA3:Gbp": "GAIA/GAIA3.Gbp",
    "GAIA/GAIA3:Grp": "GAIA/GAIA3.Grp",
    "GAIA/GAIA2:G": "GAIA/GAIA2.G",
    "GAIA/GAIA2:Gbp": "GAIA/GAIA2.Gbp",
    "GAIA/GAIA2:Grp": "GAIA/GAIA2.Grp",
    "Gaia:G": "GAIA/GAIA2.G",
    "SDSS:u": "SLOAN/SDSS.u",
    "SDSS:g": "SLOAN/SDSS.g",
    "SDSS:r": "SLOAN/SDSS.r",
    "SDSS:i": "SLOAN/SDSS.i",
    "SDSS:z": "SLOAN/SDSS.z",
    "PAN-STARRS/PS1:g": "PAN-STARRS/PS1.g",
    "PAN-STARRS/PS1:r": "PAN-STARRS/PS1.r",
    "PAN-STARRS/PS1:i": "PAN-STARRS/PS1.i",
    "PAN-STARRS/PS1:z": "PAN-STARRS/PS1.z",
    "PAN-STARRS/PS1:y": "PAN-STARRS/PS1.y",
    "SkyMapper/SkyMapper:u": "SkyMapper/SkyMapper.u",
    "SkyMapper/SkyMapper:v": "SkyMapper/SkyMapper.v",
    "SkyMapper/SkyMapper:g": "SkyMapper/SkyMapper.g",
    "SkyMapper/SkyMapper:r": "SkyMapper/SkyMapper.r",
    "SkyMapper/SkyMapper:i": "SkyMapper/SkyMapper.i",
    "SkyMapper/SkyMapper:z": "SkyMapper/SkyMapper.z",
    "Johnson:U": "Generic/Johnson.U",
    "Johnson:B": "Generic/Johnson.B",
    "Johnson:V": "Generic/Johnson.V",
    "Johnson:R": "Generic/Johnson.R",
    "Johnson:I": "Generic/Johnson.I",
    "Johnson:J": "Generic/Johnson_UBVRIJHKL.J",
    "Johnson:H": "Generic/Johnson_UBVRIJHKL.H",
    "Johnson:K": "Generic/Johnson_UBVRIJHKL.K",
    "Cousins:R": "Generic/Cousins.R",
    "Cousins:I": "Generic/Cousins.I",
    "Cousins:V": "Generic/Johnson.V",
    "VISTA:Z": "Paranal/VISTA.Z",
    "VISTA:Y": "Paranal/VISTA.Y",
    "VISTA:J": "Paranal/VISTA.J",
    "VISTA:H": "Paranal/VISTA.H",
    "VISTA:Ks": "Paranal/VISTA.Ks",
    "UKIDSS:Y": "UKIRT/UKIDSS.Y",
    "UKIDSS:J": "UKIRT/UKIDSS.J",
    "UKIDSS:H": "UKIRT/UKIDSS.H",
    "UKIDSS:K": "UKIRT/UKIDSS.K",
    "GALEX:FUV": "GALEX/GALEX.FUV",
    "GALEX:NUV": "GALEX/GALEX.NUV",
    "Spitzer/IRAC:3.6": "Spitzer/IRAC.I1",
    "Spitzer/IRAC:4.5": "Spitzer/IRAC.I2",
    "Spitzer/IRAC:5.8": "Spitzer/IRAC.I3",
    "Spitzer/IRAC:8.0": "Spitzer/IRAC.I4",
}

# Photographic / plate filters: no reliable transmission, and their photometric
# calibration is poor for SED fitting anyway.
UNSUPPORTED_FILTER_PATTERNS = (
    re.compile(r"^POSS", re.I),
    re.compile(r"^DENIS", re.I),
    re.compile(r"^USNO", re.I),
)


@dataclass(frozen=True)
class Bandpass:
    """Bandpass: dimensionless transmission sampled in Angstrom."""

    name: str
    wave_AA: np.ndarray
    throughput: np.ndarray
    svo_id: str | None = None
    synthetic: bool = False

    @property
    def pivot_wavelength_AA(self):
        t, w = self.throughput, self.wave_AA
        return float(np.sqrt(np.trapezoid(t * w, w) / np.trapezoid(t / w, w)))


def resolve_svo_id(vizier_name):
    """SVO FPS identifier matching a VizieR filter name, or None."""
    name = str(vizier_name).strip()
    if name in VIZIER_TO_SVO:
        return VIZIER_TO_SVO[name]
    for pattern in UNSUPPORTED_FILTER_PATTERNS:
        if pattern.match(name):
            return None
    # Heuristic: "A/B:c" -> "A/B.c" and "A:c" -> "A/A.c"
    if ":" in name:
        head, band = name.split(":", 1)
        if "/" in head:
            return f"{head}.{band}"
        return f"{head}/{head}.{band}"
    return None


def _bandpass_cache_path(svo_id, cache_dir):
    safe = svo_id.replace("/", "__").replace(".", "_")
    return Path(cache_dir) / "filters" / f"{safe}.npz"


def load_svo_bandpass(svo_id, cache_dir=DEFAULT_CACHE_DIR):
    """Load a transmission curve from SVO FPS, with on-disk cache."""
    path = _bandpass_cache_path(svo_id, cache_dir)
    if path.exists():
        with np.load(path) as data:
            return Bandpass(
                name=svo_id,
                wave_AA=data["wave"],
                throughput=data["throughput"],
                svo_id=svo_id,
            )

    from astroquery.svo_fps import SvoFps

    table = SvoFps.get_transmission_data(svo_id)
    wave = np.asarray(table["Wavelength"], dtype=float)
    thru = np.asarray(table["Transmission"], dtype=float)
    order = np.argsort(wave)
    wave, thru = wave[order], thru[order]
    keep = np.isfinite(wave) & np.isfinite(thru) & (thru >= 0)
    wave, thru = wave[keep], thru[keep]
    if wave.size < 5 or np.max(thru) <= 0:
        raise ValueError(f"Unusable SVO transmission for {svo_id}")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, wave=wave, throughput=thru)
    return Bandpass(name=svo_id, wave_AA=wave, throughput=thru, svo_id=svo_id)


def tophat_bandpass(name, lambda_eff_AA, width_frac=0.02):
    """Narrow top-hat bandpass, used when no transmission curve is known."""
    lo = lambda_eff_AA * (1.0 - 0.5 * width_frac)
    hi = lambda_eff_AA * (1.0 + 0.5 * width_frac)
    wave = np.linspace(lo, hi, 21)
    return Bandpass(
        name=name,
        wave_AA=wave,
        throughput=np.ones_like(wave),
        svo_id=None,
        synthetic=True,
    )


def resolve_bandpasses(vizier_names, lambda_eff_AA, cache_dir=DEFAULT_CACHE_DIR, verbose=True):
    """Resolve a list of filters, reusing transmissions for duplicates."""
    lambda_eff_AA = np.asarray(lambda_eff_AA, dtype=float)
    resolved = {}
    out = []
    for name, lam in zip(vizier_names, lambda_eff_AA):
        key = str(name)
        if key not in resolved:
            resolved[key] = resolve_bandpass(key, lam, cache_dir=cache_dir, verbose=verbose)
        out.append(resolved[key])
    return out


def resolve_bandpass(vizier_name, lambda_eff_AA, cache_dir=DEFAULT_CACHE_DIR, verbose=True):
    """Real bandpass if the filter is identifiable, otherwise a narrow top-hat."""
    svo_id = resolve_svo_id(vizier_name)
    if svo_id is not None:
        try:
            band = load_svo_bandpass(svo_id, cache_dir=cache_dir)
            # Guard: a wrong match shows up as a pivot-wavelength mismatch.
            if 0.7 < band.pivot_wavelength_AA / float(lambda_eff_AA) < 1.4:
                return band
            if verbose:
                _log(
                    f"[filter] {vizier_name} -> {svo_id} rejected "
                    f"(pivot {band.pivot_wavelength_AA:.0f} A vs SED {lambda_eff_AA:.0f} A)"
                )
        except Exception as exc:
            if verbose:
                _log(f"[filter] {vizier_name} -> {svo_id} unavailable ({exc})")
    if verbose:
        _log(f"[filter] {vizier_name}: top-hat bandpass at {lambda_eff_AA:.0f} A")
    return tophat_bandpass(str(vizier_name), float(lambda_eff_AA))


# --------------------------------------------------------------------------
# Grid sources
# --------------------------------------------------------------------------

class _PhoenixStScI:
    """STScI PHOENIX (Allard) grid: one FITS file per Teff, all logg as columns."""

    key = "phoenix-stsci"
    label = "PHOENIX (Allard, STScI trds)"
    base_url = "https://ssb.stsci.edu/trds/grid/phoenix/phoenixm00"
    teff_all = np.arange(2000.0, 6600.0, 100.0)
    logg_all = np.arange(0.0, 6.0, 0.5)

    def nodes(self, teff_range, logg_range):
        teff = self.teff_all[
            (self.teff_all >= teff_range[0]) & (self.teff_all <= teff_range[1])
        ]
        logg = self.logg_all[
            (self.logg_all >= logg_range[0]) & (self.logg_all <= logg_range[1])
        ]
        return teff, logg

    def raw_path(self, teff, cache_dir):
        return Path(cache_dir) / "raw" / self.key / f"phoenixm00_{int(round(teff))}.fits"

    def fetch(self, teff, cache_dir):
        path = self.raw_path(teff, cache_dir)
        if path.exists() and path.stat().st_size > 0:
            return path
        import requests

        url = f"{self.base_url}/phoenixm00_{int(round(teff))}.fits"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        with requests.get(url, stream=True, timeout=180) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        tmp.replace(path)
        return path

    def read(self, teff, logg_values, cache_dir):
        """Return (wave_AA, flux[nlogg, nwave]) as surface flux in erg/s/cm2/A.

        The highest gravities disappear from the grid at high Teff (logg 5.5
        beyond 5000 K): fall back to the nearest available gravity so the cube
        stays rectangular. That corner of parameter space corresponds to
        (Teff, logg) pairs with no physical main-sequence counterpart and is
        not visited in practice.
        """
        from astropy.io import fits

        path = self.fetch(teff, cache_dir)
        with fits.open(path, memmap=False) as hdul:
            data = hdul[1].data
            available = {
                float(name[1:]) / 10.0: name
                for name in data.columns.names
                if re.fullmatch(r"g\d{2}", name)
            }
            wave = np.asarray(data["WAVELENGTH"], dtype=float)
            out = np.empty((len(logg_values), wave.size), dtype=float)
            keys = np.array(sorted(available))
            for i, g in enumerate(logg_values):
                nearest = float(keys[np.argmin(np.abs(keys - g))])
                out[i] = np.asarray(data[available[nearest]], dtype=float)
        order = np.argsort(wave)
        return wave[order], out[:, order]


class _BtSettlCifistSVO:
    """BT-Settl CIFIST via the SVO Theoretical Spectra Server (one file per Teff/logg)."""

    key = "bt-settl-cifist"
    label = "BT-Settl CIFIST (SVO)"
    ssap_url = "http://svo2.cab.inta-csic.es/theory/newov2/ssap.php"

    def __init__(self):
        self._index = None

    def _load_index(self, cache_dir):
        if self._index is not None:
            return self._index
        path = Path(cache_dir) / "raw" / self.key / "index.npz"
        if path.exists():
            with np.load(path) as d:
                self._index = (d["teff"], d["logg"], d["fid"])
            return self._index

        import requests
        from astropy.io.votable import parse

        resp = requests.get(f"{self.ssap_url}?model={self.key}", timeout=180)
        resp.raise_for_status()
        table = parse(io.BytesIO(resp.content)).get_first_table().to_table()
        teff = np.asarray(table["teff"], dtype=float)
        logg = np.asarray(table["logg"], dtype=float)
        fid = np.array(
            [int(re.search(r"fid=(\d+)", str(r)).group(1)) for r in table["Access.Reference"]]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, teff=teff, logg=logg, fid=fid)
        self._index = (teff, logg, fid)
        return self._index

    def nodes(self, teff_range, logg_range, cache_dir=DEFAULT_CACHE_DIR):
        teff, logg, _ = self._load_index(cache_dir)
        sel = (
            (teff >= teff_range[0])
            & (teff <= teff_range[1])
            & (logg >= logg_range[0])
            & (logg <= logg_range[1])
        )
        return np.unique(teff[sel]), np.unique(logg[sel])

    def _fid(self, teff, logg, cache_dir):
        t, g, f = self._load_index(cache_dir)
        sel = (np.abs(t - teff) < 1e-6) & (np.abs(g - logg) < 1e-6)
        if not np.any(sel):
            return None
        return int(f[sel][0])

    def raw_path(self, teff, logg, cache_dir):
        return (
            Path(cache_dir)
            / "raw"
            / self.key
            / f"btsettl_{int(round(teff))}_{int(round(logg * 10)):02d}.npz"
        )

    def read(self, teff, logg_values, cache_dir):
        import requests
        from astropy.io.votable import parse

        spectra = []
        wave_ref = None
        for g in logg_values:
            path = self.raw_path(teff, g, cache_dir)
            if path.exists():
                with np.load(path) as d:
                    w, f = d["wave"], d["flux"]
            else:
                fid = self._fid(teff, g, cache_dir)
                if fid is None:
                    raise KeyError(f"BT-Settl: missing node Teff={teff}, logg={g}")
                resp = requests.get(f"{self.ssap_url}?model={self.key}&fid={fid}", timeout=600)
                resp.raise_for_status()
                tbl = parse(io.BytesIO(resp.content)).get_first_table().to_table()
                w = np.asarray(tbl["WAVELENGTH"], dtype=float)
                f = np.asarray(tbl["FLUX"], dtype=float)
                order = np.argsort(w)
                w, f = w[order], f[order]
                keep = (w >= 0.5 * GRID_WAVE_MIN_AA) & (w <= 2.0 * GRID_WAVE_MAX_AA)
                w, f = w[keep], f[keep]
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, wave=w, flux=f)
            if wave_ref is None:
                wave_ref = w
            spectra.append(np.interp(wave_ref, w, f))
        return wave_ref, np.array(spectra)


# Same location as main_all.KOESTER_DIR: <repository>/models/koester2, or
# $WDMS_MODELS_DIR/koester2 when that environment variable is set.
KOESTER_DIR_DEFAULT = Path(
    os.environ.get("WDMS_MODELS_DIR", Path(__file__).resolve().parent / "models")
) / "koester2"


class _KoesterDA:
    """Koester DA white-dwarf atmospheres, read from a local model directory.

    Files are named ``da{Teff:05d}_{100*logg:03d}.dk.dat.txt`` and already hold
    the surface flux F_lambda in erg/s/cm2/A, so they feed the same band
    integration as the cool-companion grids with no rescaling.
    """

    key = "koester-da"
    label = "Koester DA (local grid)"
    directory = KOESTER_DIR_DEFAULT

    _NAME_RE = re.compile(r"da(\d{5})_(\d{3})\.dk\.dat\.txt$")

    def __init__(self, directory=None):
        # Read WDMS_MODELS_DIR when the grid is built, not only at import, so a
        # folder chosen at start-up (main_all.configure_paths) is honoured.
        if directory is None:
            env = os.environ.get("WDMS_MODELS_DIR")
            directory = Path(env) / "koester2" if env else self.directory
        self.directory = Path(directory)
        self._catalog = None

    def _load_catalog(self):
        if self._catalog is None:
            entries = {}
            for path in sorted(self.directory.glob("da*.dk.dat.txt")):
                match = self._NAME_RE.search(path.name)
                if match:
                    entries[(float(match.group(1)), float(match.group(2)) / 100.0)] = path
            if not entries:
                raise FileNotFoundError(f"No Koester DA model in {self.directory}")
            self._catalog = entries
        return self._catalog

    def nodes(self, teff_range, logg_range, cache_dir=DEFAULT_CACHE_DIR):
        catalog = self._load_catalog()
        teff = np.unique([t for t, _ in catalog])
        logg = np.unique([g for _, g in catalog])
        teff = teff[(teff >= teff_range[0]) & (teff <= teff_range[1])]
        logg = logg[(logg >= logg_range[0]) & (logg <= logg_range[1])]
        return teff, logg

    @staticmethod
    def _read_one(path):
        table = np.loadtxt(path, comments="#", usecols=(0, 1))
        wave, flux = table[:, 0], table[:, 1]
        order = np.argsort(wave)
        wave, flux = wave[order], flux[order]
        unique = np.concatenate(([True], np.diff(wave) > 0))
        wave, flux = wave[unique], flux[unique]
        # The Koester grid stops near 3 um whereas the band grid runs to 40 um.
        # A white dwarf is deep in the Rayleigh-Jeans regime there, where
        # F_lambda * lambda^4 is constant, so extend the tail instead of
        # letting the rebinning read the far infrared as zero flux.
        if wave[-1] < GRID_WAVE_MAX_AA:
            tail = np.geomspace(wave[-1] * 1.001, GRID_WAVE_MAX_AA * 1.05, 200)
            flux = np.concatenate([flux, flux[-1] * (wave[-1] / tail) ** 4])
            wave = np.concatenate([wave, tail])
        return wave, flux

    def read(self, teff, logg_values, cache_dir):
        catalog = self._load_catalog()
        available = np.unique([g for t, g in catalog if abs(t - teff) < 1e-6])
        if available.size == 0:
            raise KeyError(f"Koester DA: no model at Teff = {teff} K")
        wave_ref = None
        spectra = []
        for g in logg_values:
            nearest = float(available[np.argmin(np.abs(available - g))])
            wave, flux = self._read_one(catalog[(float(teff), nearest)])
            if wave_ref is None:
                wave_ref = wave
                spectra.append(flux)
            else:
                spectra.append(np.interp(wave_ref, wave, flux))
        return wave_ref, np.array(spectra)


_SOURCES = {
    _PhoenixStScI.key: _PhoenixStScI,
    _BtSettlCifistSVO.key: _BtSettlCifistSVO,
    _KoesterDA.key: _KoesterDA,
}


# --------------------------------------------------------------------------
# Blackbody
# --------------------------------------------------------------------------

def blackbody_flux_lambda(wave_AA, teff):
    """Blackbody surface flux pi*B_lambda, in erg/s/cm2/A."""
    from astropy.constants import c, h, k_B

    wave_cm = np.asarray(wave_AA, dtype=float) * 1e-8
    x = (h.cgs.value * c.cgs.value) / (wave_cm * k_B.cgs.value * float(teff))
    x = np.clip(x, 1e-8, 700.0)
    prefactor = 2.0 * h.cgs.value * c.cgs.value**2 / wave_cm**5
    return np.pi * prefactor / np.expm1(x) * 1e-8


# --------------------------------------------------------------------------
# Flux-conserving rebinning
# --------------------------------------------------------------------------

def _rebin_flux_conserving(wave, flux, edges):
    """Mean F_lambda over each interval defined by `edges`.

    Conserves the flux integral, unlike naive resampling: essential here because
    the native spectra resolve deep lines that the output grid does not sample.
    """
    good = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    wave, flux = wave[good], flux[good]
    if wave.size < 2:
        return np.zeros(edges.size - 1)
    cumulative = np.concatenate(
        ([0.0], np.cumsum(0.5 * (flux[1:] + flux[:-1]) * np.diff(wave)))
    )
    integral_at_edges = np.interp(edges, wave, cumulative)
    return np.diff(integral_at_edges) / np.diff(edges)


def _log_wave_grid(wave_min=GRID_WAVE_MIN_AA, wave_max=GRID_WAVE_MAX_AA, resolution=GRID_RESOLUTION):
    n = int(np.ceil(np.log(wave_max / wave_min) * resolution))
    edges = np.geomspace(wave_min, wave_max, n + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return centers, edges


def band_weights(band, wave):
    """Weights w such that <F_lambda> = w . F[idx] on the model grid `wave`.

    Photon weighting, i.e. the integrand is F(lambda) T(lambda) lambda, which
    matches photon-counting detectors and the `sed_flux` convention of VizieR.
    Shared by every photometry class so that a blackbody and an atmosphere grid
    are integrated through strictly the same filter weights.
    """
    lo, hi = band.wave_AA[0], band.wave_AA[-1]
    idx = np.where((wave >= lo) & (wave <= hi))[0]
    if idx.size < 3:
        # Band narrower than the grid spacing: fall back to the nearest point.
        idx = np.array([int(np.argmin(np.abs(wave - band.pivot_wavelength_AA)))])
        return idx, np.array([1.0])
    w = wave[idx]
    t = np.interp(w, band.wave_AA, band.throughput, left=0.0, right=0.0)
    dw = np.empty_like(w)
    dw[1:-1] = 0.5 * (w[2:] - w[:-2])
    dw[0] = 0.5 * (w[1] - w[0])
    dw[-1] = 0.5 * (w[-1] - w[-2])
    numerator = t * w * dw
    denominator = float(np.sum(t * w * dw))
    if denominator <= 0:
        idx = np.array([int(np.argmin(np.abs(wave - band.pivot_wavelength_AA)))])
        return idx, np.array([1.0])
    return idx, numerator / denominator


# --------------------------------------------------------------------------
# Rebinned and cached spectral grid
# --------------------------------------------------------------------------

@dataclass
class SpectralGrid:
    """Surface-flux cube F_lambda[Teff, logg, lambda] in erg/s/cm2/A."""

    source_key: str
    teff: np.ndarray
    logg: np.ndarray
    wave_AA: np.ndarray
    flux: np.ndarray  # (n_teff, n_logg, n_wave)

    @staticmethod
    def cache_path(source_key, teff_range, logg_range, resolution, cache_dir):
        name = (
            f"{source_key}_T{int(teff_range[0])}-{int(teff_range[1])}"
            f"_g{logg_range[0]:.1f}-{logg_range[1]:.1f}_R{int(resolution)}.npz"
        )
        return Path(cache_dir) / "grids" / name

    @classmethod
    def load(
        cls,
        source_key="phoenix-stsci",
        teff_range=(2300.0, 5900.0),
        logg_range=(4.0, 5.5),
        resolution=GRID_RESOLUTION,
        cache_dir=DEFAULT_CACHE_DIR,
        n_workers=4,
        verbose=True,
    ):
        """Load the grid from cache, building it if needed."""
        path = cls.cache_path(source_key, teff_range, logg_range, resolution, cache_dir)
        if path.exists():
            with np.load(path) as d:
                if verbose:
                    _log(f"[grid] cache: {path.name}")
                return cls(
                    source_key=source_key,
                    teff=d["teff"],
                    logg=d["logg"],
                    wave_AA=d["wave"],
                    flux=d["flux"],
                )

        if source_key not in _SOURCES:
            raise KeyError(f"Unknown source: {source_key} (expected: {list(_SOURCES)})")
        source = _SOURCES[source_key]()

        try:
            teff_nodes, logg_nodes = source.nodes(teff_range, logg_range, cache_dir=cache_dir)
        except TypeError:
            teff_nodes, logg_nodes = source.nodes(teff_range, logg_range)
        if teff_nodes.size == 0 or logg_nodes.size == 0:
            raise ValueError("No grid node in the requested interval")

        centers, edges = _log_wave_grid(resolution=resolution)
        cube = np.zeros((teff_nodes.size, logg_nodes.size, centers.size), dtype=float)

        if verbose:
            _log(
                f"[grid] building {source.label}: "
                f"{teff_nodes.size} Teff x {logg_nodes.size} logg, "
                f"{centers.size} spectral points"
            )

        def build_one(i_teff):
            teff_value = teff_nodes[i_teff]
            wave, spectra = source.read(teff_value, logg_nodes, cache_dir)
            out = np.empty((logg_nodes.size, centers.size))
            for j in range(logg_nodes.size):
                out[j] = _rebin_flux_conserving(wave, spectra[j], edges)
            if verbose:
                _log(f"[grid]   Teff = {teff_value:.0f} K done")
            return i_teff, out

        with ThreadPoolExecutor(max_workers=max(1, int(n_workers))) as pool:
            for i_teff, block in pool.map(build_one, range(teff_nodes.size)):
                cube[i_teff] = block

        cube = np.clip(cube, 0.0, None)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, teff=teff_nodes, logg=logg_nodes, wave=centers, flux=cube.astype(np.float32)
        )
        if verbose:
            _log(f"[grid] saved: {path}")
        return cls(
            source_key=source_key,
            teff=teff_nodes,
            logg=logg_nodes,
            wave_AA=centers,
            flux=cube,
        )

    @classmethod
    def blackbody(cls, teff_min=1500.0, teff_max=150_000.0, n_teff=250, resolution=GRID_RESOLUTION):
        """Blackbody grid in the same format as the atmosphere grids.

        Lets a blackbody component (e.g. a white dwarf) go through exactly the
        same band-integration pipeline.
        """
        centers, _ = _log_wave_grid(resolution=resolution)
        teff_nodes = np.geomspace(teff_min, teff_max, int(n_teff))
        cube = np.empty((teff_nodes.size, 2, centers.size), dtype=np.float32)
        for i, t in enumerate(teff_nodes):
            spectrum = blackbody_flux_lambda(centers, t)
            cube[i, 0] = spectrum
            cube[i, 1] = spectrum
        return cls(
            source_key="blackbody",
            teff=teff_nodes,
            logg=np.array([0.0, 1.0]),
            wave_AA=centers,
            flux=cube,
        )

    @property
    def teff_bounds(self):
        return float(self.teff[0]), float(self.teff[-1])

    @property
    def logg_bounds(self):
        return float(self.logg[0]), float(self.logg[-1])

    def spectrum(self, teff, logg, a_v=0.0, wave_out=None, R_V=3.1):
        """Interpolated, reddened, optionally resampled spectrum (for plotting)."""
        flux = _bilinear(self.flux, self.teff, self.logg, float(teff), float(logg))
        if a_v > 0:
            flux = flux * redden_factor(self.wave_AA, float(a_v), R_V)
        if wave_out is None:
            return self.wave_AA, flux
        out = np.exp(
            np.interp(
                np.log(np.asarray(wave_out, dtype=float)),
                np.log(self.wave_AA),
                np.log(np.maximum(flux, 1e-300)),
            )
        )
        return np.asarray(wave_out, dtype=float), out


def _bilinear(cube, x_nodes, y_nodes, x, y):
    """Bilinear interpolation of log(flux) in (Teff, logg)."""
    x = min(max(x, x_nodes[0]), x_nodes[-1])
    y = min(max(y, y_nodes[0]), y_nodes[-1])
    i = min(int(np.searchsorted(x_nodes, x, side="right") - 1), x_nodes.size - 2)
    j = min(int(np.searchsorted(y_nodes, y, side="right") - 1), y_nodes.size - 2)
    i = max(i, 0)
    j = max(j, 0)
    tx = (x - x_nodes[i]) / (x_nodes[i + 1] - x_nodes[i])
    ty = (y - y_nodes[j]) / (y_nodes[j + 1] - y_nodes[j])
    block = np.log(np.maximum(cube[i : i + 2, j : j + 2], 1e-300))
    interp = (
        (1 - tx) * (1 - ty) * block[0, 0]
        + tx * (1 - ty) * block[1, 0]
        + (1 - tx) * ty * block[0, 1]
        + tx * ty * block[1, 1]
    )
    return np.exp(interp)


# --------------------------------------------------------------------------
# Precomputed synthetic photometry
# --------------------------------------------------------------------------

class BandPhotometry:
    """Table (Teff, logg, A_V) -> mean band flux density.

    Photometry is computed with photon weighting,

        <F_lambda> = int F_lambda T(lambda) lambda dlambda / int T(lambda) lambda dlambda

    which matches the convention of photon-counting detectors and therefore the
    `sed_flux` values published by VizieR.

    Reddening is applied **before** band integration, on the full spectral grid:
    essential for broad blue bands, where A(lambda) varies strongly inside the
    filter.
    """

    def __init__(self, grid, bandpasses, av_nodes=None, R_V=3.1, verbose=True):
        self.grid = grid
        self.bandpasses = list(bandpasses)
        self.R_V = float(R_V)
        if av_nodes is None:
            av_nodes = np.linspace(0.0, 1.0, 11)
        self.av_nodes = np.asarray(av_nodes, dtype=float)
        if self.av_nodes.size < 2:
            self.av_nodes = np.array([float(self.av_nodes[0]), float(self.av_nodes[0]) + 1e-3])

        wave = grid.wave_AA
        self._weights = [self._band_weights(band, wave) for band in self.bandpasses]
        self._extinction_curve = ccm89_odonnell_axav(wave, self.R_V)

        n_t, n_g = grid.teff.size, grid.logg.size
        n_a, n_b = self.av_nodes.size, len(self.bandpasses)
        table = np.empty((n_t, n_g, n_a, n_b), dtype=float)

        if verbose:
            _log(f"[photometry] computing table {n_t}x{n_g}x{n_a}x{n_b}")

        for i_a, a_v in enumerate(self.av_nodes):
            attenuation = 10.0 ** (-0.4 * a_v * self._extinction_curve)
            reddened = grid.flux * attenuation  # broadcast over (Teff, logg)
            for i_b, (idx, w) in enumerate(self._weights):
                table[:, :, i_a, i_b] = reddened[:, :, idx] @ w

        self.table = np.log(np.maximum(table, 1e-300))
        if verbose:
            _log("[photometry] table ready")

    _band_weights = staticmethod(band_weights)

    def __call__(self, teff, logg, a_v):
        """Surface flux densities (erg/s/cm2/A) in each band."""
        return np.exp(_trilinear(self.table, self.grid.teff, self.grid.logg, self.av_nodes,
                                 float(teff), float(logg), float(a_v)))


class BlackbodyBandPhotometry:
    """Band-integrated blackbody fluxes, evaluated exactly at the requested Teff.

    Uses the filter weights and the pre-integration reddening of
    `BandPhotometry`, but calls the Planck function directly instead of
    interpolating a precomputed cube. A blackbody component is then free of the
    grid-interpolation error that would otherwise be mistaken for a genuine
    model systematic when a blackbody run is compared against an atmosphere run.
    """

    def __init__(self, bandpasses, R_V=3.1, resolution=GRID_RESOLUTION, verbose=False):
        self.bandpasses = list(bandpasses)
        self.R_V = float(R_V)
        wave, _ = _log_wave_grid(resolution=resolution)
        self.wave_AA = wave

        weights = [band_weights(band, wave) for band in self.bandpasses]
        # Only the wavelengths actually covered by a filter are evaluated: the
        # Planck call is the cost driver inside the likelihood.
        used = np.unique(np.concatenate([idx for idx, _ in weights]))
        position = {int(k): n for n, k in enumerate(used)}
        matrix = np.zeros((len(self.bandpasses), used.size), dtype=float)
        for i, (idx, w) in enumerate(weights):
            matrix[i, [position[int(k)] for k in idx]] = w

        self._matrix = matrix
        self._wave_used = wave[used]
        self._extinction_used = ccm89_odonnell_axav(self._wave_used, self.R_V)
        if verbose:
            _log(
                f"[photometry] exact blackbody over {len(self.bandpasses)} bands "
                f"({used.size} wavelengths)"
            )

    def __call__(self, teff, logg=0.0, a_v=0.0):
        """Surface flux densities (erg/s/cm2/A) in each band."""
        spectrum = blackbody_flux_lambda(self._wave_used, float(teff))
        if a_v > 0:
            spectrum = spectrum * 10.0 ** (-0.4 * float(a_v) * self._extinction_used)
        return self._matrix @ spectrum


# --------------------------------------------------------------------------
# Ready-to-use stellar component for fitting
# --------------------------------------------------------------------------

class Component:
    """One component of the SED model: per-band surface flux and spectrum.

    `band_flux` returns **surface** flux densities: the calling code applies the
    (R/d)**2 factor itself, as with the blackbody model.
    """

    def __init__(self, photometry, kind, uses_logg):
        self.photometry = photometry
        self.kind = kind
        self.uses_logg = bool(uses_logg)

    @property
    def grid(self):
        return self.photometry.grid

    @property
    def teff_bounds(self):
        return self.grid.teff_bounds

    @property
    def logg_bounds(self):
        return self.grid.logg_bounds if self.uses_logg else (0.0, 0.0)

    def band_flux(self, teff, logg=0.0, a_v=0.0):
        return self.photometry(teff, logg if self.uses_logg else 0.0, a_v)

    def spectrum(self, teff, logg=0.0, a_v=0.0, wave_out=None):
        return self.grid.spectrum(
            teff, logg if self.uses_logg else 0.0, a_v, wave_out=wave_out, R_V=self.photometry.R_V
        )

    def __repr__(self):
        t0, t1 = self.teff_bounds
        return f"<Component {self.kind} Teff {t0:.0f}-{t1:.0f} K, {len(self.photometry.bandpasses)} bands>"


class BlackbodyComponent:
    """Blackbody component sharing the Component interface, without a grid."""

    uses_logg = False

    def __init__(self, bandpasses, R_V=3.1, teff_range=(1500.0, 150_000.0),
                 resolution=GRID_RESOLUTION, verbose=False):
        self.photometry = BlackbodyBandPhotometry(
            bandpasses, R_V=R_V, resolution=resolution, verbose=verbose
        )
        self.kind = "blackbody-exact"
        self._teff_bounds = (float(teff_range[0]), float(teff_range[1]))

    @property
    def teff_bounds(self):
        return self._teff_bounds

    @property
    def logg_bounds(self):
        return (0.0, 0.0)

    def band_flux(self, teff, logg=0.0, a_v=0.0):
        return self.photometry(teff, 0.0, a_v)

    def spectrum(self, teff, logg=0.0, a_v=0.0, wave_out=None):
        wave = self.photometry.wave_AA if wave_out is None else np.asarray(wave_out, dtype=float)
        flux = blackbody_flux_lambda(wave, float(teff))
        if a_v > 0:
            flux = flux * redden_factor(wave, float(a_v), self.photometry.R_V)
        return wave, flux

    def __repr__(self):
        t0, t1 = self.teff_bounds
        return (f"<Component blackbody-exact Teff {t0:.0f}-{t1:.0f} K, "
                f"{len(self.photometry.bandpasses)} bands>")


def make_component(
    kind,
    bandpasses,
    av_nodes=None,
    R_V=3.1,
    teff_range=(2300.0, 5900.0),
    logg_range=(4.0, 5.5),
    cache_dir=DEFAULT_CACHE_DIR,
    blackbody_teff_range=(1500.0, 150_000.0),
    verbose=True,
):
    """Build a component from a model keyword.

    kind is "blackbody", "blackbody-exact", "phoenix-stsci", "bt-settl-cifist"
    or "koester-da". "blackbody-exact" skips the Teff cube and evaluates the
    Planck function directly; prefer it whenever a blackbody run is meant to be
    differenced against an atmosphere run.
    """
    if kind == "blackbody-exact":
        return BlackbodyComponent(
            bandpasses, R_V=R_V, teff_range=blackbody_teff_range, verbose=verbose
        )
    if kind == "blackbody":
        grid = SpectralGrid.blackbody(
            teff_min=blackbody_teff_range[0], teff_max=blackbody_teff_range[1]
        )
        uses_logg = False
    elif kind in _SOURCES:
        grid = SpectralGrid.load(
            source_key=kind,
            teff_range=teff_range,
            logg_range=logg_range,
            cache_dir=cache_dir,
            verbose=verbose,
        )
        uses_logg = True
    else:
        raise KeyError(
            f"Unknown atmosphere model: {kind} "
            f"(expected: 'blackbody', 'blackbody-exact', {list(_SOURCES)})"
        )
    photometry = BandPhotometry(grid, bandpasses, av_nodes=av_nodes, R_V=R_V, verbose=verbose)
    return Component(photometry, kind=kind, uses_logg=uses_logg)


def _trilinear(table, x_nodes, y_nodes, z_nodes, x, y, z):
    """Trilinear interpolation in (Teff, logg, A_V); returns a band vector."""
    x = min(max(x, x_nodes[0]), x_nodes[-1])
    y = min(max(y, y_nodes[0]), y_nodes[-1])
    z = min(max(z, z_nodes[0]), z_nodes[-1])

    i = max(0, min(int(np.searchsorted(x_nodes, x, side="right") - 1), x_nodes.size - 2))
    j = max(0, min(int(np.searchsorted(y_nodes, y, side="right") - 1), y_nodes.size - 2))
    k = max(0, min(int(np.searchsorted(z_nodes, z, side="right") - 1), z_nodes.size - 2))

    tx = (x - x_nodes[i]) / (x_nodes[i + 1] - x_nodes[i])
    ty = (y - y_nodes[j]) / (y_nodes[j + 1] - y_nodes[j])
    tz = (z - z_nodes[k]) / (z_nodes[k + 1] - z_nodes[k])

    block = table[i : i + 2, j : j + 2, k : k + 2]
    c00 = block[0, 0, 0] * (1 - tz) + block[0, 0, 1] * tz
    c01 = block[0, 1, 0] * (1 - tz) + block[0, 1, 1] * tz
    c10 = block[1, 0, 0] * (1 - tz) + block[1, 0, 1] * tz
    c11 = block[1, 1, 0] * (1 - tz) + block[1, 1, 1] * tz
    c0 = c00 * (1 - ty) + c01 * ty
    c1 = c10 * (1 - ty) + c11 * ty
    return c0 * (1 - tx) + c1 * tx
