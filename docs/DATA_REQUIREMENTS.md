# External data: models, tables, filters, downloads and caches

This page lists every file the pipeline reads that is **not** part of this repository, and every download the code performs on its own. Model and table installation instructions, including where to obtain each file, will be added in a later stage. Until then, these files must be copied from an existing working installation.

The sizes and node ranges below were measured on the author's local copy. They describe that copy, not a minimum requirement.

## Model files: `<WORKSPACE>/models/` (downloaded) and `2_sed_dilution/tables/` (included)

The default location is `<WORKSPACE>/models/`, with `<WORKSPACE>` = `<DOWNLOADS>/WDMS_fitting_pipeline/` (see the README). Set the environment variable `WDMS_MODELS_DIR`, or `MODELS_DIR` in `main_all.py`, to use another folder; `main_all.py` passes the resolved folder to `sed_atmosphere_models.py` for the Koester grid.

| Path under `models/` (Koester) or `tables/` (others) | Format expected by the code | Read by | Needed when | Local copy |
|---|---|---|---|---|
| `koester2/da{Teff:05d}_{100·logg:03d}.dk.dat.txt` | 2 columns: wavelength [Å], surface F_λ [erg s⁻¹ cm⁻² Å⁻¹]; `#` comments | `sed_atmosphere_models._KoesterDA`, `depth_model.DepthFluxModel` | SED fit with `SED_WD_ATMOSPHERE = "koester"`; dilution, joint and eclipse-radius stages | 1 066 files, 140 MB, Teff 5 000–80 000 K, log g 6.50–9.50 |
| `bedard2020/Table_DA.csv` | CSV with columns `Teff`, `RadiusRsun`, `MassMsun`, optional `CoolingAgeGyr` and `hypothesis` (rows with `thick` are kept) | `mass.load_bedard_grid` (also through `sed_fit._WDCoolingLogg`, the white-dwarf gravity of every stage) | SED fit with Koester and `cooling_track`; dilution; joint; mass stage | 1.1 MB |
| `bedard2020/Table_DB.csv` | same format | `sed_fit._WDCoolingLogg` | only when `WD_ATMO_TYPE = "DB"` | 1.1 MB |
| `BHAC2015/BHAC15_tracks.csv` | CSV with columns `M/Ms`, `log t(yr)`, `Teff`, `g`, `R/Rs` | `dilution_fit._load_bhac15`, `mass.bhac15_isochrone` | BHAC15 dilution; ideal joint fit (always: log g_MS and T_MS coverage); mass stage; `brown2022` prior | 0.4 MB |
| `laplata2013/DAwhitedwarfsZ0.02.dat` | whitespace table: log g, log Teff, M, cooling age, R, core (`He`/`CO`/`ONe`), …; `#` comments | `mass.load_laplata_grid` | mass stage (He-core grid) | 0.6 MB |

**Koester DA source.** `0_setup/setup_pipeline.ipynb` downloads them. The 1 066 Koester DA spectra are the model `koester2` of the SVO Theoretical Spectra Server (`http://svo2.cab.inta-csic.es/theory/newov2/ssap.php?model=koester2`): 82 Teff × 13 log g, the same grid as the local copy. Three spectra downloaded from it matched the local files to 6×10⁻⁸ in flux and 3×10⁻⁸ in SPECULOOS band flux.

The Bédard, BHAC15 and La Plata tables are **included in the repository** (`2_sed_dilution/tables/`, see its README); only the Koester spectra must be downloaded. The local PHOENIX-ACES SpecInt files (`models/phoenix2021/`, ≈ 4.3 GB) are **no longer read**. The eclipse-depth term now uses the same STScI PHOENIX grid as the SED term (see below), so the remaining local models total about 0.15 GB. All model grids are excluded from Git by `.gitignore`.

## Files included in this repository

| Path | Content | Used for |
|---|---|---|
| `2_sed_dilution/filters/SPECULOOS_iz.dat` | 2 columns: wavelength [Å], transmission | eclipse-depth filters `iz`, `i` (cut at 8 150 Å) and `z` (cut from 8 050 Å) |
| `2_sed_dilution/filters/SPECULOOS_clear.dat` | same | eclipse-depth filter `exo` (cut from 4 750 Å) |

The mapping and the cuts are set in `FILTER_PROFILE_FILES` and `FILTER_WAVELENGTH_CUTS` in `main_all.py`. A depth given for a filter key that is not in `FILTER_PROFILE_FILES` is reported by the input check.

## Optional local inputs

| Path | Content | Effect if missing |
|---|---|---|
| `<WORKSPACE>/TIC<id>/speculoos_light_curve/TIC<id>_speculoos_light_curve.pdf` | SPECULOOS light-curve figure saved by the notebook; used as the composite's light-curve panel (override: `speculoos_figure` key) | the composite figure (step 6) is skipped with a warning naming the expected path; every other step runs |
| `<WORKSPACE>/TIC<id>/speculoos_csv/TIC<id>_<band>.csv` | SPECULOOS observations (notebook convention; other names via the `speculoos_csv_files` key) | listed at start-up; the fits do not read them (depths come from `depth_by_filter`) |

## Downloads performed automatically by the code

None of these needs a manual step today. They require network access, and each one is cached as described.

| What | Source in the code | Cached in | When |
|---|---|---|---|
| Gaia DR3 ra, dec, parallax | ESA Gaia TAP (`sed_fit.GAIA_TAP_ENDPOINTS`) | not cached | every SED and ideal-joint call |
| VizieR SED photometry | `https://vizier.cds.unistra.fr/viz-bin/sed` | `<WORKSPACE>/TIC<id>/SED-dilution analysis/downloads/<gaia_id>/*.vot`, reused if present | first run of each target per output folder |
| Missing photometric σ | VizieR catalogues in `sed_fit.SIGMA_LOOKUP_CATALOGS` | not cached | when VizieR points lack an uncertainty |
| TIC ID | MAST TIC catalogue | not cached | only if `tic_id = None` |
| STScI PHOENIX grid (SED term and eclipse-depth term) | `https://ssb.stsci.edu/trds/grid/phoenix/phoenixm00` (one FITS per Teff, ≈ 12 MB) | `<cache>/raw/phoenix-stsci/`, rebinned grid in `<cache>/grids/` | when a PHOENIX companion is requested and no matching grid is cached; the depth term reads the same cached file |
| BT-Settl CIFIST spectra | SVO theory server | `<cache>/raw/bt-settl-cifist/`, `<cache>/grids/` | only if `btsettl` is requested (≈ 66 MB per node; several GB) |
| Broadband filter curves | SVO Filter Profile Service (`astroquery.svo_fps`) | `<cache>/filters/*.npz` | first use of each VizieR filter in a band-integrated SED model |

`<cache>` is `SED_ATMO_CACHE_DIR` in `main_all.py`. When that is `None`, the cache is the environment variable `SED_MODEL_CACHE` if set, otherwise `<WORKSPACE>/model_cache/`.

A rebinned grid is reused only when its file name matches exactly. The name encodes the source, the Teff range, the log g range and the resolution, so changing `SED_ATMO_GRID_RANGES` builds (and possibly downloads) a new grid. The Koester rebinned grid is also stored in `<cache>/grids/`. Its name does not record which Koester directory it was built from.

## Network notes

- `sed_fit.py` sets `SSL_CERT_FILE` to the `certifi` bundle at import time.
- The VizieR SED download is called with `allow_insecure=True`.
- Neither behaviour has been changed.
