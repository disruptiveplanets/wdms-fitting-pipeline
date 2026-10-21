#!/usr/bin/env python3
"""main_all.py — Entry point of the WD+MS fitting pipeline.

For each target this script runs, in order:

    1. SED fit            (sed_fit.py)          always
    2. Dilution fit       (dilution_fit.py, or dilution_uniform_fit.py when
                           MS_RADIUS_PRIOR = "uniform")          unless SED_ONLY
    3. Joint fit          (joint_fit_ideal.py, or the legacy joint_fit.py when
                           JOINT_FIT_METHOD = "legacy")          unless SED_ONLY
    4. Eclipse-radius diagnostic (dilution_fit.py)  only if
                           RUN_ECLIPSE_RADIUS_DIAGNOSTIC = True
    5. Corner plot        (display.py)                           unless SED_ONLY
    6. Composite figure   (display.py)  skipped with a warning when the
                           SPECULOOS light-curve figure is missing
    7. Masses and orbit   (mass.py)     only if RUN_MASS_ANALYSIS = True

Usage (from any folder; the repository can be stored anywhere)
-----
    cd <repository>/2_sed_dilution          # or give the full path to main_all.py
    python main_all.py --check            # check inputs only, run nothing
    python main_all.py --example          # single example target (TIC 450781262)
    python main_all.py                    # every active entry of TARGETS
    python main_all.py --mode final       # override MODE for this run
    python main_all.py --downloads-dir "/path/to/my/downloads"   # data/results root
    python main_all.py --example --fresh  # delete the target's old results first

Where things are read and written (wdms_paths.py), with
<WORKSPACE> = <DOWNLOADS>/WDMS_fitting_pipeline:
    repository resources : filters/ and tables/ next to this file
    Koester DA spectra   : <WORKSPACE>/models/koester2/ (or $WDMS_MODELS_DIR)
    atmosphere cache     : <WORKSPACE>/model_cache/ (or $SED_MODEL_CACHE)
    target data          : <WORKSPACE>/TIC<id>/  (notebook folders, read only)
    SPECULOOS figure     : <WORKSPACE>/TIC<id>/speculoos_light_curve/
                           TIC<id>_speculoos_light_curve.pdf
    pipeline results     : <WORKSPACE>/TIC<id>/SED-dilution analysis/
                           TIC<id>_Gaia<gaia_id>/{sed, dilution, joint_ideal, ...}
    whole-sample CSVs    : <WORKSPACE>/SED-dilution summaries/

Every scientific setting is a module-level constant in the "Global settings"
section below. Per-star parameters are defined in the TARGETS list, and the
single-target example in EXAMPLE_TARGET.

MODE = "test" only shortens the sampling (fewer live points, fewer Monte Carlo
draws, a coarser dilution grid). It is a quick diagnostic run, not a converged
result: model grids are still loaded in full, and network queries and
first-time downloads still happen. Use MODE = "final" for any number you intend
to quote.
"""

import os
import sys
from pathlib import Path

# The pipeline modules sit next to this file: make them importable whatever
# the current working directory or the way Python was started.
_REPO_DIR = str(Path(__file__).resolve().parent)
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

import wdms_paths as paths  # noqa: E402
# ══════════════════════════════════════════════════════════════════════════════
# Global settings  (shared by all stars)
# ══════════════════════════════════════════════════════════════════════════════

# ── Run mode ──────────────────────────────────────────────────────────────────
# "test"  : quick diagnostic. SED fit: 16 live points, at most 20 000 iterations.
#           Dilution: 5 000 Monte Carlo draws, 10 x 10 x n(T_MS) likelihood grid.
#           Joint fits: 16 live points (ideal fit: at most 20 000 iterations).
# "final" : production. SED fit: 500 live points, dlogz = 0.2. Dilution: 50 000
#           draws, 30 x 30 x n(T_MS) grid. Joint fits: 400 live points (ideal
#           fit: dlogz = 0.2).
# Can be overridden from the command line with --mode.
MODE = "test"

# ── Run controls ──────────────────────────────────────────────────────────────
# What a run does. Set them here and run this file directly (e.g. the "Run
# Python File" button of VS Code), or use the matching command-line options;
# an option given on the command line always wins.
RUN_EXAMPLE        = False  # True: run EXAMPLE_TARGET only, not TARGETS   (--example)
CHECK_ONLY         = False  # True: check the inputs and stop, no fit       (--check)
FRESH              = False  # True: delete each target's previous results   (--fresh)
                            #       folder before running it
REFRESH_PHOTOMETRY = False  # True: also download its photometry again      (--refresh-photometry)
SUMMARIES_ONLY     = False  # True: only rebuild the summary tables from    (--summaries-only)
                            #       the results already on disk, no fit

# ── Project paths ─────────────────────────────────────────────────────────────
# Repository resources are located from this file, whatever the current
# working directory and wherever the repository was downloaded or cloned.
# PROJECT_ROOT is the folder of this file, <repository>/2_sed_dilution/.
PROJECT_ROOT = paths.CODE_DIR

# Downloads folder containing the workspace <WORKSPACE> =
# <DOWNLOADS>/WDMS_fitting_pipeline/, which holds the models, the caches, the
# per-target data (<WORKSPACE>/TIC<id>/) and every result.
# None = the user's Downloads folder, found automatically (wdms_paths.
# resolve_downloads_dir: the WDMS_DOWNLOADS_DIR environment variable, then the
# platform's Downloads folder). Overridden by --downloads-dir on the command line.
# Each target's results go to <WORKSPACE>/TIC<id>/SED-dilution analysis/, the
# whole-sample summaries to <WORKSPACE>/SED-dilution summaries/.
DOWNLOADS_DIR = None

# ── Atmospheric model directories and cooling tracks ──────────────────────────
# MODELS_DIR = None: <DOWNLOADS>/WDMS_fitting_pipeline/models/, or the folder
# given by the environment variable WDMS_MODELS_DIR. It holds the Koester DA
# spectra (koester2/), which are downloaded, not stored in the repository.
# TABLES_DIR = None: the tables shipped with the repository,
# <repository>/2_sed_dilution/tables/ (Bédard 2020, BHAC15, La Plata).
# Set a Path in either to force another folder. The paths below are filled in
# at start-up by configure_paths() and read by every stage;
# sed_atmosphere_models.py reads MODELS_DIR for the Koester grid too.
# See docs/DATA_REQUIREMENTS.md for the expected content.
# (models/phoenix2021, the PHOENIX-ACES SpecInt files, is no longer read: the
# eclipse-depth term uses the SED companion grid; see docs/KNOWN_ISSUES.md.)
MODELS_DIR   = None
TABLES_DIR   = None
KOESTER_DIR  = None   # <MODELS_DIR>/koester2/
DA_TRACK     = None   # <TABLES_DIR>/bedard2020/Table_DA.csv
DB_TRACK     = None   # <TABLES_DIR>/bedard2020/Table_DB.csv
BHAC15_CSV   = None   # <TABLES_DIR>/BHAC2015/BHAC15_tracks.csv
# He-core white dwarf sequences (Althaus et al. 2013), needed to test the core
# composition against the measured (T_WD, R_WD) rather than assume it.
LAPLATA_DAT  = None   # <TABLES_DIR>/laplata2013/DAwhitedwarfsZ0.02.dat

# ── Filter transmission profile files (same for all stars) ────────────────────
# z, i : same iz file, wavelength-truncated via FILTER_WAVELENGTH_CUTS
# exo  : clear filter file
FILTER_PROFILE_FILES = {
    "iz":  PROJECT_ROOT / "filters" / "SPECULOOS_iz.dat",
    "i":   PROJECT_ROOT / "filters" / "SPECULOOS_iz.dat",
    "z":   PROJECT_ROOT / "filters" / "SPECULOOS_iz.dat",
    "exo": PROJECT_ROOT / "filters" / "SPECULOOS_clear.dat",
}

# ── Wavelength cuts per filter (same for all stars) ───────────────────────────
FILTER_WAVELENGTH_CUTS = {
    "exo": {"min_A": 4750.0},
    "i":   {"max_A": 8150.0},
    "z":   {"min_A": 8050.0},
}

# ── BHAC15 / T_MS grid ────────────────────────────────────────────────────────
# Companion temperature fitting bound, passed explicitly to the SED fit, both
# dilution methods and the ideal joint fit (same values as the module defaults,
# wdms_common.T_MS_MIN_K / T_MS_MAX_K). Model grids that do not cover it narrow
# it with a warning; nothing is extrapolated. White-dwarf bounds are separate.
T_MS_MIN             = 2300.0   # K
T_MS_MAX             = 4000.0   # K
T_MS_STEP            = 100.0    # K
BHAC_LOCAL_DELTA_T   = 100.0    # K
BHAC_LOCAL_MIN_PTS   = 3
BHAC_AGE_LOGT_MIN    = 8.0      # log10 yr  — exclude pre-MS
BHAC_AGE_LOGT_MAX    = None
# Reject photometry with σ/F above this value. Passed explicitly to both the SED
# fit and the SED term of the ideal joint fit, so both select the same points.
SED_MAX_REL_ERR_DEFAULT = 0.25

# Joint-fit method:
#   "ideal"  : joint_fit_ideal.py — one nested-sampling run on the raw SED
#              photometry, the eclipse depths and the Gaia parallax.
#   "legacy" : joint_fit.py — KDE of the SED posterior multiplied by the
#              tabulated dilution likelihood (lnL_dil.npz). Needs the BHAC15
#              dilution branch, i.e. MS_RADIUS_PRIOR = "bhac15" or "brown2022".
JOINT_FIT_METHOD = "ideal"

if JOINT_FIT_METHOD not in {"legacy", "ideal"}:
    raise ValueError(
        f"JOINT_FIT_METHOD must be 'legacy' or 'ideal', got {JOINT_FIT_METHOD!r}"
    )


# ── Physics flags ─────────────────────────────────────────────────────────────
# Only DA white-dwarf atmospheres are currently supported consistently
# throughout the pipeline: every stage reads Koester DA spectra (da*.dk.dat.txt)
# and the SED stage always takes its gravity from DA_TRACK. WD_ATMO_TYPE only
# switches the Bédard 2020 cooling track (DA_TRACK or DB_TRACK) used for logg_WD
# in the dilution and joint stages, so "DB" would pair a DB track with DA
# spectra. Keep "DA"; DB support may be added later (docs/KNOWN_ISSUES.md).
WD_ATMO_TYPE                  = "DA"
# Band integration of the eclipse-depth filters. True: photon-weighted,
# ∫F T λ dλ / ∫T λ dλ (photon-counting detectors), the same convention as the
# SED term. False: energy-weighted, ∫F T dλ / ∫T dλ (depth term only).
PHOTON_COUNTING               = True

# ── Companion spectral model in the SED fit ───────────────────────────────────
# This setting changes the M dwarf's model only. The white-dwarf model is set
# separately by SED_WD_ATMOSPHERE below and is the same for every variant, so
# any parameter shift between variants measures the companion model alone.
#
#   "blackbody" : monochromatic Planck at each filter's effective wavelength
#                 (historical behaviour, the numbers currently in Table 2)
#   "bb-int"    : same blackbodies, integrated through the real transmissions
#   "phoenix"   : PHOENIX (Allard/STScI) grid for the companion
#   "btsettl"   : BT-Settl CIFIST grid for the companion
#
# SED_MS_ATMOSPHERE selects the headline fit, i.e. the posterior handed to the
# dilution and joint stages. Together with SED_ATMO_MS_LOGG_MODE it also sets the
# companion model and gravity of every eclipse-depth calculation (depth_model).
SED_MS_ATMOSPHERE = "phoenix"

# Extra companion models refitted on the same photometry with the same priors,
# to quantify the systematic. None disables it and leaves the runtime as before;
# "all" runs every variant. Each extra variant costs one more nested-sampling
# run per star.
#
# Beware of "btsettl": the SVO server sends ~66 MB per (Teff, logg) node, so the
# ranges set below already amount to ~2 GB on the first run (cached afterwards).
SED_COMPARE_ATMOSPHERES = ["phoenix", "blackbody"]

# Gravity used when reading the companion's atmosphere grid:
#   "ms_relation" — from the empirical dwarf Teff-logg sequence
#   "from_radius" — from the empirical mass at T_MS and the fitted R_MS
#   "fixed"       — SED_ATMO_MS_LOGG_FIXED
SED_ATMO_MS_LOGG_MODE  = "ms_relation"
SED_ATMO_MS_LOGG_FIXED = 5.0

# ── White dwarf spectral model in the SED term ────────────────────────────────
#   "blackbody" : historical behaviour, kept for reproducibility
#   "koester"   : Koester DA model atmospheres from models/koester2/
#
# The blackbody is a poor description of a hot DA: its optical continuum forms
# below the surface, where the gas is hotter than Teff, and much of its flux
# leaves below the Lyman limit. Fitting a blackbody to optical photometry
# therefore returns a Teff too high by 15-20 per cent and a radius too small by
# about a quarter, which roughly doubles the mass read off a cooling track and
# pushes cores towards spurious CO classifications. "koester" also makes the
# SED term consistent with the eclipse-depth term, which is Koester either way.
SED_WD_ATMOSPHERE = "koester"

# Gravity used when reading the white dwarf's atmosphere grid:
#   "cooling_track" — mass interpolated on DA_TRACK at the sampled (T_WD, R_WD)
#   "fixed"         — SED_ATMO_WD_LOGG_FIXED
# A fixed value is defensible above ~40 000 K, where logg moves the spectral
# shape by well under a per cent, but not near 10 000 K where it reaches 20.
SED_ATMO_WD_LOGG_MODE  = "cooling_track"
SED_ATMO_WD_LOGG_FIXED = 7.75

# Extent of the grids to load. The PHOENIX defaults match the grid already
# cached on disk, so nothing is re-downloaded.
SED_ATMO_GRID_RANGES = {
    "phoenix-stsci":   {"teff": (2300.0, 5900.0), "logg": (4.0, 5.5)},
    "bt-settl-cifist": {"teff": (2600.0, 4200.0), "logg": (5.0, 5.5)},
}

# Where atmosphere grids and filter transmissions are cached. None means
# <DOWNLOADS>/WDMS_fitting_pipeline/model_cache/, or the environment variable
# SED_MODEL_CACHE if set (resolved at start-up by configure_paths()).
# Anything missing from the cache (STScI PHOENIX spectra, SVO filter curves,
# BT-Settl spectra) is downloaded on first use and kept there.
SED_ATMO_CACHE_DIR = None

# True: stop after the SED step of every star. Meant for the systematics study,
# where the dilution and joint stages would only add runtime.
SED_ONLY = False 

# ── M dwarf radius: prior in the fits, relation applied afterwards ────────────
# These are two separate decisions and are kept separate here.
#
# MS_RADIUS_PRIOR sets what the joint fit assumes about the companion before
# looking at the data:
#
#   "uniform"   : R_MS ~ U(0.05, 0.80) R_sun, the same flat prior as the SED
#                 fit. T_MS and R_MS are then independent a priori, no
#                 evolutionary relation enters the sampled parameters, and the
#                 SED and the eclipse depths alone place the companion in the
#                 temperature-radius plane. A radius above the models is then a
#                 measurement rather than something the prior forbade.
#   "bhac15"    : R_MS | T_MS from the theoretical BHAC15 radii, the historical
#                 behaviour
#   "brown2022" : same, with the radii scaled to the semi-empirical relation of
#                 Brown et al. 2022, so the 5-15 per cent inflation M dwarfs
#                 show relative to the models is carried by the prior
#
# MS_MASS_RELATION only converts the fitted radius into a mass afterwards
# (mass.py), which stays possible whatever the prior was:
#
#   "bhac15"    : theoretical 1 Gyr isochrone
#   "brown2022" : semi-empirical relation, extended by BHAC15 below
#                 0.121 M_sun where the two cross
MS_RADIUS_PRIOR  = "bhac15"
MS_MASS_RELATION = "brown2022"

# Dilution follows the same switch. "uniform" calls dilution_uniform_fit.py:
# T_MS and R_MS have the same flat priors as the joint fit, and the eclipse
# depth is inverted for R_MS(T_MS) with no BHAC15 relation. "bhac15" and
# "brown2022" keep the historical dilution_fit.py, which still walks along
# R(T). Mixing a uniform joint fit with a Baraffe dilution overlay is what
# made the orange corner-plot cloud look identical between the two modes.

# ── Model-free eclipse-radius diagnostic ──────────────────────────────────────
# Inverts the dilution equation for the radius instead of solving it along a
# relation:
#
#   R_MS(T_MS) = R_WD sqrt[ (1/depth - 1) F_WD / F_MS(T_MS) ]
#
# and lays the resulting band over the SED posterior in the (T_MS, R_MS) plane.
# The figure then shows "what the eclipse demands" against "what the SED
# demands" with no radius-temperature relation in the construction, which tests
# whether the result depends on such a relation. It only needs the SED posterior, so
# it also runs with SED_ONLY = True; the joint posterior is added when the joint
# stage has run.
RUN_ECLIPSE_RADIUS_DIAGNOSTIC = False

# All that survives of the models is the companion band flux. The reference
# band uses the shared companion model and gravity prescription (the SED
# settings); these fixed gravities are scanned on top of it to measure the
# residual gravity dependence: the figure and the JSON report how far the
# required radius moves, in per cent and in units of the width of the band.
ECLIPSE_MS_LOGG_VALUES    = (4.5, 5.0, 5.5)
ECLIPSE_T_MS_STEP         = 25.0    # K, temperature axis of the curves
ECLIPSE_N_DRAWS           = 4000    # SED posterior draws folded through it
# Draw the BHAC15 relation as a faint reference line. Nothing uses it; it is
# there to show what the relation would have imposed.
ECLIPSE_SHOW_BHAC15       = False

# ── Masses, core composition and orbital configuration ────────────────────────
# False skips the whole stage. It needs the joint posterior, so it also needs
# SED_ONLY = False. Orbital quantities are only produced for the targets whose
# period_d is filled in below.
RUN_MASS_ANALYSIS = True
MASS_SEED         = 42
MASS_MAX_SAMPLES  = 20000   # posterior samples folded through the relations
#
# Which cooling sequences supply the adopted M_WD (and cooling age, and the
# masses that enter the orbital quantities):
#
#   "auto" : current behaviour — He (Althaus 2013) or CO (Bédard 2020) is
#            chosen from the measured (T_WD, R_WD) by the Brown et al. rule
#   "co"   : always read M_WD off the Bédard et al. 2020 CO-core sequences
#            (DA_TRACK). The He grid is still evaluated for the diagnostic
#            columns and the plot, but it never becomes the reported mass.
WD_MASS_TRACK = "co"   # "auto" | "co"

# ══════════════════════════════════════════════════════════════════════════════
# Fit settings (passed to the modules)
# ══════════════════════════════════════════════════════════════════════════════
# The values below are the module defaults. apply_fit_settings() copies them
# into sed_fit, dilution_fit, dilution_uniform_fit, joint_fit, joint_fit_ideal
# and mass before the first target, so changing them here changes every stage
# that uses them. They are written to each target's run_info.json.

# ── SED photometric uncertainties ─────────────────────────────────────────────
# σ of every photometric point is raised to at least FRAC_FLOOR × F (catalogue
# uncertainties are far smaller than the model systematics). Used by the SED
# fit and by the SED term of the ideal joint fit.
FRAC_FLOOR = 0.18

# ── SED outlier clipping ──────────────────────────────────────────────────────
# Off by default. When on, a point is rejected when |residual − median| exceeds
# SED_MAD_SIGMA robust sigmas (1.4826 × MAD) around a continuum rebuilt after
# each pass; a sampling guard keeps enough optical and infrared points.
SED_OUTLIER_CLIPPING   = False   # True: apply the clipping
SED_OUTLIER_ALL_POINTS = False   # True: test every point; False: Gaia points only
SED_MAD_SIGMA          = 3.0     # clipping threshold [robust σ]
SED_MAX_REJECT_FRAC    = 0.2     # at most this fraction of the tested points rejected
SED_OUTLIER_MAX_ITERS  = 1       # clipping passes
SED_REJECT_BOTH        = True    # True: points above and below; False: above only

# ── Prior bounds of the SED and ideal joint fits ──────────────────────────────
# T_MS_MIN / T_MS_MAX are set above. R_MS_MIN_RSUN / R_MS_MAX_RSUN also bound the
# uniform dilution stage. White-dwarf radius prior: log-uniform; the others uniform.
T_WD_MIN      = 5_000.0     # K
T_WD_MAX      = 100_000.0   # K
R_WD_MIN_RSUN = 0.002       # R_sun
R_WD_MAX_RSUN = 0.08        # R_sun
R_MS_MIN_RSUN = 0.05        # R_sun
R_MS_MAX_RSUN = 0.80        # R_sun
AV_MIN        = 0.0         # mag
AV_MAX        = 0.6         # mag

# ── Sampling per MODE ─────────────────────────────────────────────────────────
# UltraNest keywords (min_num_live_points, max_iters, dlogz) and Monte Carlo
# sizes used by MODE = "test" and MODE = "final".
SED_SAMPLING = {                       # SED fit, 1- and 2-component runs
    "test":  dict(min_num_live_points=16, max_iters=20_000),
    "final": dict(min_num_live_points=500, dlogz=0.2),
}
DILUTION_SAMPLING = {                  # dilution: MC draws, lnL grid axes (BHAC15 branch)
    "test":  dict(n_mc=5_000,  n_grid=10),
    "final": dict(n_mc=50_000, n_grid=30),
}
JOINT_IDEAL_SAMPLING = {               # ideal joint fit
    "test":  dict(min_num_live_points=16, max_iters=20_000),
    "final": dict(min_num_live_points=400, dlogz=0.2),
}
JOINT_LEGACY_N_LIVE = {"test": 16, "final": 400}   # legacy joint fit: live points

# ── Mass systematic floors ────────────────────────────────────────────────────
# Fractional floors added in quadrature to the statistical mass uncertainty
# (Brown et al. 2022, sect. 2.1).
MS_MASS_REL_SYST = 0.05    # M dwarf
WD_MASS_REL_SYST = 0.01    # white dwarf

FIT_SETTINGS = (
    "FRAC_FLOOR", "SED_MAX_REL_ERR_DEFAULT",
    "SED_MAD_SIGMA", "SED_MAX_REJECT_FRAC", "SED_OUTLIER_MAX_ITERS", "SED_REJECT_BOTH",
    "T_WD_MIN", "T_WD_MAX", "R_WD_MIN_RSUN", "R_WD_MAX_RSUN",
    "R_MS_MIN_RSUN", "R_MS_MAX_RSUN", "AV_MIN", "AV_MAX",
    "SED_SAMPLING", "DILUTION_SAMPLING", "JOINT_IDEAL_SAMPLING", "JOINT_LEGACY_N_LIVE",
    "MS_MASS_REL_SYST", "WD_MASS_REL_SYST",
)

# ── Seeds ─────────────────────────────────────────────────────────────────────
# SED_SEED is the master seed of the SED stage: each UltraNest run (1-component,
# and each 2-component companion variant) gets its own sub-seed derived from it
# and from the variant name (wdms_common.derive_seed). JOINT_SEED seeds the
# joint-fit UltraNest run (legacy or ideal) the same way. DILUTION_SEED drives
# the Monte Carlo draws of the dilution stage and the eclipse diagnostic. The
# seeds used are written to the params_*.json files. They make a run repeatable
# on the same software stack; results can still differ across library versions,
# platforms, or when an earlier UltraNest run is resumed.
SED_SEED      = 42
DILUTION_SEED = 42
JOINT_SEED    = 42

# ── Composite figure ──────────────────────────────────────────────────────────
# Light-curve panel of the composite figure: the SPECULOOS figure saved by the
# notebook, <WORKSPACE>/TIC<id>/speculoos_light_curve/TIC<id>_speculoos_light_curve.pdf
# (per-target override: the "speculoos_figure" key of a TARGETS entry). When it
# is missing, only the composite figure is skipped, with a warning.
CORNER_SIDE_PX = 980    # size of the corner plot square [px]
PAD_H          = 24     # horizontal gap between left and right columns [px]
PAD_V          = 20     # vertical gap between transit and SED in left column [px]
COMPOSITE_DPI  = 150    # DPI used when rendering PDF → PIL


# ══════════════════════════════════════════════════════════════════════════════
# Per-star parameters
# ══════════════════════════════════════════════════════════════════════════════
# Every uncommented entry of TARGETS is analysed, one after the other. To add a
# star, uncomment the "Star 1" block below (remove the leading "# " of each
# line), replace every TODO, and copy the block for further stars. The values
# come from Part 1 (1_lightcurves_environment/Light_curves_analysis.ipynb) and
# from the Gaia archive. To run the predefined example instead, use --example or
# RUN_EXAMPLE = True (see EXAMPLE_TARGET below).
#
# What to write on each line:
#   gaia_id              : Gaia DR3 source_id of the system (integer, from the Gaia archive)
#   tic_id               : TESS Input Catalog ID, 123456789 or "TIC123456789";
#                          None = looked up on MAST from the Gaia position
#   depth_by_filter      : eclipse depth Δ and its 1σ error per SPECULOOS band, from
#                          Part 1, as fractions of the out-of-eclipse flux:
#                          67.8 ± 1.5 % → {"iz": (0.678, 0.015)}. Band keys: "exo",
#                          "i", "z", "iz"; one entry per measured band, e.g.
#                          {"exo": (0.827, 0.016), "iz": (0.678, 0.015)}
#   sed_exclude_windows  : wavelength windows removed from the SED, [(min, max), ...]
#                          in µm; [] = keep every point
#   sed_below_um         : remove SED points bluer than this wavelength [µm]; 0 = keep all
#   sed_above_um         : remove SED points redder than this wavelength [µm]
#   vizier_radius_arcsec : radius of the VizieR photometry search around the Gaia
#                          position [arcsec]; keep it small (1–3) so that
#                          neighbouring stars are not included
#   period_d             : orbital period from Part 1 [days]; None only skips the
#                          orbital quantities of the mass stage
# Optional lines:
#   speculoos_figure     : composite light-curve panel, if the file in
#                          speculoos_light_curve/ is not TIC<id>_speculoos_light_curve.pdf
#   speculoos_csv_files  : {band: file name}, for SPECULOOS CSVs in speculoos_csv/
#                          not named TIC<id>_<band>.csv
# Entries with gaia_id = None or an empty depth_by_filter are skipped.

TARGETS = [
    #  # ── Star 1 ───────────────────────────────────────────────────────────────
    # dict(
    #     gaia_id              = ,   # TODO
    #     tic_id               = None,
    #     depth_by_filter      = {"Filter": (0.xx, 0.xx)},     # TODO
    #     sed_exclude_windows  = [],
    #     sed_below_um         = 0,
    #     sed_above_um         = 10.0,
    #     vizier_radius_arcsec = 1,
    #     period_d             = ,   # TODO
    # ),
]


# ══════════════════════════════════════════════════════════════════════════════
# Single-target example  (python main_all.py --example)
# ══════════════════════════════════════════════════════════════════════════════
# TIC 450781262, the system of the related publication. The TIC ID is filled in
# so that no MAST lookup is needed; it matches the result folder name
# TIC450781262_Gaia5346312514819760896 used by earlier runs. The example uses
# every global setting above. Its results go to the target's own folder,
# <WORKSPACE>/TIC450781262/SED-dilution analysis/, like any other target.
EXAMPLE_TARGET = dict(
    gaia_id              = 5346312514819760896,
    tic_id               = 450781262,
    depth_by_filter      = {"iz": (0.678, 0.015)},
    sed_exclude_windows  = [(0.34, 0.35), (0.905, 0.91)],
    sed_below_um         = 0,
    sed_above_um         = 10.0,
    vizier_radius_arcsec = 1,
    period_d             = 0.09388575,
)



# ══════════════════════════════════════════════════════════════════════════════
# Input checks, run before any calculation
# ══════════════════════════════════════════════════════════════════════════════

# Python packages imported somewhere in the pipeline (import name → pip name).
_REQUIRED_PACKAGES = {
    "numpy": "numpy", "scipy": "scipy", "matplotlib": "matplotlib",
    "pandas": "pandas", "astropy": "astropy", "astroquery": "astroquery",
    "ultranest": "ultranest", "certifi": "certifi", "requests": "requests",
}
# Only needed by the composite figure (step 6), which is skipped without them.
_OPTIONAL_PACKAGES = {"PIL": "pillow", "fitz": "pymupdf", "pdf2image": "pdf2image"}

_KOESTER_WD_ATMOSPHERES = {"koester", "koester2", "koester-da", "da", "atmosphere"}
KOESTER_N_SPECTRA = 1066   # spectra in the complete SVO "koester2" grid (82 Teff x 13 logg)


def configure_paths(downloads):
    """Fill in the model and cache locations from the resolved Downloads folder.

    MODELS_DIR (unless set explicitly above) becomes <workspace>/models/ or
    $WDMS_MODELS_DIR; TABLES_DIR (unless set) the repository's tables/ folder;
    SED_ATMO_CACHE_DIR (unless set) becomes
    <workspace>/model_cache/ or $SED_MODEL_CACHE. WDMS_MODELS_DIR is then set
    for this process so that the SED term's Koester reader
    (sed_atmosphere_models) uses the same folder.
    """
    global MODELS_DIR, TABLES_DIR, KOESTER_DIR, DA_TRACK, DB_TRACK, BHAC15_CSV, LAPLATA_DAT
    global SED_ATMO_CACHE_DIR
    MODELS_DIR = Path(MODELS_DIR) if MODELS_DIR is not None else paths.models_dir(downloads)
    TABLES_DIR = Path(TABLES_DIR) if TABLES_DIR is not None else paths.TABLES_DIR
    KOESTER_DIR = MODELS_DIR / "koester2"
    DA_TRACK    = TABLES_DIR / "bedard2020" / "Table_DA.csv"
    DB_TRACK    = TABLES_DIR / "bedard2020" / "Table_DB.csv"
    BHAC15_CSV  = TABLES_DIR / "BHAC2015" / "BHAC15_tracks.csv"
    LAPLATA_DAT = TABLES_DIR / "laplata2013" / "DAwhitedwarfsZ0.02.dat"
    if SED_ATMO_CACHE_DIR is None:
        SED_ATMO_CACHE_DIR = paths.atmosphere_cache_dir(downloads)
    os.environ[paths.MODELS_ENV_VAR] = str(MODELS_DIR)


def apply_fit_settings():
    """Copy the FIT_SETTINGS values of this file into the pipeline modules.

    A module that imported a setting by name (from sed_fit import AV_MIN) holds
    its own copy, so every module that has the name is updated. Returns the
    applied values, for run_info.json.
    """
    import copy
    import importlib

    applied = {}
    for name in FIT_SETTINGS:
        value = globals()[name]
        found = False
        for mod_name in ("sed_fit", "dilution_fit", "dilution_uniform_fit",
                         "joint_fit", "joint_fit_ideal", "mass"):
            module = importlib.import_module(mod_name)
            if hasattr(module, name):
                setattr(module, name, copy.deepcopy(value))
                found = True
        if not found:
            raise RuntimeError(f"apply_fit_settings: no module defines {name}")
        applied[name] = value
    return applied


def check_inputs(targets, downloads=None):
    """Check local files and Python packages needed by the current settings.

    Nothing is imported, downloaded or computed. Returns (errors, notes): each
    error would stop the pipeline later on, each note flags something optional
    or a download that will happen on the first run. `downloads` is the
    resolved Downloads folder (needed for the per-target optional files).
    """
    import importlib.util

    errors, notes = [], []
    missing = {}   # path → (description, [stages needing it]), reported once

    def need(path, what, used_by):
        path = Path(path)
        if not path.exists():
            entry = missing.setdefault(str(path), (what, []))
            if used_by not in entry[1]:
                entry[1].append(used_by)
            return False
        return True

    def need_files(directory, pattern, what, used_by):
        if need(directory, what, used_by) and not any(Path(directory).glob(pattern)):
            need(Path(directory) / pattern, what, used_by)

    # ── Python packages ───────────────────────────────────────────────────
    for module, pip_name in _REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(module) is None:
            errors.append(f"Python package '{pip_name}' is not installed "
                          f"(import name '{module}').")
    if importlib.util.find_spec("PIL") is None or (
            importlib.util.find_spec("fitz") is None
            and importlib.util.find_spec("pdf2image") is None):
        notes.append("Composite figure (step 6) needs pillow and either pymupdf "
                     "or pdf2image; it will be skipped or fail without them.")

    # ── Stage by stage, following the switches above ──────────────────────
    sed_koester = str(SED_WD_ATMOSPHERE).strip().lower() in _KOESTER_WD_ATMOSPHERES
    if sed_koester:
        need_files(KOESTER_DIR, "da*.dk.dat.txt", "Koester DA spectra",
                   "the SED fit (SED_WD_ATMOSPHERE = 'koester')")
        if SED_ATMO_WD_LOGG_MODE == "cooling_track":
            need(DA_TRACK, "Bédard 2020 DA cooling track",
                 "the SED fit (SED_ATMO_WD_LOGG_MODE = 'cooling_track')")

    depth_stages = (not SED_ONLY) or RUN_ECLIPSE_RADIUS_DIAGNOSTIC
    if depth_stages:
        used_by = "the dilution / joint / eclipse-radius stages"
        need_files(KOESTER_DIR, "da*.dk.dat.txt", "Koester DA spectra", used_by)
        track = DB_TRACK if str(WD_ATMO_TYPE).strip().upper() == "DB" else DA_TRACK
        need(track, f"Bédard 2020 {WD_ATMO_TYPE} cooling track", used_by)
        for star in targets:
            for fk in star["depth_by_filter"]:
                if fk not in FILTER_PROFILE_FILES:
                    errors.append(
                        f"Gaia {star['gaia_id']}: depth filter '{fk}' has no entry "
                        f"in FILTER_PROFILE_FILES (known: {sorted(FILTER_PROFILE_FILES)}).")
                else:
                    need(FILTER_PROFILE_FILES[fk], f"filter profile '{fk}'", used_by)

    bhac_users = []
    if not SED_ONLY and (MS_RADIUS_PRIOR != "uniform" or JOINT_FIT_METHOD == "ideal"):
        bhac_users.append("the dilution / joint stages")
    if RUN_MASS_ANALYSIS or MS_RADIUS_PRIOR == "brown2022":
        bhac_users.append("the M dwarf mass-radius relation")
    if RUN_ECLIPSE_RADIUS_DIAGNOSTIC and ECLIPSE_SHOW_BHAC15:
        bhac_users.append("the eclipse-radius reference curve")
    if bhac_users:
        need(BHAC15_CSV, "BHAC15 evolutionary tracks", " and ".join(bhac_users))

    if RUN_MASS_ANALYSIS and not SED_ONLY:
        need(DA_TRACK, "Bédard 2020 DA cooling track", "the mass stage (CO cores)")
        need(LAPLATA_DAT, "La Plata DA table", "the mass stage (He cores)")

    # A partly downloaded Koester folder would silently narrow the white-dwarf
    # grid: require the complete set (1 066 spectra in the SVO koester2 grid).
    if (sed_koester or depth_stages) and Path(KOESTER_DIR).is_dir():
        n_koester = len(list(Path(KOESTER_DIR).glob("da*.dk.dat.txt")))
        if 0 < n_koester < KOESTER_N_SPECTRA:
            errors.append(f"Koester folder incomplete: {n_koester}/{KOESTER_N_SPECTRA} spectra in "
                          f"{KOESTER_DIR}\n      Rerun the download cell of "
                          f"0_setup/setup_pipeline.ipynb (files already present are kept).")

    # ── Target identifiers and optional per-target files ──────────────────
    for star in targets:
        if star.get("tic_id") is None:
            continue
        try:
            paths.normalize_tic(star["tic_id"])
        except ValueError as exc:
            errors.append(f"Gaia {star['gaia_id']}: {exc}")
            continue
        if downloads is None or SED_ONLY:
            continue
        fig = paths.speculoos_figure_path(downloads, star["tic_id"], star.get("speculoos_figure"))
        if not fig.exists():
            notes.append(f"Optional SPECULOOS figure not found: {fig} "
                         f"(only the composite figure will be skipped).")

    if any(star.get("tic_id") is None for star in targets):
        notes.append("At least one target has tic_id = None: its TIC ID will be "
                     "looked up on MAST (network access needed).")

    cache = (Path(SED_ATMO_CACHE_DIR) if SED_ATMO_CACHE_DIR is not None else
             paths.atmosphere_cache_dir(downloads) if downloads is not None else
             Path(os.environ.get(paths.CACHE_ENV_VAR, "model_cache")))
    variants = {str(SED_MS_ATMOSPHERE).lower()}
    compare = SED_COMPARE_ATMOSPHERES
    if isinstance(compare, str):
        variants |= {"phoenix", "btsettl"} if compare.lower() == "all" else {compare.lower()}
    elif compare:
        variants |= {str(v).lower() for v in compare}
    grids = cache / "grids"
    if variants & {"phoenix", "phoenix-stsci"} and not any(grids.glob("phoenix-stsci_*.npz")):
        notes.append(f"No cached STScI PHOENIX grid in {grids}: the SED stage will "
                     f"download one FITS file per Teff (~12 MB each) on first use.")
    if variants & {"btsettl", "bt-settl", "bt-settl-cifist"}:
        notes.append("BT-Settl is requested: about 66 MB per (Teff, logg) node is "
                     "downloaded from SVO on first use (several GB in total).")
    if not (cache / "filters").exists():
        notes.append(f"No cached filter curves in {cache / 'filters'}: SVO "
                     f"transmissions will be downloaded on first use.")
    notes.append("Every run queries the Gaia archive (astrometry) and VizieR "
                 "(photometry; cached in <output>/downloads/): network access needed.")

    for path, (what, users) in missing.items():
        errors.append(f"Missing {what}: {path}\n      needed by {'; '.join(users)}. "
                      f"See docs/DATA_REQUIREMENTS.md.")
    return errors, notes


# ══════════════════════════════════════════════════════════════════════════════
# TIC ID auto-lookup
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_tic_id(gaia_id, radius_arcsec=1.0):
    from sed_fit import fetch_gaia_astrometry
    from astroquery.mast import Catalogs
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    import numpy as np

    try:
        ra, dec, _, _ = fetch_gaia_astrometry(gaia_id)
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        tbl = Catalogs.query_region(coord, radius=radius_arcsec * u.arcsec, catalog="TIC")
        if len(tbl) == 0:
            return None
        tic_coords = SkyCoord(ra=tbl["ra"] * u.deg, dec=tbl["dec"] * u.deg, frame="icrs")
        j = int(np.argmin(coord.separation(tic_coords).arcsec))
        return int(tbl["ID"][j])
    except Exception as exc:
        print(f"[TIC lookup] Failed: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Single-star pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _clean_previous_results(output_base, tic_id, gaia_id, fresh, refresh_photometry):
    """Delete the previous results (and optionally photometry) of one target.

    Only folders strictly inside the target's analysis folder are removed; the
    notebook folders (tess_cutouts/, speculoos_csv/, …) are never touched.
    """
    import shutil

    base = Path(output_base).resolve()
    victims = []
    if fresh:
        victims.append(base / f"TIC{tic_id}_Gaia{gaia_id}")
    if refresh_photometry:
        victims.append(base / "downloads" / str(gaia_id))
    for folder in victims:
        folder = folder.resolve()
        if base not in folder.parents:          # safety: stay inside the analysis folder
            raise RuntimeError(f"Refusing to delete a folder outside {base}: {folder}")
        if folder.exists():
            shutil.rmtree(folder)
            print(f"[main_all] Deleted previous files: {folder}")


def _write_run_info(star_dir, star, tic_id, gaia_id, mode, status, started):
    """Record how this target was last run: <star_dir>/run_info.json.

    Read back by rebuild_summaries() to label each row of the summary tables
    (test or final, completed or not, when).
    """
    import json
    from datetime import datetime, timezone

    info = {
        "tic_id": int(tic_id),
        "gaia_id": int(gaia_id),
        "mode": mode,
        "status": status,                       # "running" until the run completes
        "started_utc": started,
        "finished_utc": (datetime.now(timezone.utc).isoformat(timespec="seconds")
                         if status == "completed" else None),
        "settings": {
            "SED_MS_ATMOSPHERE": SED_MS_ATMOSPHERE,
            "SED_COMPARE_ATMOSPHERES": SED_COMPARE_ATMOSPHERES,
            "SED_WD_ATMOSPHERE": SED_WD_ATMOSPHERE,
            "MS_RADIUS_PRIOR": MS_RADIUS_PRIOR,
            "JOINT_FIT_METHOD": JOINT_FIT_METHOD,
            "MS_MASS_RELATION": MS_MASS_RELATION,
            "WD_MASS_TRACK": WD_MASS_TRACK,
            "SED_ONLY": SED_ONLY,
            "RUN_MASS_ANALYSIS": RUN_MASS_ANALYSIS,
            "SED_OUTLIER_CLIPPING": SED_OUTLIER_CLIPPING,
            "SED_OUTLIER_ALL_POINTS": SED_OUTLIER_ALL_POINTS,
            **{name: globals()[name] for name in FIT_SETTINGS},
        },
        "target": {k: v for k, v in star.items() if k != "tic_id"},
    }
    Path(star_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(star_dir) / "run_info.json", "w", encoding="utf-8") as fp:
        json.dump(info, fp, indent=2, default=str)


def run_star(star, mode, ms_relation=None, ms_relation_prior=None,
             co_grid=None, he_grid=None, downloads=None,
             fresh=False, refresh_photometry=False):
    """Run the full SED → Dilution → Joint → Display → Mass pipeline for one star.

    The mass-radius models are built once by the caller and handed down, since
    rebuilding the cooling-track triangulations for every target is wasteful.

    ms_relation drives the mass derived at the end and nothing else.
    ms_relation_prior is None unless MS_RADIUS_PRIOR asks the fits themselves to
    use the inflated radii, which keeps the two choices independent.

    downloads is the resolved Downloads folder. Every stage of this target
    writes under the same output base, <downloads>/TIC<id>/SED-dilution
    analysis/, inside which the usual TIC<id>_Gaia<gaia_id>/ tree is kept.

    fresh=True deletes this target's previous results folder,
    <output base>/TIC<id>_Gaia<gaia_id>/, before the run, so that no file of an
    earlier run (e.g. a composite figure the new run skips) is left behind.
    refresh_photometry=True also deletes the cached VizieR photometry of this
    target, <output base>/downloads/<gaia_id>/, so it is downloaded again.
    Nothing outside the target's "SED-dilution analysis" folder is touched.
    """
    from sed_fit      import run_sed_fit
    from display      import run_display

    gaia_id = int(star["gaia_id"])

    # ── TIC ID ────────────────────────────────────────────────────────────
    tic_id = star["tic_id"]
    if tic_id is None:
        print(f"[main_all] TIC_ID not set for Gaia {gaia_id} — querying MAST…")
        tic_id = _fetch_tic_id(gaia_id, radius_arcsec=star["vizier_radius_arcsec"])
        if tic_id is None:
            raise RuntimeError(
                f"TIC ID could not be retrieved for Gaia {gaia_id}. "
                "Set tic_id manually in the TARGETS list."
            )
        print(f"[main_all] TIC ID found: {tic_id}")
    tic_id = paths.normalize_tic(tic_id)      # 450781262 and "TIC450781262" alike

    label = f"Gaia {gaia_id} / TIC {tic_id}"

    # ── Per-target folders ────────────────────────────────────────────────
    if downloads is None:
        downloads = paths.resolve_downloads_dir(DOWNLOADS_DIR)
    if KOESTER_DIR is None:
        configure_paths(downloads)
    data_dir    = paths.target_dir(downloads, tic_id)
    output_base = paths.ensure_dir(paths.analysis_dir(downloads, tic_id),
                                   "analysis output folder")
    lc_figure   = paths.speculoos_figure_path(downloads, tic_id,
                                              star.get("speculoos_figure"))
    sp_csv      = paths.speculoos_csv_paths(downloads, tic_id, star["depth_by_filter"],
                                            star.get("speculoos_csv_files"))
    print(f"[main_all] Target data folder    : {data_dir}"
          f"{'' if data_dir.is_dir() else '   (not found; created for the results only)'}")
    print(f"[main_all] Analysis output folder: {output_base}")
    _clean_previous_results(output_base, tic_id, gaia_id, fresh, refresh_photometry)
    from datetime import datetime, timezone
    star_dir = Path(output_base) / f"TIC{tic_id}_Gaia{gaia_id}"
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_run_info(star_dir, star, tic_id, gaia_id, mode, "running", started)
    print(f"[main_all] SPECULOOS figure      : {lc_figure}"
          f"{'' if lc_figure.exists() else '   (missing: composite figure will be skipped)'}")
    for band, csv_path in sp_csv.items():
        # Listed for reference only: the pipeline does not read these CSVs;
        # the eclipse depths come from depth_by_filter.
        print(f"[main_all] SPECULOOS CSV [{band:>3s}]  : {csv_path}"
              f"{'' if csv_path.exists() else '   (not found)'}")

    # ── Step 1: SED fit ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STEP 1 — SED FIT  [{label}]")
    print("=" * 60)

    res_sed = run_sed_fit(
        gaia_id                = gaia_id,
        tic_id                 = tic_id,
        output_base            = output_base,
        mode                   = mode,
        sed_exclude_windows_um = star["sed_exclude_windows"],
        sed_exclude_below_um   = star["sed_below_um"],
        sed_exclude_above_um   = star["sed_above_um"],
        enable_outlier_filter  = SED_OUTLIER_CLIPPING,
        filter_all_sed_points  = SED_OUTLIER_ALL_POINTS,
        outlier_max_iters      = SED_OUTLIER_MAX_ITERS,
        vizier_radius_arcsec   = star["vizier_radius_arcsec"],
        ms_atmosphere          = SED_MS_ATMOSPHERE,
        compare_atmospheres    = SED_COMPARE_ATMOSPHERES,
        atmo_cache_dir         = SED_ATMO_CACHE_DIR,
        atmo_grid_ranges       = SED_ATMO_GRID_RANGES,
        ms_logg_mode           = SED_ATMO_MS_LOGG_MODE,
        ms_logg_fixed          = SED_ATMO_MS_LOGG_FIXED,
        wd_atmosphere          = SED_WD_ATMOSPHERE,
        wd_logg_mode           = SED_ATMO_WD_LOGG_MODE,
        wd_logg_fixed          = SED_ATMO_WD_LOGG_FIXED,
        wd_cooling_track       = DA_TRACK,
        sed_max_rel_err        = SED_MAX_REL_ERR_DEFAULT,
        t_ms_min               = T_MS_MIN,
        t_ms_max               = T_MS_MAX,
        seed                   = SED_SEED,
    )

    print(f"\n— SED [{label}] —")
    print(f"  Model : {res_sed['model_preferred_sed']}  "
          f"ΔlogZ = {res_sed['dlogZ_sed']:.3f}")
    print(f"  T_WD = {res_sed['T_WD_sed']:.0f} K   R_WD = {res_sed['R_WD_sed']:.5f} R_sun")
    print(f"  T_MS = {res_sed['T_MS_sed']:.0f} K   R_MS = {res_sed['R_MS_sed']:.4f} R_sun")
    print(f"  d = {res_sed['d_sed']:.1f} pc   A_V = {res_sed['AV_sed']:.3f} mag")

    # Companion model, gravity prescription and white-dwarf gravity policy of
    # the SED term, shared by every eclipse-depth calculation (depth_model).
    depth_settings = dict(
        ms_atmosphere    = SED_MS_ATMOSPHERE,
        atmo_cache_dir   = SED_ATMO_CACHE_DIR,
        atmo_grid_ranges = SED_ATMO_GRID_RANGES,
        ms_logg_mode     = SED_ATMO_MS_LOGG_MODE,
        ms_logg_fixed    = SED_ATMO_MS_LOGG_FIXED,
        wd_logg_mode     = SED_ATMO_WD_LOGG_MODE,
        wd_logg_fixed    = SED_ATMO_WD_LOGG_FIXED,
    )

    # ── Model-free eclipse-radius diagnostic ──────────────────────────────
    # Defined here and called once, either right away when the pipeline stops
    # after the SED, or after the joint fit so its posterior can be overlaid.
    def _eclipse_radius_diagnostic(joint_samples_path=None):
        from dilution_fit import run_eclipse_radius_diagnostic
        print(f"\n{'='*60}")
        print(f"STEP 4 — ECLIPSE RADIUS DIAGNOSTIC  [{label}]")
        print("=" * 60)
        return run_eclipse_radius_diagnostic(
            gaia_id                       = gaia_id,
            tic_id                        = tic_id,
            output_base                   = output_base,
            depth_by_filter               = star["depth_by_filter"],
            filter_profile_files          = FILTER_PROFILE_FILES,
            posterior_samples_sed_path    = res_sed["posterior_samples_sed_path"],
            koester_dir                   = KOESTER_DIR,
            da_track_path                 = DA_TRACK,
            db_track_path                 = DB_TRACK,
            wd_atmo_type                  = WD_ATMO_TYPE,
            t_ms_min                      = T_MS_MIN,
            t_ms_max                      = T_MS_MAX,
            t_ms_step                     = ECLIPSE_T_MS_STEP,
            ms_logg_values                = ECLIPSE_MS_LOGG_VALUES,
            photon_counting               = PHOTON_COUNTING,
            filter_wavelength_cuts        = FILTER_WAVELENGTH_CUTS,
            posterior_samples_joint_path  = joint_samples_path,
            bhac15_csv                    = BHAC15_CSV if ECLIPSE_SHOW_BHAC15 else None,
            bhac_age_logt_min             = BHAC_AGE_LOGT_MIN,
            bhac_age_logt_max             = BHAC_AGE_LOGT_MAX,
            bhac_local_delta_t            = BHAC_LOCAL_DELTA_T,
            bhac_local_min_points         = BHAC_LOCAL_MIN_PTS,
            n_draws                       = ECLIPSE_N_DRAWS,
            seed                          = DILUTION_SEED,
            **depth_settings,
        )

    if SED_ONLY:
        if RUN_ECLIPSE_RADIUS_DIAGNOSTIC:
            _eclipse_radius_diagnostic()
        print(f"\n[main_all] SED_ONLY=True — skipping dilution and joint for {label}")
        _write_run_info(star_dir, star, tic_id, gaia_id, mode, "completed", started)
        return res_sed, None, None, None

    # ── Step 2: Dilution fit ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STEP 2 — DILUTION FIT  [{label}]")
    print("=" * 60)

    if MS_RADIUS_PRIOR == "uniform":
        if JOINT_FIT_METHOD == "legacy":
            raise ValueError(
                "MS_RADIUS_PRIOR='uniform' has no BHAC15 lnL_dil table; "
                "set JOINT_FIT_METHOD='ideal'."
            )
        from dilution_uniform_fit import run_dilution_uniform_fit
        print("[MAIN] Dilution method: uniform (no BHAC15 R(T))")
        res_dil = run_dilution_uniform_fit(
            gaia_id                       = gaia_id,
            tic_id                        = tic_id,
            output_base                   = output_base,
            depth_by_filter               = star["depth_by_filter"],
            filter_profile_files          = FILTER_PROFILE_FILES,
            posterior_samples_sed_path    = res_sed["posterior_samples_sed_path"],
            koester_dir                   = KOESTER_DIR,
            da_track_path                 = DA_TRACK,
            db_track_path                 = DB_TRACK,
            wd_atmo_type                  = WD_ATMO_TYPE,
            mode                          = mode,
            t_ms_min                      = T_MS_MIN,
            t_ms_max                      = T_MS_MAX,
            photon_counting               = PHOTON_COUNTING,
            filter_wavelength_cuts        = FILTER_WAVELENGTH_CUTS,
            seed                          = DILUTION_SEED,
            **depth_settings,
        )
    else:
        from dilution_fit import run_dilution_fit
        print(f"[MAIN] Dilution method: {MS_RADIUS_PRIOR} R(T) relation")
        res_dil = run_dilution_fit(
            gaia_id                       = gaia_id,
            tic_id                        = tic_id,
            output_base                   = output_base,
            depth_by_filter               = star["depth_by_filter"],
            filter_profile_files          = FILTER_PROFILE_FILES,
            posterior_samples_sed_path    = res_sed["posterior_samples_sed_path"],
            bhac15_csv                    = BHAC15_CSV,
            koester_dir                   = KOESTER_DIR,
            da_track_path                 = DA_TRACK,
            db_track_path                 = DB_TRACK,
            wd_atmo_type                  = WD_ATMO_TYPE,
            mode                          = mode,
            t_ms_min                      = T_MS_MIN,
            t_ms_max                      = T_MS_MAX,
            t_ms_step                     = T_MS_STEP,
            bhac_local_delta_t            = BHAC_LOCAL_DELTA_T,
            bhac_local_min_points         = BHAC_LOCAL_MIN_PTS,
            bhac_age_logt_min             = BHAC_AGE_LOGT_MIN,
            bhac_age_logt_max             = BHAC_AGE_LOGT_MAX,
            photon_counting               = PHOTON_COUNTING,
            filter_wavelength_cuts        = FILTER_WAVELENGTH_CUTS,
            seed                          = DILUTION_SEED,
            ms_relation                   = ms_relation_prior,
            **depth_settings,
        )

    print(f"\n— DILUTION [{label}] —")
    print(f"  T_MS = {res_dil['T_MS_dil']:.0f} K  "
          f"(-{res_dil['T_MS_dil']-res_dil['T_MS_p16_dil']:.0f}"
          f" /+{res_dil['T_MS_p84_dil']-res_dil['T_MS_dil']:.0f})")
    print(f"  R_MS = {res_dil['R_MS_dil']:.4f} R_sun  "
          f"(-{res_dil['R_MS_dil']-res_dil['R_MS_p16_dil']:.4f}"
          f" /+{res_dil['R_MS_p84_dil']-res_dil['R_MS_dil']:.4f})")
    print(f"  logg_WD = {res_dil['logg_WD_dil']:.3f}  "
          f"(-{res_dil['logg_WD_dil']-res_dil['logg_WD_p16_dil']:.3f}"
          f" /+{res_dil['logg_WD_p84_dil']-res_dil['logg_WD_dil']:.3f})")

    # ── Step 3: Joint fit ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STEP 3 — JOINT FIT  [{label}]")
    print("=" * 60)
    print(f"[MAIN] Joint-fit method: {JOINT_FIT_METHOD}")

    if JOINT_FIT_METHOD == "legacy":
        from joint_fit import run_joint_fit
        res_joint = run_joint_fit(
            gaia_id                    = gaia_id,
            tic_id                     = tic_id,
            output_base                = output_base,
            posterior_samples_sed_path = res_sed["posterior_samples_sed_path"],
            lnL_dil_path               = res_dil["lnL_dil_path"],
            mode                       = mode,
            seed                       = JOINT_SEED,
        )
    else:
        from joint_fit_ideal import run_joint_fit_ideal
        from sed_fit import AV_2C_SIGMA_MIN
        import json as _json
        with open(res_sed["params_sed_path"], encoding="utf-8") as _fp:
            _av1c = _json.load(_fp)["1comp"]["AV_1c_sed"]
        res_joint = run_joint_fit_ideal(
            gaia_id                       = gaia_id,
            tic_id                        = tic_id,
            output_base                   = output_base,
            depth_by_filter               = star["depth_by_filter"],
            filter_profile_files          = FILTER_PROFILE_FILES,
            bhac15_csv                    = BHAC15_CSV,
            koester_dir                   = KOESTER_DIR,
            da_track_path                 = DA_TRACK,
            db_track_path                 = DB_TRACK,
            wd_atmo_type                  = WD_ATMO_TYPE,
            mode                          = mode,
            sed_exclude_windows_um        = star["sed_exclude_windows"],
            sed_exclude_below_um          = star["sed_below_um"],
            sed_exclude_above_um          = star["sed_above_um"],
            sed_max_rel_err               = SED_MAX_REL_ERR_DEFAULT,
            vizier_radius_arcsec          = star["vizier_radius_arcsec"],
            t_ms_min                      = T_MS_MIN,
            t_ms_max                      = T_MS_MAX,
            t_ms_step                     = T_MS_STEP,
            bhac_local_delta_t            = BHAC_LOCAL_DELTA_T,
            bhac_local_min_points         = BHAC_LOCAL_MIN_PTS,
            bhac_age_logt_min             = BHAC_AGE_LOGT_MIN,
            bhac_age_logt_max             = BHAC_AGE_LOGT_MAX,
            photon_counting               = PHOTON_COUNTING,
            filter_wavelength_cuts        = FILTER_WAVELENGTH_CUTS,
            seed                          = JOINT_SEED,
            av_1c_mu                      = float(_av1c["p50"]),
            av_1c_sigma                   = max(
                AV_2C_SIGMA_MIN,
                0.5 * float(_av1c["p84"] - _av1c["p16"]),
            ),
            ms_atmosphere                 = SED_MS_ATMOSPHERE,
            atmo_cache_dir                = SED_ATMO_CACHE_DIR,
            atmo_grid_ranges              = SED_ATMO_GRID_RANGES,
            ms_logg_mode                  = SED_ATMO_MS_LOGG_MODE,
            ms_logg_fixed                 = SED_ATMO_MS_LOGG_FIXED,
            ms_radius_prior               = MS_RADIUS_PRIOR,
            ms_relation                   = ms_relation_prior,
            wd_atmosphere                 = SED_WD_ATMOSPHERE,
            wd_logg_mode                  = SED_ATMO_WD_LOGG_MODE,
            wd_logg_fixed                 = SED_ATMO_WD_LOGG_FIXED,
            wd_cooling_track              = DA_TRACK,
        )

    print(f"\n— JOINT [{label}] —")
    print(f"  T_WD = {res_joint['T_WD_joint']:.0f} K   R_WD = {res_joint['R_WD_joint']:.5f} R_sun")
    print(f"  T_MS = {res_joint['T_MS_joint']:.0f} K   R_MS = {res_joint['R_MS_joint']:.4f} R_sun")
    print(f"  d = {res_joint['d_joint']:.1f} pc   A_V = {res_joint['AV_joint']:.3f} mag")
    print(f"  logZ = {res_joint['logZ_joint']:.3f} ± {res_joint['logZerr_joint']:.3f}")

    # ── Step 4: Eclipse-radius diagnostic ─────────────────────────────────
    if RUN_ECLIPSE_RADIUS_DIAGNOSTIC:
        _eclipse_radius_diagnostic(res_joint["posterior_samples_joint_path"])

    # ── Step 5: Corner plot ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STEP 5 — CORNER PLOT  [{label}]")
    print("=" * 60)

    corner_path = run_display(
        gaia_id                      = gaia_id,
        tic_id                       = tic_id,
        output_base                  = output_base,
        posterior_samples_sed_path   = res_sed["posterior_samples_sed_path"],
        posterior_samples_dil_path   = res_dil["posterior_samples_dil_path"],
        posterior_samples_dil_matched_path = res_dil.get("posterior_samples_dil_matched_path"),
        posterior_samples_joint_path = res_joint["posterior_samples_joint_path"],
        seed                         = JOINT_SEED,
    )
    print(f"  Corner plot : {corner_path}")

    # ── Step 6: Composite figure ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STEP 6 — COMPOSITE FIGURE  [{label}]")
    print("=" * 60)

    try:
        from display import run_composite_figure
        composite_path = run_composite_figure(
            gaia_id        = gaia_id,
            tic_id         = tic_id,
            output_base    = output_base,
            light_curve_pdf = lc_figure,
            corner_side_px = CORNER_SIDE_PX,
            pad_h          = PAD_H,
            pad_v          = PAD_V,
            dpi            = COMPOSITE_DPI,
        )
        print(f"  Composite   : {composite_path}")
    except FileNotFoundError as exc:
        # Only the composite is skipped; every analysis product is kept.
        print(f"  [WARNING] Composite figure skipped: {exc}")
        print(f"            Expected SPECULOOS figure: {lc_figure}")

    # ── Step 7: Masses, core composition, orbital configuration ────────────
    res_mass = None
    if RUN_MASS_ANALYSIS:
        print(f"\n{'='*60}")
        print(f"STEP 7 — MASSES  [{label}]")
        print("=" * 60)

        from mass import run_mass_analysis
        res_mass = run_mass_analysis(
            gaia_id                      = gaia_id,
            tic_id                       = tic_id,
            output_base                  = output_base,
            posterior_samples_joint_path = res_joint["posterior_samples_joint_path"],
            bedard_da_csv                = DA_TRACK,
            laplata_dat                  = LAPLATA_DAT,
            bhac15_csv                   = BHAC15_CSV,
            period_d                     = star.get("period_d"),
            seed                         = MASS_SEED,
            max_samples                  = MASS_MAX_SAMPLES,
            co_grid                      = co_grid,
            he_grid                      = he_grid,
            ms_relation                  = ms_relation,
            wd_mass_track                = WD_MASS_TRACK,
        )

    _write_run_info(star_dir, star, tic_id, gaia_id, mode, "completed", started)
    return res_sed, res_dil, res_joint, res_mass


# ══════════════════════════════════════════════════════════════════════════════
# Companion-model systematics, gathered over all targets
# ══════════════════════════════════════════════════════════════════════════════

def rebuild_summaries(downloads, summaries):
    """Rebuild the two summary tables from every result folder on disk.

    Scans <WORKSPACE>/TIC*/SED-dilution analysis/TIC*_Gaia*/ and reads, for each
    target, sed/params_sed.json (+ systematics_atmosphere_sed.json),
    mass/params_mass.json and run_info.json. The tables therefore always cover
    every analysed target, whichever run produced it. Each row carries
    run_mode (test / final), run_status and run_finished_utc, so that
    diagnostic runs can be told apart from production runs.
    """
    import json
    import re

    def _load(path):
        with open(path, encoding="utf-8") as fp:
            return json.load(fp)

    sed_rows, mass_rows = [], []
    workspace = paths.workspace_dir(downloads)
    pattern = f"TIC*/{paths.ANALYSIS_DIRNAME}/TIC*_Gaia*"
    for star_dir in sorted(workspace.glob(pattern)):
        match = re.fullmatch(r"TIC(\d+)_Gaia(\d+)", star_dir.name)
        if not match or not star_dir.is_dir():
            continue
        tic_id, gaia_id = int(match.group(1)), int(match.group(2))
        info = _load(star_dir / "run_info.json") if (star_dir / "run_info.json").exists() else {}
        meta = {"run_mode": info.get("mode"), "run_status": info.get("status"),
                "run_finished_utc": info.get("finished_utc")}

        p_sed = star_dir / "sed" / "params_sed.json"
        if p_sed.exists():
            params_sed = _load(p_sed)
            compared = (params_sed.get("companion_model_sed") or {}).get("variants_compared", [])
            p_syst = star_dir / "sed" / "systematics_atmosphere_sed.json"
            # A systematics file only belongs to this result if the run compared
            # several companion models (otherwise it is a leftover).
            systematics = _load(p_syst) if (p_syst.exists() and len(compared) > 1) else None
            sed_rows.append((gaia_id, tic_id, {
                "uncertainty_budget_sed": params_sed.get("uncertainty_budget_sed"),
                "systematics_atmosphere_sed": systematics, **meta}))

        p_mass = star_dir / "mass" / "params_mass.json"
        if p_mass.exists():
            params_mass = _load(p_mass)
            skip = {"gaia_id", "tic_id", "models_mass", "relation_systematics_mass"}
            res = {f"{k}_mass": v for k, v in params_mass.items() if k not in skip}
            res.update(gaia_id=gaia_id, tic_id=tic_id, **meta)
            mass_rows.append(res)

    print(f"\n[main_all] Summary tables rebuilt from {len(sed_rows)} target folder(s) "
          f"found under {workspace}")
    n_test = sum(1 for _, _, r in sed_rows if r.get("run_mode") == "test")
    n_unknown = sum(1 for _, _, r in sed_rows if r.get("run_mode") is None)
    if n_test:
        print(f"[main_all] {n_test} of them come from a 'test' run: quick diagnostics, "
              f"not results to quote (see the run_mode column)")
    if n_unknown:
        print(f"[main_all] {n_unknown} of them have no run_info.json (older results): "
              f"their mode is unknown")
    paths.ensure_dir(summaries, "summary folder")
    if sed_rows:
        write_atmosphere_systematics_summary(sed_rows, summaries)
    if mass_rows:
        from mass import write_mass_summary
        write_mass_summary(mass_rows, summaries)
    return sed_rows, mass_rows


def write_atmosphere_systematics_summary(results, output_base):
    """One CSV row per (target, parameter), plus a sample-wide summary.

    This is the table to quote when asked whether the Table-2 uncertainties are
    only the formal nested-sampling ones: it puts the model-dependent shift next
    to the statistical width, for every star and every parameter.
    """
    import csv

    rows = []
    for gaia_id, tic_id, res in results:
        budget = res.get("uncertainty_budget_sed") or {}
        systematics = res.get("systematics_atmosphere_sed") or {}
        shifts = systematics.get("shifts_vs_reference", {})
        if not budget.get("params"):
            continue
        for name, entry in budget["params"].items():
            row = {
                "gaia_id": gaia_id,
                "tic_id": tic_id,
                "run_mode": res.get("run_mode"),
                "run_status": res.get("run_status"),
                "run_finished_utc": res.get("run_finished_utc"),
                "parameter": name,
                "reference": budget.get("reference_variant"),
                "p50": entry["p50"],
                "sigma_stat": entry["sigma_stat"],
                "sigma_syst": entry["sigma_syst"],
                "sigma_total": entry["sigma_total"],
                "syst_over_stat": entry["syst_over_stat"],
            }
            for variant, per_param in shifts.items():
                row[f"delta_{variant}"] = per_param[name]["delta"]
                row[f"delta_over_sigma_{variant}"] = per_param[name]["delta_over_sigma_stat"]
            rows.append(row)

    if not rows:
        print("[main_all] No atmosphere-model comparison to summarise.")
        return None

    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    path = Path(output_base) / "sed_atmosphere_systematics_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[main_all] Atmosphere systematics summary → {path}")

    print("\n[main_all] Median systematic / statistical ratio over the sample:")
    by_param = {}
    for row in rows:
        if row["syst_over_stat"] is not None:
            by_param.setdefault(row["parameter"], []).append(row["syst_over_stat"])
    for name, values in by_param.items():
        values = sorted(values)
        median = values[len(values) // 2]
        print(f"    {name:<12s} median {median:5.2f}   max {values[-1]:5.2f}   "
              f"({len(values)} targets)")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="WD+MS pipeline: SED -> dilution -> joint fit -> figures -> masses. "
                    "Scientific settings are constants at the top of main_all.py.")
    parser.add_argument(
        "--mode", choices=("test", "final"), default=None,
        help="override MODE for this run (test = quick diagnostic, "
             "final = production sampling)")
    parser.add_argument(
        "--example", action="store_true",
        help="run the single example target (EXAMPLE_TARGET) instead of the "
             "TARGETS list")
    parser.add_argument(
        "--check", action="store_true",
        help="only check that the required files and packages are present, "
             "then exit without fitting anything")
    parser.add_argument(
        "--fresh", action="store_true",
        help="before running each target, delete its previous results folder "
             "(<WORKSPACE>/TIC<id>/SED-dilution analysis/TIC<id>_Gaia<gaia>/) so "
             "that only the new results remain; the cached photometry is kept")
    parser.add_argument(
        "--refresh-photometry", action="store_true",
        help="also delete the target's cached VizieR photometry so that it is "
             "downloaded again")
    parser.add_argument(
        "--summaries-only", action="store_true",
        help="only rebuild the summary tables from the results already on disk")
    parser.add_argument(
        "--downloads-dir", default=None, metavar="PATH",
        help="folder holding the per-target data (TIC<id>/) and receiving every "
             "result; default: DOWNLOADS_DIR in main_all.py, else the "
             f"{paths.DOWNLOADS_ENV_VAR} environment variable, else the user's "
             "Downloads folder")
    return parser.parse_args(argv)


def main(argv=None):
    global MODE

    args = _parse_args(argv)
    if args.mode is not None:
        MODE = args.mode
    # Run controls: a command-line option wins, otherwise the constant above.
    args.example            = args.example or RUN_EXAMPLE
    args.check              = args.check or CHECK_ONLY
    args.fresh              = args.fresh or FRESH
    args.refresh_photometry = args.refresh_photometry or REFRESH_PHOTOMETRY
    args.summaries_only     = args.summaries_only or SUMMARIES_ONLY
    targets_all = [EXAMPLE_TARGET] if args.example else TARGETS

    # ── Data / results root ───────────────────────────────────────────────
    try:
        if args.downloads_dir is not None:
            downloads = paths.resolve_downloads_dir(args.downloads_dir, "--downloads-dir")
        else:
            downloads = paths.resolve_downloads_dir(DOWNLOADS_DIR, "DOWNLOADS_DIR in main_all.py")
    except paths.PathSetupError as exc:
        print(f"\n[main_all] ERROR: {exc}")
        raise SystemExit(1)
    summaries = paths.summary_dir(downloads)
    configure_paths(downloads)
    if args.summaries_only:
        rebuild_summaries(downloads, summaries)
        return

    # Skip incomplete entries (gaia_id = None or empty depth_by_filter)
    targets_ready = [
        t for t in targets_all
        if t["gaia_id"] is not None and len(t["depth_by_filter"]) > 0
    ]

    # ── Input checks before any expensive step ────────────────────────────
    errors, notes = check_inputs(targets_ready, downloads)
    print(f"\n[main_all] Code folder       : {PROJECT_ROOT}")
    print(f"[main_all] Models (Koester)  : {MODELS_DIR}")
    print(f"[main_all] Tables            : {TABLES_DIR}")
    print(f"[main_all] Downloads folder  : {downloads}")
    print(f"[main_all] Workspace         : {paths.workspace_dir(downloads)}"
          f"{'' if paths.workspace_dir(downloads).is_dir() else '   (not created yet)'}")
    print(f"[main_all] Atmosphere cache  : {SED_ATMO_CACHE_DIR}")
    for t in targets_ready:
        if t.get("tic_id") is None:
            print(f"[main_all] Gaia {t['gaia_id']}: TIC ID to be looked up on MAST; "
                  f"its folders are reported when the target starts")
            continue
        try:
            print(f"[main_all] {paths.tic_label(t['tic_id'])}: data   {paths.target_dir(downloads, t['tic_id'])}")
            print(f"[main_all] {paths.tic_label(t['tic_id'])}: output {paths.analysis_dir(downloads, t['tic_id'])}")
        except ValueError:
            pass                                   # reported by check_inputs
    print(f"[main_all] Summary CSVs      : {summaries}")
    print("[main_all] Input check:")
    for note in notes:
        print(f"  [note]  {note}")
    for err in errors:
        print(f"  [ERROR] {err}")
    if not targets_ready:
        errors.append("no target to run")
        print("  [ERROR] No target to run: every TARGETS entry is commented out, "
              "has gaia_id = None or an empty depth_by_filter. Fill in TARGETS, "
              "or run the example target (--example or RUN_EXAMPLE = True).")
    if errors:
        print(f"\n[main_all] {len(errors)} problem(s) found — nothing was run. "
              f"Fix them (see docs/DATA_REQUIREMENTS.md) and start again.")
        raise SystemExit(1)
    if args.check:
        print("\n[main_all] Input check passed (--check: nothing was run).")
        return

    try:
        paths.ensure_dir(summaries, "summary folder")
    except paths.PathSetupError as exc:
        print(f"\n[main_all] ERROR: {exc}")
        raise SystemExit(1)

    apply_fit_settings()

    n_total = len(targets_ready)
    n_period = sum(1 for t in targets_ready if t.get("period_d"))
    print(f"\n{'#'*60}")
    print(f"# main_all.py — {n_total}/{len(targets_all)} targets ready — mode={MODE}"
          f"{'   [example run]' if args.example else ''}")
    if MODE == "test":
        print("# MODE='test': quick diagnostic sampling — not a converged result")
    print(f"# companion model: {SED_MS_ATMOSPHERE}   compared with: "
          f"{SED_COMPARE_ATMOSPHERES}")
    print(f"# white dwarf model: {SED_WD_ATMOSPHERE}   "
          f"logg: {SED_ATMO_WD_LOGG_MODE}")
    print(f"# M dwarf radius prior: {MS_RADIUS_PRIOR}   "
          f"mass relation: {MS_MASS_RELATION}")
    print(f"# dilution: {'uniform (no BHAC15 R(T))' if MS_RADIUS_PRIOR == 'uniform' else MS_RADIUS_PRIOR}")
    if SED_ONLY:
        print("# SED_ONLY=True — dilution and joint stages skipped")
    if RUN_MASS_ANALYSIS and not SED_ONLY:
        print(f"# mass stage on, WD track: {WD_MASS_TRACK}   "
              f"orbital configuration for {n_period}/{n_total} "
              f"targets with a period")
    print(f"{'#'*60}\n")

    if MS_RADIUS_PRIOR not in {"uniform", "bhac15", "brown2022"}:
        raise ValueError(
            f"MS_RADIUS_PRIOR must be 'uniform', 'bhac15' or 'brown2022', "
            f"got {MS_RADIUS_PRIOR!r}"
        )
    if MS_MASS_RELATION not in {"bhac15", "brown2022"}:
        raise ValueError(
            f"MS_MASS_RELATION must be 'bhac15' or 'brown2022', "
            f"got {MS_MASS_RELATION!r}"
        )
    if WD_MASS_TRACK not in {"auto", "co"}:
        raise ValueError(
            f"WD_MASS_TRACK must be 'auto' or 'co', got {WD_MASS_TRACK!r}"
        )

    # Built once and shared: each carries a Delaunay triangulation or an
    # interpolation grid that does not depend on the target. ms_relation is the
    # one folded through the posterior at the mass stage; ms_relation_prior is
    # only needed when the fits themselves are asked for inflated radii, and the
    # two are the same object whenever the two settings agree.
    ms_relation = ms_relation_prior = co_grid = he_grid = None
    if RUN_MASS_ANALYSIS or MS_RADIUS_PRIOR == "brown2022":
        from mass import MDwarfRelation, load_bedard_grid, load_laplata_grid
        if RUN_MASS_ANALYSIS:
            ms_relation = MDwarfRelation(BHAC15_CSV, relation=MS_MASS_RELATION)
        if MS_RADIUS_PRIOR == "brown2022":
            ms_relation_prior = (
                ms_relation if MS_MASS_RELATION == "brown2022"
                else MDwarfRelation(BHAC15_CSV, relation="brown2022")
            )
        if RUN_MASS_ANALYSIS and not SED_ONLY:
            co_grid = load_bedard_grid(DA_TRACK)
            he_grid = load_laplata_grid(LAPLATA_DAT, core="He")

    failed = []

    for i, star in enumerate(targets_ready, start=1):
        print(f"\n{'#'*60}")
        print(f"# Target {i}/{n_total}  —  Gaia DR3 {star['gaia_id']}")
        print(f"{'#'*60}")
        try:
            run_star(
                star, mode=MODE,
                ms_relation=ms_relation, ms_relation_prior=ms_relation_prior,
                co_grid=co_grid, he_grid=he_grid, downloads=downloads,
                fresh=args.fresh, refresh_photometry=args.refresh_photometry,
            )
        except Exception as exc:
            print(f"\n[ERROR] Target Gaia {star['gaia_id']} failed: {exc}")
            failed.append(star["gaia_id"])

    # Whole-sample tables, rebuilt from every target folder on disk (not only
    # the targets of this run).
    rebuild_summaries(downloads, summaries)

    print(f"\n{'#'*60}")
    print(f"# Run complete.  {n_total - len(failed)}/{n_total} succeeded.")
    if failed:
        print(f"# Failed targets: {failed}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
