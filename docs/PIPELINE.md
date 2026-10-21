# Pipeline stages, branches and outputs

This page describes what `main_all.py` does for each target, which modules are called under which settings, and what every stage writes. All settings named here are constants at the top of `main_all.py`.

## Order of operations

For every active entry of `TARGETS` (or the single `EXAMPLE_TARGET`, TIC 450781262, with `--example`), `main_all.run_star` runs these steps:

| Step | Module / function | Runs when |
|---|---|---|
| 0. Input check | `main_all.check_inputs` | always, before any target; stops the run if a required file or package is missing |
| TIC lookup | `main_all._fetch_tic_id` (MAST) | `tic_id` is `None` |
| 1. SED fit | `sed_fit.run_sed_fit` | always |
| 2. Dilution fit | `dilution_fit.run_dilution_fit` | `SED_ONLY = False` and `MS_RADIUS_PRIOR` is `"bhac15"` or `"brown2022"` |
| 2. Dilution fit | `dilution_uniform_fit.run_dilution_uniform_fit` | `SED_ONLY = False` and `MS_RADIUS_PRIOR = "uniform"` |
| 3. Joint fit | `joint_fit_ideal.run_joint_fit_ideal` | `SED_ONLY = False` and `JOINT_FIT_METHOD = "ideal"` |
| 3. Joint fit | `joint_fit.run_joint_fit` | `SED_ONLY = False` and `JOINT_FIT_METHOD = "legacy"` |
| 4. Eclipse-radius diagnostic | `dilution_fit.run_eclipse_radius_diagnostic` | `RUN_ECLIPSE_RADIUS_DIAGNOSTIC = True`. After step 1 if `SED_ONLY = True`, otherwise after step 3 with the joint posterior overlaid |
| 5. Corner plot | `display.run_display` | `SED_ONLY = False` |
| 6. Composite figure | `display.run_composite_figure` | `SED_ONLY = False`. Skipped with a warning (expected path printed) when the SPECULOOS figure `<WORKSPACE>/TIC<id>/speculoos_light_curve/TIC<id>_speculoos_light_curve.pdf` is missing |
| 7. Masses and orbit | `mass.run_mass_analysis` | `RUN_MASS_ANALYSIS = True` and `SED_ONLY = False`; the orbital part only when `period_d` is set |

After the loop, `main_all.py` writes two run-wide summaries:

- `sed_atmosphere_systematics_summary.csv`, when more than one companion model was fitted;
- `mass_summary.csv`, when the mass stage ran.

A target that raises an exception is reported as failed, and the loop moves on to the next one.

Combination rules enforced by the code:

- `MS_RADIUS_PRIOR = "uniform"` together with `JOINT_FIT_METHOD = "legacy"` raises an error. The uniform dilution writes no `lnL_dil.npz` table, and the legacy fit needs it.
- `MS_RADIUS_PRIOR`, `MS_MASS_RELATION` and `WD_MASS_TRACK` are checked against their allowed values before the loop.

## Current configuration (as committed)

| Item | Setting | Effect |
|---|---|---|
| Mode | `MODE = "test"` | quick diagnostic sampling |
| SED | `SED_MS_ATMOSPHERE = "phoenix"`, `SED_COMPARE_ATMOSPHERES = ["phoenix", "blackbody"]` | reference SED fit with a PHOENIX companion, plus a monochromatic-blackbody companion refit for the systematic |
| SED, white dwarf | `SED_WD_ATMOSPHERE = "koester"`, `SED_ATMO_WD_LOGG_MODE = "cooling_track"` | Koester DA white dwarf, gravity read off the Bédard DA track |
| Dilution | `MS_RADIUS_PRIOR = "bhac15"` | `dilution_fit.py`, BHAC15 R(T) relation |
| Joint | `JOINT_FIT_METHOD = "ideal"` | `joint_fit_ideal.py`, with a BHAC15 conditional R_MS prior |
| Eclipse-radius diagnostic | `RUN_ECLIPSE_RADIUS_DIAGNOSTIC = False` | not run |
| Masses | `RUN_MASS_ANALYSIS = True`, `MS_MASS_RELATION = "brown2022"`, `WD_MASS_TRACK = "co"` | Brown 2022 M-dwarf relation; CO-core white-dwarf mass adopted |

Under this configuration, `dilution_uniform_fit.py` and `joint_fit.py` are **not used**.

## Stage details

### 1. SED fit (`sed_fit.py`, `sed_atmosphere_models.py`)

- **Data.**
  - Gaia DR3 position and parallax come from the ESA TAP service, queried every run.
  - The VizieR SED photometry is cached as `downloads/<gaia_id>/<gaia_id>_vizier_sed_<UTC>.vot` and reused on later runs.
- **Photometry selection.**
  1. Keep finite, positive fluxes.
  2. Deduplicate (catalogue, filter) pairs.
  3. Apply the user's wavelength windows and cuts (µm).
  4. Points without σ get the relative uncertainty published by their source catalogue, where one can be found (`SIGMA_LOOKUP_CATALOGS`).
  5. Points still without σ are dropped, and so are points with σ/F above `SED_MAX_REL_ERR_DEFAULT` (0.25, passed by `main_all.py`; the same cut is used by the joint SED term).
  6. In the likelihood, σ is floored at `FRAC_FLOOR` × F (18 %).

  The optional outlier clipping is off (`SED_OUTLIER_CLIPPING = False` in `main_all.py`).
- **Models.**
  - The 1-component model (white dwarf only) has parameters T_WD, R_WD, parallax ϖ and A_V.
  - The 2-component model adds T_MS and R_MS.
  - Both use CCM89 + O'Donnell extinction with R_V = 3.1.
  - The parallax prior is Gaussian on ϖ, with d = 1000/ϖ.
- **Priors.**
  - T_WD is uniform on 5 000–100 000 K, clipped to the Koester grid when it is used.
  - R_WD is log-uniform on 0.002–0.08 R☉.
  - T_MS is uniform on 2 300–4 000 K (`T_MS_MIN`, `T_MS_MAX`), the same bound as the dilution and joint stages, for every companion variant. A companion grid that does not cover it narrows it with a warning; the grids may extend further (STScI PHOENIX: 2 300–5 900 K).
  - R_MS is uniform on 0.05–0.80 R☉.
  - A_V is uniform on 0–0.6 mag for the 1-component fit. For the 2-component fit it is weighted by a Gaussian taken from the 1-component posterior.
- **Companion-model systematic.** Every model in `SED_COMPARE_ATMOSPHERES` is refitted on the same photometry with the same priors. The largest shift of each median is reported as `sigma_syst` next to `sigma_stat` in `params_sed.json`.
- **Seeds.** NumPy's global random state is seeded before each UltraNest run with a sub-seed derived from `SED_SEED` and the run name (`sed:1comp:<reference>`, `sed:2comp:<variant>`). The seeds are stored in `params_sed.json` under `seeds_sed`.

Outputs (`sed/`):

| File | Content |
|---|---|
| `posterior_samples_1c_sed.npy` | [N, 4]: T_WD [K], R_WD [R☉], d [pc], A_V [mag] |
| `posterior_samples_sed.npy` | [N, 6]: T_WD, R_WD, T_MS [K], R_MS [R☉], d, A_V. Reference companion model; read by every later stage |
| `posterior_samples_sed_<variant>.npy` | same, for each comparison model |
| `params_sed.json` | p16/p50/p84, evidences, χ², priors, model choices, uncertainty budget |
| `systematics_atmosphere_sed.json` | per-variant results and shifts (when more than one variant) |
| `plot_sed.pdf`, `plot_sed_residuals.pdf`, `plot_sed_atmosphere_comparison.pdf` | figures |
| `ultranest_1c[_<variant>]/`, `ultranest_2c[_<variant>]/` | UltraNest logs (no suffix for the `bb` variant) |

### 2. Dilution fit

Both variants use the same depth model for a total eclipse of the white dwarf:

    depth = 1 / (1 + q),   q = (R_MS / R_WD)² · F_MS,band / F_WD,band

Every band flux and gravity comes from the shared depth model, `depth_model.DepthFluxModel`, which is also used by the joint fit and the eclipse diagnostic:

- **F_WD,band:** the local Koester DA spectra, integrated at native resolution over the filter, then interpolated bilinearly in ln F over the (T_WD, log g_WD) nodes.
- **log g_WD:** the SED policy. The Bédard cooling track is interpolated with a Delaunay triangulation; outside the track the fallback `SED_ATMO_WD_LOGG_FIXED` (7.75) is used; the result is clipped to the Koester log g range.
- **F_MS,band:** the SED reference companion model (the STScI PHOENIX grid used by the SED fit, same cached file), interpolated bilinearly in ln F over (T_MS, log g_MS).
- **log g_MS:** the SED prescription (`SED_ATMO_MS_LOGG_MODE`; currently the empirical main-sequence log g(T_MS)), clipped to the grid.
- **Integration:** over the SPECULOOS filter profiles with the cuts in `FILTER_WAVELENGTH_CUTS`, photon-weighted when `PHOTON_COUNTING = True`, without reddening.
- **No extrapolation:** temperatures outside a grid give no flux, and the sample is rejected.
- T_WD and R_WD are drawn from the SED posterior.

**`dilution_fit.py` (BHAC15 / Brown 2022).**

- T_MS is tabulated on a grid with a 100 K step. At each grid point, R_MS is the median of the BHAC15 radii within ±100 K, with an age cut of log t ≥ 8. `brown2022` rescales these radii and widens them by the relation's scatter.
- Monte Carlo draws pick T_MS from the depth likelihood along that relation, then refine it continuously.
- R_MS is then drawn around the relation's median at that T_MS from a median-centred, two-sided half-normal distribution: with probability 0.5 below the median with width r50 − r16, otherwise above it with width r84 − r50 (`dilution_fit.draw_two_sided_half_normal`).
- It also tabulates `lnL_dil.npz` on a (T_WD, R_WD, T_MS) grid for the legacy joint fit.

**`dilution_uniform_fit.py` (uniform).**

- T_MS is drawn uniformly. At each draw, R_MS is the radius that reproduces the depth, with the companion band flux read from PHOENIX at log g = `SED_ATMO_MS_LOGG_FIXED`.
- No relation and no `lnL_dil.npz`.

Outputs (`dilution/`):

| File | Content |
|---|---|
| `posterior_samples_dil.npy` | [N, 2]: T_MS [K], R_MS [R☉] |
| `posterior_samples_dil_matched.npy` | [N, 4]: T_WD [K], R_WD [R☉], T_MS [K], R_MS [R☉]; the same rows, with the white dwarf each sample used |
| `params_dil.json` | p16/p50/p84 for T_MS, R_MS, log g_WD; depth summary |
| `plot_dil.pdf` | figure |
| `lnL_dil.npz` | BHAC15 branch only |

### 3. Joint fit

**Ideal (`joint_fit_ideal.py`).**

One UltraNest run over T_WD, R_WD, T_MS, R_MS, ϖ and A_V. Its log-likelihood is the sum of:

- the raw-photometry SED term (same selection, σ/F cut 0.25 and models as step 1);
- the eclipse-depth term (the shared depth model of step 2). log g_WD and log g_MS are computed once per sample and passed to both the SED term and the depth term;
- the Gaia parallax term;
- the A_V Gaussian taken from the 1-component SED posterior.

The R_MS prior follows `MS_RADIUS_PRIOR`. The live points move with a slice sampler. The run is seeded with a sub-seed of `JOINT_SEED`, stored in `params_joint_ideal.json` (`seed_joint_ideal`); the legacy fit records its seed in `params_joint.json` (`seed_joint`).

Outputs (`joint_ideal/`):

| File | Content |
|---|---|
| `posterior_samples_joint_ideal.npy` | [N, 7]: T_WD, R_WD, T_MS, R_MS, d [pc], A_V, ϖ [mas] |
| `params_joint_ideal.json` | summary |
| `plot_joint_ideal.pdf` | figure |
| `ultranest_joint_ideal/` | UltraNest logs, overwritten every run |

**Legacy (`joint_fit.py`).**

- The likelihood is ln KDE_SED(θ) + lnL_dil(T_WD, R_WD, T_MS), where the dilution table is interpolated linearly.
- The bounds are flat between the 0.5 and 99.5 percentiles of the SED posterior, widened by 2 %.

Outputs (`joint/`): `posterior_samples_joint.npy` [N, 6], `params_joint.json`, `plot_joint.pdf`.

### 4. Eclipse-radius diagnostic (optional)

`dilution_fit.run_eclipse_radius_diagnostic` inverts the depth equation into the companion radius it requires, R_MS(T_MS), with no radius–temperature relation. It overlays that band on the SED posterior. The reference band uses the shared depth model and gravity prescription; the fixed gravities in `ECLIPSE_MS_LOGG_VALUES` are scanned on top of it to show the residual gravity dependence.

Outputs go to `eclipse_radius/`: `eclipse_radius_curves.npz`, `params_eclipse_radius.json` and `plot_eclipse_radius.pdf`.

### 5–6. Figures (`display.py`)

- `plot_corner.pdf` shows three panels: T_WD–R_WD, T_WD–T_MS and R_MS–T_MS. It overlays the SED posterior (blue), the dilution posterior (orange-brown) and the joint posterior (black contours). The dilution posterior is drawn with the (T_WD, R_WD) each sample was computed with (`posterior_samples_dil_matched.npy`). Older outputs without that file show it in the R_MS–T_MS panel only.
- `plot_composite.pdf` stacks the external light-curve PDF above `sed/plot_sed.pdf`, then places the corner plot on the right. Rendering the PDFs needs PyMuPDF, or pdf2image with poppler.

### 7. Masses and orbit (`mass.py`)

**White-dwarf mass.**

- It is interpolated from the Bédard 2020 DA sequences (CO, thick H layer) and from the La Plata table (He cores, Althaus et al. 2013), at each joint-posterior (T_WD, R_WD).
- Two core labels are reported: one from the Brown et al. decision rule, one from posterior fractions.
- `WD_MASS_TRACK = "co"` always adopts the CO mass; `"auto"` follows the Brown rule.

**M-dwarf mass.**

- It is obtained by inverting `MS_MASS_RELATION`: Brown et al. 2022 above 0.121 M☉, or BHAC15 at 1 Gyr.
- Relation floors are added in quadrature: 5 % on M_MS and 1 % on M_WD.

**Orbit** (needs `period_d`).

- The separation comes from Kepler's third law.
- The Roche lobes come from Eggleton 1983.
- The RV semi-amplitudes assume i = 90°.

Outputs (`mass/`):

| File | Content |
|---|---|
| `posterior_samples_mass.npy` | [N, 5]: M_WD(CO), M_WD(He), M_MS [M☉], a [R☉], R_MS/R_L |
| `params_mass.json` | summary |
| `plot_mass.pdf` | figure |

## Complete output tree

The per-target output base is `<WORKSPACE>/TIC<id>/SED-dilution analysis/`, with `<WORKSPACE>` = `<DOWNLOADS>/WDMS_fitting_pipeline/` (see the README for how `<DOWNLOADS>` is chosen). Every stage of a target receives this same folder.

```
<WORKSPACE>/TIC<id>/SED-dilution analysis/       per-target output base
├── downloads/<gaia_id>/            VizieR SED VOTables (cache, reused)
└── TIC<tic>_Gaia<gaia>/
    ├── sed/
    ├── dilution/
    ├── joint_ideal/   or  joint/
    ├── eclipse_radius/             only with RUN_ECLIPSE_RADIUS_DIAGNOSTIC
    ├── mass/
    ├── plot_corner.pdf
    └── plot_composite.pdf          only if the SPECULOOS figure exists

<WORKSPACE>/SED-dilution summaries/              whole-sample CSVs of a normal run
├── sed_atmosphere_systematics_summary.csv
└── mass_summary.csv
```


## Settings reference

Every setting is a constant near the top of `2_sed_dilution/main_all.py`, with a comment explaining its options. The "Fit settings" section (priors, σ floor, outlier clipping, sampling, mass floors) is copied into the modules by `apply_fit_settings()` before the first target; the modules keep the same values as defaults when they are used alone. Every target's `run_info.json` records the values used.

| Setting | Current value | Meaning |
|---|---|---|
| `MODE` | `"test"` | sampling depth (see below); `--mode` overrides it |
| `DOWNLOADS_DIR` | `None` (automatic) | Downloads folder containing the workspace; `--downloads-dir` overrides it |
| `MODELS_DIR` | `None` → `<WORKSPACE>/models/` (or `$WDMS_MODELS_DIR`) | Koester DA spectra (`koester2/`) |
| `TABLES_DIR` | `None` → `2_sed_dilution/tables/` | Bédard, BHAC15 and La Plata tables (included) |
| `SED_ATMO_CACHE_DIR` | `None` → `<WORKSPACE>/model_cache/` (or `$SED_MODEL_CACHE`) | atmosphere-grid and filter-curve cache |
| `SED_MS_ATMOSPHERE` | `"phoenix"` | companion model of the reference SED fit and of every depth calculation (`blackbody`, `bb-int`, `phoenix`, `btsettl`) |
| `SED_COMPARE_ATMOSPHERES` | `["phoenix", "blackbody"]` | extra companion models refitted to measure a systematic (`None` disables) |
| `SED_WD_ATMOSPHERE` | `"koester"` | white-dwarf model in the SED term (`blackbody` or `koester`) |
| `SED_ATMO_WD_LOGG_MODE` | `"cooling_track"` | white-dwarf gravity (`cooling_track` or `fixed`) |
| `MS_RADIUS_PRIOR` | `"bhac15"` | companion radius prior: `uniform`, `bhac15` or `brown2022` (also chooses the dilution module) |
| `JOINT_FIT_METHOD` | `"ideal"` | `ideal` or `legacy` |
| `MS_MASS_RELATION` | `"brown2022"` | M-dwarf mass–radius relation of the mass stage |
| `WD_MASS_TRACK` | `"co"` | adopted white-dwarf mass: `co` (CO-core tracks) or `auto` (Brown et al. CO/He rule) |
| `WD_ATMO_TYPE` | `"DA"` | only DA is supported consistently (see KNOWN_ISSUES) |
| `T_MS_MIN`, `T_MS_MAX` | 2300, 4000 K | companion temperature bound of the SED, dilution and ideal joint fits |
| `SED_MAX_REL_ERR_DEFAULT` | 0.25 | photometric σ/F cut of the SED fit and of the joint SED term |
| `FRAC_FLOOR` | 0.18 | σ floor as a fraction of the flux, in the SED fit and the joint SED term |
| `SED_OUTLIER_CLIPPING` | `False` | optional SED outlier clipping; with `SED_OUTLIER_ALL_POINTS`, `SED_MAD_SIGMA` (3.0), `SED_MAX_REJECT_FRAC` (0.2), `SED_OUTLIER_MAX_ITERS` (1), `SED_REJECT_BOTH` (`True`) |
| `T_WD_MIN`, `T_WD_MAX` | 5 000, 100 000 K | white-dwarf temperature prior (SED and ideal joint fits) |
| `R_WD_MIN_RSUN`, `R_WD_MAX_RSUN` | 0.002, 0.08 R☉ | white-dwarf radius prior, log-uniform |
| `R_MS_MIN_RSUN`, `R_MS_MAX_RSUN` | 0.05, 0.80 R☉ | companion radius bounds (SED, uniform dilution, ideal joint fit) |
| `AV_MIN`, `AV_MAX` | 0.0, 0.6 mag | extinction prior |
| `SED_SAMPLING`, `DILUTION_SAMPLING`, `JOINT_IDEAL_SAMPLING`, `JOINT_LEGACY_N_LIVE` | see below | sampling of each stage, per `MODE` |
| `MS_MASS_REL_SYST`, `WD_MASS_REL_SYST` | 0.05, 0.01 | fractional mass floors added in quadrature (mass stage) |
| `SED_SEED`, `DILUTION_SEED`, `JOINT_SEED` | 42 | seeds of the SED, dilution and joint stages |
| `SED_ONLY` | `False` | stop after the SED fit of every star |
| `RUN_ECLIPSE_RADIUS_DIAGNOSTIC` | `False` | optional step 4 |
| `RUN_MASS_ANALYSIS` | `True` | step 7 |
| `RUN_EXAMPLE`, `CHECK_ONLY`, `FRESH`, `REFRESH_PHOTOMETRY`, `SUMMARIES_ONLY` | `False` | run controls, same as `--example`, `--check`, `--fresh`, `--refresh-photometry`, `--summaries-only` (the command line wins) |
| `TARGETS` / `EXAMPLE_TARGET` | empty / TIC 450781262 | per-target Gaia ID, TIC ID, depths per filter, SED cuts, VizieR radius, period; optional `speculoos_figure` and `speculoos_csv_files`. The comment above `TARGETS` explains every line |

## Sampling modes

| | `test` | `final` |
|---|---|---|
| SED fit (UltraNest) | 16 live points, at most 20 000 iterations | 500 live points, `dlogz = 0.2` |
| Dilution (Monte Carlo) | 5 000 draws, 10 × 10 × n(T_MS) likelihood grid | 50 000 draws, 30 × 30 × n(T_MS) grid |
| Joint fits (UltraNest) | 16 live points (ideal fit: at most 20 000 iterations) | 400 live points (ideal fit: `dlogz = 0.2`) |

These values are the `SED_SAMPLING`, `DILUTION_SAMPLING`, `JOINT_IDEAL_SAMPLING` and `JOINT_LEGACY_N_LIVE` tables of `main_all.py`.

- **`test`** is a quick diagnostic: it shows that the inputs load, the stages connect and the plots are written. Its posteriors are under-sampled and must not be quoted.
- **Even in `test`:** the model grids are loaded in full, the network queries still run, and first-run downloads still happen.
- **Seeds:** every UltraNest run is seeded (sub-seeds of `SED_SEED` and `JOINT_SEED`, written to the `params_*.json` files). Results can still differ across library versions, platforms, or resumed runs.

## Folders and files

**The Downloads folder `<DOWNLOADS>`** is chosen in this order:

1. `--downloads-dir`;
2. `DOWNLOADS_DIR` in `main_all.py`;
3. the environment variable `WDMS_DOWNLOADS_DIR`;
4. the system Downloads folder:
   - the registry entry on Windows;
   - `XDG_DOWNLOAD_DIR` on Linux;
   - otherwise `~/Downloads`, then `~/Téléchargements`.

An explicit folder (1–3) must exist; otherwise the run stops instead of using another location.

**Other conventions:**

- **TIC identifiers:** `450781262` and `"TIC450781262"` give the same folder.
- **SPECULOOS figure:** `<WORKSPACE>/TIC<id>/speculoos_light_curve/TIC<id>_speculoos_light_curve.pdf`. The `speculoos_figure` key of a target overrides it. If the figure is missing, only the composite is skipped, with a warning naming the expected path.
- **SPECULOOS CSVs:** `<WORKSPACE>/TIC<id>/speculoos_csv/TIC<id>_<band>.csv`. The `speculoos_csv_files` key declares other names. Part 2 lists them but does not read them: its depths come from `depth_by_filter`.

**Running again:**

- **Reuse:** a rerun never reuses earlier results. Each stage reads the files the previous stage of the same run has just written.
- **`--fresh`:** deletes `<WORKSPACE>/TIC<id>/SED-dilution analysis/TIC<id>_Gaia<gaia>/` before the run. Without it, files that the new run does not write again (e.g. a composite figure that is now skipped, or the `eclipse_radius/` folder after the diagnostic is switched off) stay next to the new ones.
- **`--refresh-photometry`:** also deletes `SED-dilution analysis/downloads/<gaia_id>/`.
- **Overwritten:** posterior samples, JSON files, figures, summary CSVs and the joint-fit UltraNest folders.
- **Kept:**
  - each SED fit adds a `run<N>/` folder;
  - the VizieR photometry cache (`SED-dilution analysis/downloads/`) and the model caches are reused.
- **Never written by Part 2:** the notebook folders.

**Summary CSVs:**

- They live in `<WORKSPACE>/SED-dilution summaries/`.
- They are rebuilt at the end of every run (and by `--summaries-only` / `SUMMARIES_ONLY = True`) from every `TIC*/SED-dilution analysis/TIC*_Gaia*/` folder on disk:
  - `sed/params_sed.json` (+ `systematics_atmosphere_sed.json` when several companion models were compared);
  - `mass/params_mass.json`;
  - `run_info.json`.
- `run_info.json` is written by each run: mode, status (`running` until the run completes), start and end times, main settings, target entry. Its fields fill the `run_mode`, `run_status` and `run_finished_utc` columns. Results older than this file show an empty mode.
- A target analysed several times appears once, with its latest results. Use `--fresh` so that a target's folder holds the files of a single run.
