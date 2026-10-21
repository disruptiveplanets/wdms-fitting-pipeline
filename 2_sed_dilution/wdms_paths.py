"""wdms_paths.py — Where the pipeline reads its inputs and writes its results.

Two locations are kept apart:

* the REPOSITORY: source code and repository resources. This file lives in
  <repository>/2_sed_dilution/ (CODE_DIR), next to the other modules, the
  SPECULOOS filter profiles (filters/) and the small model tables (tables/:
  Bédard, BHAC15, La Plata). Always resolved from this file's location, never
  from the terminal's current working directory, so the repository can be
  downloaded or cloned anywhere.

* the WORKSPACE, a folder named WDMS_fitting_pipeline inside the user's
  Downloads folder, holding the models, the caches, the per-target data and
  every generated result:

      <DOWNLOADS>/WDMS_fitting_pipeline/
          models/                                   ← Koester DA spectra (koester2/)
          model_cache/                              ← PHOENIX, BT-Settl, SVO filter curves
          TIC<id>/
              tess_cutouts/  light_curve/  speculoos_csv/
              speculoos_light_curve/  environment_maps/   ← notebook products
              SED-dilution analysis/                       ← this pipeline
          SED-dilution summaries/                   ← whole-sample CSVs

The Downloads folder is resolved by resolve_downloads_dir(); see its
docstring for the order of the candidates. Only the standard library is used,
so main_all.py can import this module before any dependency check.
"""

import os
import re
import sys
from pathlib import Path

# ── Repository ────────────────────────────────────────────────────────────────
CODE_DIR    = Path(__file__).resolve().parent      # <repository>/2_sed_dilution/
REPO_ROOT   = CODE_DIR.parent                       # <repository>/
FILTERS_DIR = CODE_DIR / "filters"                  # SPECULOOS filter profiles
TABLES_DIR  = CODE_DIR / "tables"                   # Bédard, BHAC15, La Plata tables

# ── Folder names ──────────────────────────────────────────────────────────────
# Workspace inside the Downloads folder; everything below lives in it.
WORKSPACE_DIRNAME = "WDMS_fitting_pipeline"
MODELS_DIRNAME = "models"
CACHE_DIRNAME = "model_cache"
MODELS_ENV_VAR = "WDMS_MODELS_DIR"        # overrides <workspace>/models
CACHE_ENV_VAR = "SED_MODEL_CACHE"         # overrides <workspace>/model_cache
# Folders created and filled by the notebooks; this pipeline never writes there.
NOTEBOOK_SUBDIRS = ("tess_cutouts", "light_curve", "speculoos_csv",
                    "speculoos_light_curve", "environment_maps")
SPECULOOS_CSV_DIRNAME = "speculoos_csv"
SPECULOOS_FIGURE_DIRNAME = "speculoos_light_curve"
# Per-target output base of this pipeline (inside <workspace>/TIC<id>/).
ANALYSIS_DIRNAME = "SED-dilution analysis"
# Whole-sample summary CSVs (directly inside <workspace>/).
SUMMARY_DIRNAME = "SED-dilution summaries"

DOWNLOADS_ENV_VAR = "WDMS_DOWNLOADS_DIR"


class PathSetupError(RuntimeError):
    """A required folder could not be found or created (message is user-facing)."""


# ══════════════════════════════════════════════════════════════════════════════
# TIC identifiers
# ══════════════════════════════════════════════════════════════════════════════

_TIC_RE = re.compile(r"^\s*(?:tic)?[\s_\-:]*0*(\d+)\s*$", re.IGNORECASE)


def normalize_tic(value):
    """Return the TIC identifier as an int.

    Accepts 450781262, "450781262", "TIC450781262", "TIC 450781262",
    "tic_450781262" … so that every spelling maps to the same folder and a
    "TIC" prefix is never doubled.
    """
    if isinstance(value, bool):
        raise ValueError(f"Invalid TIC identifier: {value!r}")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"Invalid TIC identifier: {value!r}")
        return value
    match = _TIC_RE.match(str(value))
    if not match or int(match.group(1)) <= 0:
        raise ValueError(f"Invalid TIC identifier: {value!r} (expected e.g. 450781262 "
                         f"or 'TIC450781262')")
    return int(match.group(1))


def tic_label(value):
    """'TIC<id>' for any accepted spelling of the identifier."""
    return f"TIC{normalize_tic(value)}"


# ══════════════════════════════════════════════════════════════════════════════
# Downloads folder
# ══════════════════════════════════════════════════════════════════════════════

def _xdg_download_dir():
    """XDG_DOWNLOAD_DIR from ~/.config/user-dirs.dirs (Linux desktops), or None."""
    cfg = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "user-dirs.dirs"
    try:
        text = cfg.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r'^XDG_DOWNLOAD_DIR\s*=\s*"?([^"\n]+)"?', text, re.MULTILINE)
    if not match:
        return None
    return Path(match.group(1).replace("$HOME", str(Path.home())))


def _windows_download_dir():
    """The Downloads known folder from the Windows registry, or None."""
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders")
        value, _ = winreg.QueryValueEx(key, "{374DE290-123F-4565-9164-39C4925E467B}")
        return Path(os.path.expandvars(value))
    except (ImportError, OSError):
        return None


def downloads_candidates():
    """Default Downloads candidates, in the order they are tried."""
    home = Path.home()
    candidates = []
    if sys.platform.startswith("win"):
        candidates.append(_windows_download_dir())
    elif sys.platform.startswith("linux"):
        candidates.append(_xdg_download_dir())
    # The standard folder, then the French-localised name used by the notebooks.
    candidates += [home / "Downloads", home / "Téléchargements"]
    return [c for c in candidates if c is not None]


def resolve_downloads_dir(override=None, source="--downloads-dir"):
    """Absolute path of the Downloads folder that contains the workspace.

    The workspace itself is <DOWNLOADS>/WDMS_fitting_pipeline/ (workspace_dir).

    Order:
      1. `override` (the --downloads-dir option, or DOWNLOADS_DIR in
         main_all.py), used as given; `source` names it in error messages;
      2. the environment variable WDMS_DOWNLOADS_DIR;
      3. the platform's Downloads folder: the registry entry on Windows, the
         XDG user directory on Linux, then ~/Downloads, then ~/Téléchargements.

    An explicit choice (1 or 2) must be an existing folder: it is never
    replaced silently by another location. A PathSetupError explains what to do
    when no folder is found.
    """
    explicit = override if override is not None else os.environ.get(DOWNLOADS_ENV_VAR)
    if explicit:
        path = Path(os.path.expanduser(str(explicit))).resolve()
        source = source if override is not None else DOWNLOADS_ENV_VAR
        if not path.is_dir():
            raise PathSetupError(
                f"The Downloads folder given by {source} does not exist or is not a "
                f"folder:\n    {path}\nCreate it, or pass the correct path, e.g.\n"
                f'    python main_all.py --downloads-dir "/path/to/my/downloads"')
        return path
    for candidate in downloads_candidates():
        if candidate.is_dir():
            return candidate.resolve()
    tried = "\n    ".join(str(c) for c in downloads_candidates())
    raise PathSetupError(
        f"No Downloads folder found. Tried:\n    {tried}\nPass the folder explicitly, e.g.\n"
        f'    python main_all.py --downloads-dir "/path/to/my/downloads"\n'
        f"or set the environment variable {DOWNLOADS_ENV_VAR}.")


# ══════════════════════════════════════════════════════════════════════════════
# Per-target and summary locations
# ══════════════════════════════════════════════════════════════════════════════

def workspace_dir(downloads):
    """<DOWNLOADS>/WDMS_fitting_pipeline/ — models, caches, targets, results."""
    return Path(downloads) / WORKSPACE_DIRNAME


def models_dir(downloads):
    """Local model files: $WDMS_MODELS_DIR if set, else <workspace>/models/."""
    env = os.environ.get(MODELS_ENV_VAR)
    if env:
        return Path(os.path.expanduser(env)).resolve()
    return workspace_dir(downloads) / MODELS_DIRNAME


def atmosphere_cache_dir(downloads):
    """Atmosphere-grid cache: $SED_MODEL_CACHE if set, else <workspace>/model_cache/."""
    env = os.environ.get(CACHE_ENV_VAR)
    if env:
        return Path(os.path.expanduser(env)).resolve()
    return workspace_dir(downloads) / CACHE_DIRNAME


def target_dir(downloads, tic):
    """<workspace>/TIC<id>/ — the target's data folder (notebook products)."""
    return workspace_dir(downloads) / tic_label(tic)


def analysis_dir(downloads, tic):
    """<workspace>/TIC<id>/SED-dilution analysis/ — the pipeline output base."""
    return target_dir(downloads, tic) / ANALYSIS_DIRNAME


def summary_dir(downloads):
    """<workspace>/SED-dilution summaries/ — tables over every analysed target."""
    return workspace_dir(downloads) / SUMMARY_DIRNAME


def speculoos_csv_dir(downloads, tic):
    return target_dir(downloads, tic) / SPECULOOS_CSV_DIRNAME


def speculoos_figure_path(downloads, tic, override=None):
    """SPECULOOS light-curve figure used as the composite's light-curve panel.

    Default: <workspace>/TIC<id>/speculoos_light_curve/TIC<id>_speculoos_light_curve.pdf,
    the file the notebook saves. `override` (absolute, or relative to the
    speculoos_light_curve/ folder) replaces it.
    """
    folder = target_dir(downloads, tic) / SPECULOOS_FIGURE_DIRNAME
    if override:
        p = Path(os.path.expanduser(str(override)))
        return p if p.is_absolute() else folder / p
    return folder / f"{tic_label(tic)}_speculoos_light_curve.pdf"


def speculoos_csv_paths(downloads, tic, bands, overrides=None):
    """{band: Path} of the SPECULOOS observation CSVs of one target.

    Default name per band: TIC<id>_<band>.csv in speculoos_csv/ (the notebook
    convention). `overrides` maps a band to another file name (relative to
    speculoos_csv/) or to an absolute path, for observations named differently.
    """
    folder = speculoos_csv_dir(downloads, tic)
    overrides = overrides or {}
    out = {}
    for band in bands:
        if band in overrides and overrides[band]:
            p = Path(os.path.expanduser(str(overrides[band])))
            out[band] = p if p.is_absolute() else folder / p
        else:
            out[band] = folder / f"{tic_label(tic)}_{band}.csv"
    for band, value in overrides.items():          # override-only bands
        if band not in out and value:
            p = Path(os.path.expanduser(str(value)))
            out[band] = p if p.is_absolute() else folder / p
    return out


def ensure_dir(path, what):
    """Create `path` (and parents) if needed; explain clearly if impossible."""
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PathSetupError(f"Cannot create the {what}:\n    {path}\n    ({exc})") from exc
    if not os.access(path, os.W_OK):
        raise PathSetupError(f"The {what} is not writable:\n    {path}")
    return path
