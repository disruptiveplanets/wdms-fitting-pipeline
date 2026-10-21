#!/usr/bin/env python3
"""display.py — Corner-plot overlay of SED, dilution, and joint posteriors.

Corner layout (2 × 2, top-right hidden):

  ┌──────────────┬──────────┐
  │ T_WD vs R_WD │  (hidden)│
  ├──────────────┼──────────┤
  │ T_WD vs T_MS │ R_MS vs  │
  │              │   T_MS   │
  └──────────────┴──────────┘

Three overlaid datasets (colour-coded):
  • SED posterior         — tab:blue  — posterior_samples_sed.npy   [N, 6]
  • Dilution posterior    — peru      — posterior_samples_dil_matched.npy
                                        [N, 4] (T_WD, R_WD, T_MS, R_MS), or
                                        posterior_samples_dil.npy [N, 2]
  • Joint fit posterior   — black     — posterior_samples_joint.npy [N, 6]
                                        (legacy) or
                                        posterior_samples_joint_ideal.npy [N, 7]
Only the first four columns (T_WD, R_WD, T_MS, R_MS) are read.

The joint posterior is drawn with black contours on top of everything else:
it is the result the figure is about, and blue / orange-brown / black stay
distinguishable under red-green colour blindness. Every contour sits above
every scatter cloud, so no layer can bury another.

The scatter clouds are rasterized while contours, axes and labels stay
vector: the PDF keeps sharp text but loads and prints in one pass.

Column convention for 6-column arrays: T_WD, R_WD, T_MS, R_MS, d, A_V
Column convention for 2-column arrays: T_MS, R_MS
Column convention for 4-column arrays (matched dilution): T_WD, R_WD, T_MS, R_MS

The dilution posterior is drawn with the (T_WD, R_WD) each of its samples was
computed with (posterior_samples_dil_matched.npy). Older outputs without that
file show the dilution posterior in the R_MS–T_MS panel only.

Output: plot_corner.pdf  in  <output_base>/TIC{id}_Gaia{id}/

run_composite_figure then assembles the light-curve panel (an external PDF,
LC_SPEC/TIC{tic_id}.pdf, not produced by this repository), the SED plot and the
corner plot into plot_composite.pdf. The README example image is a figure of
this kind.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.stats import gaussian_kde


# ── Visual constants ──────────────────────────────────────────────────────────
_CONTOUR_SIGMA   = (1.0, 2.0)
_SIGMA_TO_MASS   = {1.0: 0.3935, 2.0: 0.8647, 3.0: 0.9889}
_KDE_GRID        = 120
_SCATTER_ALPHA   = 0.08
_SCATTER_SIZE    = 3
_MAX_SCATTER     = 20_000
_CONTOUR_LW      = 1.1
_CONTOUR_LW_TOP  = 1.6     # joint posterior, drawn above the other two
_RASTER_DPI      = 300     # resolution of the rasterized scatter layers


# ══════════════════════════════════════════════════════════════════════════════
# KDE + scatter helper
# ══════════════════════════════════════════════════════════════════════════════

def _kde_contour(ax, x, y, color, levels_sigma=_CONTOUR_SIGMA, grid=_KDE_GRID,
                 rng=None, contour_color=None, linewidth=_CONTOUR_LW,
                 scatter_zorder=1, contour_zorder=10):
    # The scatter clouds carry tens of thousands of markers each: rasterizing
    # them keeps the PDF light while the contours stay vector.
    if contour_color is None:
        contour_color = color

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 10:
        ax.scatter(x, y, s=_SCATTER_SIZE, alpha=0.5, color=color,
                   rasterized=True, zorder=scatter_zorder)
        return

    if rng is None:
        rng = np.random.default_rng(0)
    if x.size > _MAX_SCATTER:
        idx = rng.choice(x.size, size=_MAX_SCATTER, replace=False)
        x_s, y_s = x[idx], y[idx]
    else:
        x_s, y_s = x, y
    ax.scatter(x_s, y_s, s=_SCATTER_SIZE, alpha=_SCATTER_ALPHA, color=color,
               rasterized=True, zorder=scatter_zorder)

    try:
        kde = gaussian_kde(np.vstack([x, y]))
    except Exception as exc:
        print(f"[DISPLAY] KDE skipped: {exc}")
        return

    dx = 0.02 * max(x.max() - x.min(), 1e-12)
    dy = 0.02 * max(y.max() - y.min(), 1e-12)
    X, Y = np.meshgrid(
        np.linspace(x.min() - dx, x.max() + dx, grid),
        np.linspace(y.min() - dy, y.max() + dy, grid),
    )
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    Zflat = np.sort(Z.ravel())[::-1]
    Zcum  = np.cumsum(Zflat)
    Zcum /= Zcum[-1]

    levels = []
    for s in sorted(levels_sigma, reverse=True):
        target = _SIGMA_TO_MASS.get(float(s), 1.0 - np.exp(-0.5 * s * s))
        idx    = int(np.searchsorted(Zcum, target))
        idx    = min(max(idx, 0), len(Zflat) - 1)
        levels.append(float(Zflat[idx]))

    levels = sorted(set(levels))
    if levels:
        ax.contour(X, Y, Z, levels=levels, colors=contour_color,
                   linewidths=linewidth, zorder=contour_zorder)


# ══════════════════════════════════════════════════════════════════════════════
# Corner plot builder
# ══════════════════════════════════════════════════════════════════════════════

def _make_corner(datasets, rng):
    """
    datasets : list of dict with keys
        label, color, T_WD, R_WD, T_MS, R_MS
        and optionally contour_color, linewidth, zorder (higher = drawn last)
    """
    fig, axes = plt.subplots(
        2, 2, figsize=(7.8, 7.8),
        sharex="col", sharey="row",
        gridspec_kw={"wspace": 0.08, "hspace": 0.08},
    )
    ax_tl   = axes[0, 0]   # T_WD vs R_WD
    ax_hide = axes[0, 1]   # hidden
    ax_bl   = axes[1, 0]   # T_WD vs T_MS
    ax_br   = axes[1, 1]   # R_MS vs T_MS

    ax_hide.set_visible(False)
    for ax in (ax_tl, ax_bl, ax_br):
        ax.set_axisbelow(True)

    # Every contour is placed above every scatter cloud, and the datasets are
    # stacked by increasing "zorder" so the joint posterior ends up on top.
    for rank, ds in enumerate(sorted(datasets, key=lambda d: d.get("zorder", 0))):
        style = dict(
            color          = ds["color"],
            contour_color  = ds.get("contour_color", ds["color"]),
            linewidth      = ds.get("linewidth", _CONTOUR_LW),
            scatter_zorder = 1 + rank,
            contour_zorder = 10 + rank,
            rng            = rng,
        )
        # A dataset without white-dwarf samples (T_WD is None) only appears
        # in the R_MS–T_MS panel.
        if ds.get("T_WD") is not None:
            _kde_contour(ax_tl, ds["T_WD"], ds["R_WD"], **style)
            _kde_contour(ax_bl, ds["T_WD"], ds["T_MS"], **style)
        _kde_contour(ax_br, ds["R_MS"], ds["T_MS"], **style)

    ax_tl.set_ylabel(r"$R_\mathrm{WD}\ [R_\odot]$", fontsize=14)
    ax_tl.tick_params(labelbottom=False)
    ax_tl.grid(alpha=0.25)

    ax_bl.set_xlabel(r"$T_\mathrm{WD}\ [\mathrm{K}]$", fontsize=14)
    ax_bl.set_ylabel(r"$T_\mathrm{MS}\ [\mathrm{K}]$", fontsize=14)
    ax_bl.grid(alpha=0.25)

    ax_br.set_xlabel(r"$R_\mathrm{MS}\ [R_\odot]$", fontsize=14)
    ax_br.tick_params(labelleft=False)
    ax_br.grid(alpha=0.25)

    handles = [
        Line2D([0], [0], color=ds.get("contour_color", ds["color"]),
               lw=ds.get("linewidth", _CONTOUR_LW) + 0.9, label=ds["label"])
        for ds in datasets
    ]
    ax_tl.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=12,
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Main entry
# ══════════════════════════════════════════════════════════════════════════════

def run_display(
    gaia_id,
    tic_id,
    output_base,
    posterior_samples_sed_path,    # .npy [N, 6]: T_WD R_WD T_MS R_MS d A_V
    posterior_samples_dil_path,    # .npy [N, 2]: T_MS R_MS
    posterior_samples_joint_path,  # .npy [N, ≥4]: T_WD R_WD T_MS R_MS ...
    seed=42,
    posterior_samples_dil_matched_path=None,  # .npy [N, 4]: T_WD R_WD T_MS R_MS
):
    """
    Parameters
    ----------
    gaia_id, tic_id : int
    output_base : Path
    posterior_samples_sed_path : str or Path
    posterior_samples_dil_path : str or Path
        [N, 2] dilution samples: T_MS [K], R_MS [R_sun].
    posterior_samples_joint_path : str or Path
    seed : int
    posterior_samples_dil_matched_path : str or Path or None
        [N, 4] dilution samples with the white dwarf each one was computed
        with: T_WD [K], R_WD [R_sun], T_MS [K], R_MS [R_sun]. When it is
        given and exists, the dilution posterior is drawn in all three panels
        with its real correlations. Otherwise (older outputs) it is drawn in
        the R_MS–T_MS panel only: pairing it with independent SED draws
        would show a correlation that was never measured.

    Returns
    -------
    str  — absolute path to the saved PDF
    """
    rng     = np.random.default_rng(seed)
    out_dir = Path(output_base) / f"TIC{tic_id}_Gaia{gaia_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load posteriors ───────────────────────────────────────────────────
    samp_sed   = np.load(posterior_samples_sed_path)    # [N_sed,   6]
    samp_dil   = np.load(posterior_samples_dil_path)    # [N_dil,   2]
    samp_joint = np.load(posterior_samples_joint_path)  # [N_joint, 6]

    print(f"[DISPLAY] SED posterior   : {samp_sed.shape[0]} samples")
    print(f"[DISPLAY] Dilution posterior: {samp_dil.shape[0]} samples")
    print(f"[DISPLAY] Joint posterior : {samp_joint.shape[0]} samples")

    # ── SED dataset ───────────────────────────────────────────────────────
    ds_sed = {
        "label": "SED posterior",
        "color": "tab:blue",
        "zorder": 1,
        "T_WD": samp_sed[:, 0],
        "R_WD": samp_sed[:, 1],
        "T_MS": samp_sed[:, 2],
        "R_MS": samp_sed[:, 3],
    }

    # ── Dilution dataset ──────────────────────────────────────────────────
    # With the matched file, every row holds the (T_WD, R_WD) the dilution
    # sample was actually computed with, so all panels show real correlations.
    matched = None
    if posterior_samples_dil_matched_path is not None and \
            Path(posterior_samples_dil_matched_path).exists():
        matched = np.load(posterior_samples_dil_matched_path)
        if matched.ndim != 2 or matched.shape[1] != 4:
            raise ValueError(f"Matched dilution samples must be [N, 4], got {matched.shape}")
    if matched is not None:
        T_WD_dil, R_WD_dil, T_MS_dil, R_MS_dil = (matched[:, c] for c in range(4))
        ok = (np.isfinite(T_WD_dil) & np.isfinite(R_WD_dil)
              & np.isfinite(T_MS_dil) & np.isfinite(R_MS_dil))
        ds_dil = {
            "label": "Dilution posterior",
            "color": "peru",
            "zorder": 2,
            "T_WD": T_WD_dil[ok],
            "R_WD": R_WD_dil[ok],
            "T_MS": T_MS_dil[ok],
            "R_MS": R_MS_dil[ok],
        }
        print(f"[DISPLAY] Dilution posterior drawn with its matched white-dwarf samples")
    else:
        # Older output: only T_MS and R_MS were recorded. They are shown in the
        # R_MS–T_MS panel only; no white-dwarf pairing is invented.
        T_MS_dil, R_MS_dil = samp_dil[:, 0], samp_dil[:, 1]
        ok = np.isfinite(T_MS_dil) & np.isfinite(R_MS_dil)
        ds_dil = {
            "label": "Dilution posterior (R_MS–T_MS only)",
            "color": "peru",
            "zorder": 2,
            "T_WD": None,
            "R_WD": None,
            "T_MS": T_MS_dil[ok],
            "R_MS": R_MS_dil[ok],
        }
        print("[DISPLAY] No matched dilution samples (older output): the dilution "
              "posterior is drawn in the R_MS–T_MS panel only")

    # ── Joint dataset ─────────────────────────────────────────────────────
    # Black contours on top, over a neutral grey cloud: the joint posterior is
    # the tightest of the three and must stay readable whatever it overlaps.
    ds_joint = {
        "label": "Joint posterior",
        "color": "0.55",
        "contour_color": "black",
        "linewidth": _CONTOUR_LW_TOP,
        "zorder": 3,
        "T_WD": samp_joint[:, 0],
        "R_WD": samp_joint[:, 1],
        "T_MS": samp_joint[:, 2],
        "R_MS": samp_joint[:, 3],
    }

    # ── Build and save plot ───────────────────────────────────────────────
    fig = _make_corner([ds_sed, ds_dil, ds_joint], rng=rng)

    # dpi drives the rasterized scatter layers only; text and contours stay vector.
    out_path = out_dir / "plot_corner.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=_RASTER_DPI)
    plt.close(fig)
    print(f"[DISPLAY] Corner plot saved → {out_path}")

    return str(out_path)


# ══════════════════════════════════════════════════════════════════════════════
# Composite figure  (Transit | SED) + Corner
# ══════════════════════════════════════════════════════════════════════════════

def _pdf_to_pil(path, dpi=150):
    """Render first page of a PDF to a PIL RGB Image.
    Tries PyMuPDF (fitz) first, then pdf2image as fallback.
    """
    try:
        import fitz
        doc = fitz.open(str(path))
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = doc[0].get_pixmap(matrix=mat)
        from PIL import Image
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except ImportError:
        pass
    try:
        from pdf2image import convert_from_path
        from PIL import Image
        return convert_from_path(str(path), dpi=dpi)[0].convert("RGB")
    except ImportError:
        raise ImportError(
            "PDF rendering requires PyMuPDF (pip install pymupdf) "
            "or pdf2image + poppler (pip install pdf2image)."
        )


def _fit_in_box(im, box_w, box_h, bg=(255, 255, 255)):
    from PIL import Image
    w, h   = im.size
    scale  = min(box_w / w, box_h / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    canvas = Image.new("RGB", (box_w, box_h), bg)
    canvas.paste(im.resize((nw, nh), Image.Resampling.LANCZOS),
                 ((box_w - nw) // 2, (box_h - nh) // 2))
    return canvas


def _hstack(imgs, pad=24, bg=(255, 255, 255)):
    from PIL import Image
    W = sum(im.width for im in imgs) + pad * (len(imgs) - 1)
    H = max(im.height for im in imgs)
    canvas = Image.new("RGB", (W, H), bg)
    x = 0
    for im in imgs:
        canvas.paste(im, (x, (H - im.height) // 2))
        x += im.width + pad
    return canvas


def _vstack(imgs, pad=20, bg=(255, 255, 255)):
    from PIL import Image
    W = max(im.width for im in imgs)
    H = sum(im.height for im in imgs) + pad * (len(imgs) - 1)
    canvas = Image.new("RGB", (W, H), bg)
    y = 0
    for im in imgs:
        canvas.paste(im, ((W - im.width) // 2, y))
        y += im.height + pad
    return canvas


def run_composite_figure(
    gaia_id,
    tic_id,
    output_base,
    lc_spec_dir    = None,
    corner_side_px = 980,
    pad_h          = 24,
    pad_v          = 20,
    dpi            = 150,
    light_curve_pdf = None,
):
    """Composite figure: left column (light curve / SED) + right column (corner plot).

    File locations expected:
      Light curve : light_curve_pdf — main_all.py passes the SPECULOOS figure
                    <WORKSPACE>/TIC<id>/speculoos_light_curve/TIC<id>_speculoos_light_curve.pdf.
                    Without it, lc_spec_dir / TIC{tic_id}.pdf (former layout).
      SED plot    : output_base / TIC{tic_id}_Gaia{gaia_id} / sed / plot_sed.pdf
      Corner      : output_base / TIC{tic_id}_Gaia{gaia_id} / plot_corner.pdf

    Output:
      output_base / TIC{tic_id}_Gaia{gaia_id} / plot_composite.pdf
      (overwritten on every run)

    Raises FileNotFoundError, naming the missing file, when an input is absent.
    """
    from PIL import Image

    if light_curve_pdf is None and lc_spec_dir is None:
        raise ValueError("run_composite_figure needs light_curve_pdf or lc_spec_dir")
    star_dir    = Path(output_base) / f"TIC{tic_id}_Gaia{gaia_id}"
    transit_pdf = (Path(light_curve_pdf) if light_curve_pdf is not None
                   else Path(lc_spec_dir) / f"TIC{tic_id}.pdf")
    sed_pdf     = star_dir / "sed" / "plot_sed.pdf"
    corner_pdf  = star_dir / "plot_corner.pdf"
    out_path    = star_dir / "plot_composite.pdf"

    for p in (transit_pdf, sed_pdf, corner_pdf):
        if not p.exists():
            raise FileNotFoundError(f"[COMPOSITE] Missing file: {p}")

    im_transit = _pdf_to_pil(transit_pdf, dpi=dpi)
    im_sed     = _pdf_to_pil(sed_pdf,    dpi=dpi)
    im_corner  = _pdf_to_pil(corner_pdf, dpi=dpi)

    # Left column: transit on top, SED below (scaled to transit width)
    left_w   = im_transit.width
    sed_h    = max(1, int(round(im_sed.height * left_w / im_sed.width)))
    im_sed_r = im_sed.resize((left_w, sed_h), Image.Resampling.LANCZOS)
    left_col = _vstack([im_transit, im_sed_r], pad=pad_v)

    # Scale left column so its total height matches corner_side_px exactly
    left_scale = corner_side_px / left_col.height
    left_new_w = max(1, int(round(left_col.width * left_scale)))
    left_col   = left_col.resize((left_new_w, corner_side_px), Image.Resampling.LANCZOS)

    # Right column: corner plot fitted into a square of the same height
    right_col = _fit_in_box(im_corner, corner_side_px, corner_side_px)

    composite = _hstack([left_col, right_col], pad=pad_h)
    composite.save(str(out_path), format="PDF")
    print(f"[COMPOSITE] Saved → {out_path}")

    return str(out_path)
