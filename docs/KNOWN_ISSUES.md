# Known issues and limitations

This page lists what a user should know before relying on the results: the validation status, the modelling assumptions, the known open issues and the planned improvements.

Severity labels:

- **Medium:** affects results or uncertainties, probably within or comparable to the stated uncertainties.
- **Low:** small effect, or only for an option that is currently off.
- **Info:** a consequence of a modelling choice that should be known.

---

## Validation status

- The SED and eclipse-depth terms share one companion model (STScI PHOENIX), one gravity prescription for each star, and bilinear interpolation in ln F on both grids. These were checked against each other on controlled inputs.
- **No full nested-sampling run** (SED, ideal joint or legacy joint fit) was repeated after the last changes. Results from earlier versions of the code are not reproduced exactly.
- Seeds are deterministic, but repeatability holds only on the same software stack, and not when an UltraNest run is resumed.
- Not exercised: the `SED_ATMO_MS_LOGG_MODE = "from_radius"` gravity mode and the energy-weighted depth band fluxes (`PHOTON_COUNTING = False`).

## Modelling assumptions

- **Eclipse depth.** A total eclipse is assumed, with no third light, no reflection and no reddening inside the band integrals. A blend in the aperture biases R_MS directly.
- **White dwarfs.** Only DA atmospheres are supported consistently. `WD_ATMO_TYPE = "DB"` would pair a DB cooling track with DA spectra.
- **White-dwarf gravity.** It follows the cooling track inside its domain and falls back to log g = 7.75 outside it, so it is discontinuous at the edge of the domain (≈ 0.7 % in F_WD per 0.25 dex near 18 kK).
- **Companion gravity.** The empirical dwarf table starts at 2400 K: between 2300 and 2400 K, log g_MS is held at its 2400 K value.
- **A_V prior (Info/Medium).** The 2-component and joint A_V prior comes from the 1-component posterior of the same photometry. The A_V uncertainty is therefore understated, and ΔlogZ (2-component vs 1-component) compares models with different priors.
- **Parallax.** The published Gaia parallax is used as is, with no zero-point correction or error inflation.
- **Outlier clipping (off by default).** Leave-one-out clipping can also reject the neighbours of an outlier.
- **Blackbody companion variant.** With a Koester white dwarf, it uses a 250-node blackbody grid interpolated in ln F, not the exact blackbody.
- **Legacy joint fit.** Its T_MS bounds can exceed 2300–4000 K by up to 34 K, and its dilution table uses the BHAC15 median radius.

## Open issues

| Issue | Severity | Effect | Possible fix |
|---|---|---|---|
| The standard dilution draws R_MS from the radius–temperature relation without the depth constraint (`dilution_fit.run_dilution_fit`). | Medium | The dilution posterior (`params_dil.json`, orange in the corner plot) is too broad in R_MS. The ideal joint fit is not affected. | Sample (T, R) jointly, e.g. weight the relation draws by the depth likelihood. |
| The white dwarf is integrated over the native Koester spectra in the depth term, but over the rebinned grid in the SED term. | Low | 0.05–1 % in the z-band flux, ≈ 0.002 in depth (0.15σ). | Use the rebinned grid and weights of the SED term. |
| Interpolation error between grid nodes. | Low | ≲ 1 % in flux, ≲ 0.002 in depth, common to both terms. | Finer grids (0.25 dex in companion log g) or cubic interpolation. |
| The memoised white-dwarf gravity (`sed_fit._WDCoolingLogg`) depends on the evaluation order within a cache cell. | Info | ~10⁻⁵ in depth. | Evaluate the cell centre, or drop the cache. |

## Future improvements

- Gaia parallax zero-point correction and error inflation.
- Consistent DB white-dwarf support.
- Maintenance:
  - `datetime.utcnow()` is deprecated from Python 3.12.
  - The VizieR download uses `allow_insecure=True`.
  - The dilution plot label says "BHAC15" when `brown2022` rescales the radii.
