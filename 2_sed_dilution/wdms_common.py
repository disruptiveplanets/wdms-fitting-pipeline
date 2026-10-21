"""wdms_common.py — Definitions shared by several pipeline modules.

Kept deliberately small and free of local imports, so that any module can
import it without creating a circular dependency.

* M_SUN_G            solar mass in grams, from astropy.constants.M_sun
                     (the only solar-mass value used by the pipeline)
* T_MS_MIN_K/MAX_K   companion (M dwarf) temperature interval adopted as the
                     fitting bound in every stage, in K
* derive_seed        deterministic sub-seed from a master seed and a label
"""

import hashlib

from astropy.constants import M_sun

# ── Physical constants ────────────────────────────────────────────────────────
M_SUN_G = float(M_sun.cgs.value)   # g

# ── Companion temperature interval [K] ────────────────────────────────────────
# Fitting bound for T_MS in the SED fit (every companion variant), both
# dilution methods and the ideal joint fit. Model grids may extend further
# (e.g. the STScI PHOENIX grid is built over 2300–5900 K); a grid only narrows
# this interval when it does not cover it, and the stages then say so.
T_MS_MIN_K = 2300.0
T_MS_MAX_K = 4000.0


# ── Seeds ─────────────────────────────────────────────────────────────────────
def derive_seed(master_seed, label):
    """Deterministic 32-bit sub-seed for one sampling run.

    Built from SHA-256 of "<master_seed>:<label>", so it is identical across
    processes and Python sessions (unlike the built-in hash(), which is
    randomised per process). Used to give each UltraNest run of a stage its own
    reproducible seed, e.g. derive_seed(42, "sed:2comp:phoenix").
    """
    digest = hashlib.sha256(f"{int(master_seed)}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")
