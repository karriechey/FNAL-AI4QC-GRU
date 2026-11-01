# Created: 2026-10-27
# Last modified: 2026-10-27
# Poster plot style: Helvetica (fallback Arial) for text and math, sizes in pt at final print size.
# matplotlib reads only the first face of a .ttc, so the faces of macOS Helvetica.ttc are written
# out as separate .ttf files in a local cache (never committed: Apple's font) and registered.
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

HELVETICA_TTC = Path("/System/Library/Fonts/Helvetica.ttc")
FONT_CACHE = Path.home() / "Library" / "Caches" / "poster_fonts"

AXIS_LABEL_PT = 18
TICK_PT = LEGEND_PT = CURVE_LABEL_PT = 15

MODEL = "#2A9D8F"     # GRU
NEUTRAL = "#868D95"   # MWPM in GRU vs MWPM comparisons
INK = "#1A1A1A"
DIST = {3: "#54C4FF", 5: "#5296FF", 7: "#6E65E0", 9: "#763AA8", 11: "#6B1667"}  # code distance, light to dark


def _register_helvetica_faces():
    if not HELVETICA_TTC.exists():
        return False
    from fontTools.ttLib import TTCollection
    FONT_CACHE.mkdir(parents=True, exist_ok=True)
    for i, face in enumerate(TTCollection(str(HELVETICA_TTC)).fonts):
        path = FONT_CACHE / f"Helvetica-{i}.ttf"
        if not path.exists():
            face.save(str(path))
        font_manager.fontManager.addfont(str(path))
    return True


def apply():
    family = "Helvetica" if _register_helvetica_faces() else "Arial"
    plt.rcParams.update({
        "font.family": [family, "Arial"],
        "mathtext.fontset": "custom",
        "mathtext.rm": family, "mathtext.it": f"{family}:italic", "mathtext.bf": f"{family}:bold",
        "axes.unicode_minus": False,
        "axes.labelsize": AXIS_LABEL_PT, "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT,
        "legend.fontsize": LEGEND_PT,
        "axes.linewidth": 1, "xtick.major.width": 1, "ytick.major.width": 1,
        "xtick.minor.width": 1, "ytick.minor.width": 1,
        "savefig.bbox": None,   # exact frame size, never cropped or padded
    })
    return family
