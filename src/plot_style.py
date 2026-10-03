"""Shared look of every figure.

One fixed colour per model family, always in the same order, checked for colour
blindness with the AIR Institute data-visualisation palette validator. Families
also get their own marker, because the journal is printed and read in grey.
Text stays in neutral ink; colour only marks identity.
"""
import re

import matplotlib as mpl
import matplotlib.pyplot as plt

import config as C

FAMILIES = ["simple", "statistical", "decomposition", "boosting", "recurrent", "deep",
            "foundation", "combination"]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
           "#4a3aa7", "#e34948"]
COLOR = dict(zip(FAMILIES, PALETTE))
MARKER = dict(zip(FAMILIES, ["o", "s", "D", "^", "v", "P", "*", "X"]))

# In scatter plots every pair of colours is compared, and only the first three
# slots stay distinguishable for all pairs, so families are grouped in three.
GROUP = {"simple": "classical", "statistical": "classical", "decomposition": "classical",
         "boosting": "learned", "recurrent": "learned", "deep": "learned",
         "foundation": "foundation", "combination": "combination"}
GROUP_COLOR = {"classical": PALETTE[0], "learned": PALETTE[1], "foundation": PALETTE[2],
               "combination": "#8a8985"}

INK = "#0b0b0b"
INK_2 = "#52514e"
AXIS = "#8a8985"
GRID = "#e6e5e1"
BAND = "#e9e8e4"          # shaded shift months
BAND_2 = "#f4f3f0"        # shaded rebound months
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")

SINGLE = 3.54             # 90 mm column
DOUBLE = 7.48             # 190 mm page width

RULES = {"ras", "bocpd", "fixed_share", "median_all", "comb3"}


def base_model(name: str) -> str:
    """Strip the ONI tier suffix: lgbm_oni_x2 -> lgbm_oni."""
    return re.sub(r"_x[123]$", "", name)


def family(name: str) -> str:
    if name in RULES:
        return "combination"
    return C.MODELS[base_model(name)]["family"]


def apply() -> None:
    mpl.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.titlesize": 8,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
        "lines.linewidth": 1.4, "lines.markersize": 4,
        "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, name: str) -> None:
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / f"{name}.pdf")
    fig.savefig(C.FIGURES / f"{name}.png")
    plt.close(fig)
    print(f"figure {name}")
