from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib.font_manager import fontManager


def set_plotting_theme(publication=False, palette_oldorder=False, font_size=8, rc=None):
    try:
        import pandas as pd

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
    except ImportError:
        pass

    # Add Arial ttfs since NCAR server do not have them
    for f in (Path(__file__).parent.parent / "fonts").glob("*.ttf"):
        fontManager.addfont(f)

    # https://github.com/garrettj403/SciencePlots/blob/master/scienceplots/styles/color/std-colors.mplstyle
    # palette=["#0C5DA5", "#00B945", "#FF9500", "#FF2C00", "#845B97", "#474747", "#9e9e9e"],
    # Okabe Ito (https://siegal.bio.nyu.edu/color-palette/)
    if palette_oldorder:
        palette = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]
    else:
        # orange, light blue, red, blue, green, purple, yellow
        palette = ["#E69F00", "#56B4E9", "#D55E00", "#0072B2", "#009E73", "#CC79A7", "#F0E442"]

    _rc = {
        "font.family": "sans-serif",
        "font.sans-serif": "Arial",
        "font.size": font_size,
        "axes.titlesize": font_size,
        "figure.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "lines.linewidth": 1,
        "lines.markersize": 2,
        "figure.figsize": (5.5, 2.5),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.constrained_layout.use": True,
        "xtick.bottom": True,
        "ytick.left": True,
        # "xtick.top": True,
        # "ytick.right": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.minor.bottom": False,
        "ytick.minor.left": False,
        "xtick.minor.top": False,
        "ytick.minor.right": False,
        "legend.frameon": False,
        "legend.borderpad": 0,
        "grid.linewidth": 0.5,
        # "xtick.direction": "in",
        # "ytick.direction": "in",
        # Undo seaborn changes from black to 15% gray
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
        "image.interpolation": "none",
    }

    if publication:
        _rc |= {
            "axes.titlelocation": "left",
        }
    else:
        _rc |= {
            "axes.titleweight": "bold",
            "figure.titleweight": "bold",
        }

    if rc:
        _rc |= rc

    plt.style.use("default")
    sb.set_theme(context="paper", style="white", palette=palette, rc=_rc)


set_plotting_theme()
