from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib.font_manager import fontManager


def set_plotting_theme(publication=False, force_light=False):
    try:
        import pandas as pd

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
    except ImportError:
        pass

    # Add Arial ttfs since NCAR's servers do not have them
    for f in (Path(__file__).parent.parent / "fonts").glob("*.ttf"):
        fontManager.addfont(f)

    dark = plt.rcParams["figure.facecolor"] == "black"
    if force_light:
        dark = False

    # https://github.com/garrettj403/SciencePlots/blob/master/scienceplots/styles/color/std-colors.mplstyle
    # palette=["#0C5DA5", "#00B945", "#FF9500", "#FF2C00", "#845B97", "#474747", "#9e9e9e"],
    # Okabe Ito (https://siegal.bio.nyu.edu/color-palette/)
    # orange, light blue, green, blue, red, purple, yellow
    palette = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]

    plt.style.use("default")
    if publication:
        font_size = 7.5
        sb.set_theme(
            context="paper",
            style="white",
            # https://github.com/garrettj403/SciencePlots/blob/master/scienceplots/styles/color/std-colors.mplstyle
            # palette=["#0C5DA5", "#00B945", "#FF9500", "#FF2C00", "#845B97", "#474747", "#9e9e9e"],
            # Okabe Ito (https://siegal.bio.nyu.edu/color-palette/)
            # orange, light blue, green, blue, red, purple, yellow
            palette=palette,
            rc={
                "font.size": font_size,
                "axes.titlesize": font_size,
                "figure.titlesize": font_size,
                "axes.labelsize": font_size,
                "xtick.labelsize": font_size,
                "ytick.labelsize": font_size,
                "legend.fontsize": 7,
                "font.family": "sans-serif",
                "font.sans-serif": "Arial",
                "lines.linewidth": 1,
                # "axes.titleweight": "bold",
                # "figure.titleweight": "bold",
                # "axes.labelweight": "light",
                # "font.weight": "light",
                # "mathtext.default": "regular",
                "axes.titlelocation": "left",
                "axes.spines.top": False,
                "axes.spines.right": False,
                "figure.figsize": (5.5, 2.3),
                "figure.dpi": 300,
                "savefig.dpi": 300,
                "figure.constrained_layout.use": True,
                "xtick.bottom": True,
                "ytick.left": True,
                # "xtick.top": True,
                # "ytick.right": True,
                "xtick.minor.bottom": False,
                "ytick.minor.left": False,
                "xtick.minor.top": False,
                "ytick.minor.right": False,
                "legend.frameon": False,
                "legend.borderpad": 0,
                # "xtick.direction": "in",
                # "ytick.direction": "in",
                # Undo seaborn changes from black to 15% gray
                "axes.labelcolor": "black",
                "axes.edgecolor": "black",
                "xtick.color": "black",
                "ytick.color": "black",
                "text.color": "black",
                "image.interpolation": "none",
                # "text.usetex": True,
                # "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{sfmath}",
            },
        )
    else:
        sb.set_theme(
            context="paper",
            style="white",
            palette=palette,
            font_scale=1.5,
            rc={
                "font.family": "Arial",
                "lines.linewidth": 1.5,
                "axes.titleweight": "bold",
                "figure.titleweight": "bold",
                # "axes.labelweight": "light",
                # "font.weight": "light",
                # "mathtext.default": "regular",
                "figure.figsize": (14, 6),
                "figure.dpi": 200,
                "savefig.dpi": 350,
                "figure.constrained_layout.use": True,
                "xtick.bottom": True,
                "ytick.left": True,
                # "xtick.top": True,
                # "ytick.right": True,
                "xtick.minor.bottom": False,
                "ytick.minor.left": False,
                "xtick.minor.top": False,
                "ytick.minor.right": False,
                # "xtick.direction": "in",
                # "ytick.direction": "in",
                # Undo seaborn changes from black to 15% gray
                "axes.labelcolor": "black",
                "axes.edgecolor": "black",
                "xtick.color": "black",
                "ytick.color": "black",
                "text.color": "black",
                "image.interpolation": "none",
            },
        )

    if dark:
        # For example, due to VS Code dark theme
        plt.style.use("dark_background")


set_plotting_theme()
