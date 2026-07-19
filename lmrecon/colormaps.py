from __future__ import annotations

import cmcrameri  # noqa: F401
import cmocean  # noqa: F401
import matplotlib as mpl

# 0 = transparent, 1 = black
cm_binary_alpha = mpl.colors.ListedColormap(["#0000", "#000F"], "binary_alpha")

mpl.colormaps.register(cm_binary_alpha)
