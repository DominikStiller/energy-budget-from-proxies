from __future__ import annotations

import os
import string
from pathlib import Path
from typing import TYPE_CHECKING

import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cartopy.util import add_cyclic_point

import lmrecon.colormaps  # noqa: F401
import lmrecon.plotting_config  # noqa: F401
from lmrecon.cesm_tools import map_cesm_to_cf_field
from lmrecon.util import has_cftime_timedim, round_to, to_cf_order

if TYPE_CHECKING:
    import xarray as xr


def save_plot(
    plots_dir: Path | str, name: str, fig=None, type="pdf", bbox_inches="tight", **kwargs
):
    if isinstance(plots_dir, str):
        plots_dir = Path(plots_dir)

    plots_dir.mkdir(parents=True, exist_ok=True)

    if bbox_inches and "bbox_inches" not in kwargs:
        kwargs["bbox_inches"] = "tight"

    if fig is None:
        fig = plt.gcf()
    fig.savefig(
        os.path.join(plots_dir, f"{name}.{type}"),
        pad_inches=0.03,
        **kwargs,
    )


def format_plot(
    zeroline=False,
    major_grid=False,
    minor_grid=False,
):
    fig = plt.gcf()
    for ax in fig.axes:
        if hasattr(ax, "_colorbar"):
            continue

        if zeroline:
            ax.axhline(0, linewidth=0.5, c="black", zorder=-10)

        if major_grid:
            ax.grid(visible=True, which="major")
        if minor_grid:
            ax.grid(visible=True, which="minor")

    fig.align_labels()


def plot_field(
    axs,
    das,
    /,
    colorbar=True,
    cbar_label=None,
    vmin=None,
    vmax=None,
    cmap=None,
    plot_method="pcolormesh",
    coastlines=True,
    shade_land=False,
    twoslopenorm=False,
    rotate_cbar_ticks=False,
    cbar_orientation="horizontal",
    cbar_tick_angle=18,
    cbar_aspect=None,
    cbar_ticks=None,
    cbar_kwargs=None,
    cbar_cax=None,
    n_level=50,
    same_limits=True,
    **kwargs,
):
    if not isinstance(axs, list | np.ndarray):
        axs = np.array([axs])
    if not isinstance(das, list | np.ndarray):
        das = [das]

    das = [to_cf_order(da).load() for da in das]

    same_limits = same_limits and ((vmax is None and vmin is None) or (vmax == -vmin))
    vmin = float(vmin if vmin is not None else min([da.min() for da in das]))
    vmax = float(vmax if vmax is not None else max([da.max() for da in das]))

    if same_limits:
        max_v = max(abs(vmin), abs(vmax))
        vmax = max_v
        vmin = -max_v

    if cmap is None:
        cmap = "cmc.vik" if same_limits else "cmo.amp"

    for ax, da in zip(np.array(axs).flatten(), das):
        kwargs_plot = {
            "transform": ccrs.PlateCarree(),
            "cmap": cmap,
            "extend": "both",
        } | kwargs

        if not twoslopenorm:
            kwargs_plot |= {
                "vmin": vmin,
                "vmax": vmax,
            }
            # # Use our own locator because the default locator does not respect vmin/vmax
            levels = mpl.ticker.MaxNLocator(n_level + 1).tick_values(vmin, vmax)
        else:
            kwargs_plot |= {"norm": mpl.colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)}
            levels = np.concatenate(
                [
                    mpl.ticker.MaxNLocator(n_level // 2).tick_values(vmin, 0),
                    mpl.ticker.MaxNLocator(n_level // 2).tick_values(0, vmax)[1:],
                ]
            )

        lat = da.lat
        if lat.ndim == 1:
            da, lon = add_cyclic_point(da.values, coord=da.lon)
        else:
            lon = da.lon
            da = da.values

        if plot_method == "contourf":
            if "levels" not in kwargs_plot:
                kwargs_plot |= {
                    "levels": levels,
                }
            cset = ax.contourf(
                lon,
                lat,
                da,
                **kwargs_plot,
            )
            # Rasterized kwarg does not work (https://github.com/matplotlib/matplotlib/issues/27669)
            # Therefore do it after plotting
            # This prints a warning but does, in fact, work
            cset.set_rasterized(True)
        elif plot_method == "pcolormesh":
            kwargs_pcolormesh = kwargs_plot.copy()
            kwargs_pcolormesh.pop("extend", None)
            cset = ax.pcolormesh(
                lon,
                lat,
                da,
                rasterized=True,
                **kwargs_pcolormesh,
            )
        else:
            raise ValueError("Invalid plotting method")

        if coastlines:
            ax.coastlines()
        if shade_land:
            ax.add_feature(cartopy.feature.LAND, color="lightgrey")

    if colorbar:
        if cbar_aspect is None:
            if cbar_orientation == "horizontal":
                cbar_aspect = 20 * np.atleast_2d(axs).shape[1]
            else:
                cbar_aspect = 20
        cb = plt.colorbar(
            cset,
            ax=axs if cbar_cax is None else None,
            orientation=cbar_orientation,
            extend=kwargs_plot.get("extend"),
            label=cbar_label,
            aspect=cbar_aspect,
            ticks=cbar_ticks,
            cax=cbar_cax,
            **(cbar_kwargs or {}),
        )

        if rotate_cbar_ticks:
            cb.ax.tick_params(rotation=cbar_tick_angle)


def plot_field_vertical(
    axs,
    das,
    x=None,
    lev=None,
    colorbar=True,
    cbar_label=None,
    vmin=None,
    vmax=None,
    twoslopenorm=False,
    cmap=None,
    highlight_contour=None,
    contour_interval=None,
    rotate_cbar_ticks=False,
    cbar_orientation="horizontal",
    cbar_tick_angle=18,
    n_level=50,
    same_limits=True,
    **kwargs,
):
    if not isinstance(axs, list | np.ndarray):
        axs = np.array([axs])
    if not isinstance(das, list | np.ndarray):
        das = [das]

    das = [da.load() for da in das]

    same_limits = same_limits and ((vmax is None and vmin is None) or (vmax == -vmin))
    vmin = float(vmin if vmin is not None else min([da.min() for da in das]))
    vmax = float(vmax if vmax is not None else max([da.max() for da in das]))

    if same_limits:
        max_v = max(abs(vmin), abs(vmax))
        vmax = max_v
        vmin = -max_v

    if cmap is None:
        cmap = "cmc.vik" if same_limits else "cmo.amp"

    for ax, da in zip(axs, das):
        kwargs_contourf = {
            "extend": "both" if same_limits else "neither",
            "cmap": cmap,
        } | kwargs
        if not twoslopenorm:
            kwargs_contourf |= {
                "vmin": vmin,
                "vmax": vmax,
            }
            # Use our own locator because the default locator does not respect vmin/vmax
            levels = mpl.ticker.MaxNLocator(n_level + 1).tick_values(vmin, vmax)
        else:
            kwargs_contourf |= {"norm": mpl.colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)}
            levels = np.concatenate(
                [
                    mpl.ticker.MaxNLocator(n_level // 2).tick_values(vmin, 0),
                    mpl.ticker.MaxNLocator(n_level // 2).tick_values(0, vmax)[1:],
                ]
            )

        if "levels" not in kwargs_contourf:
            kwargs_contourf |= {
                "levels": levels,
            }

        if x is None:
            x = da.lat
        if lev is None:
            lev = da.lev

        cset = ax.contourf(
            x,
            lev,
            da.values,
            **kwargs_contourf,
        )

        if contour_interval is not None:
            if isinstance(contour_interval, list | np.ndarray):
                levels = contour_interval
            else:
                levels = np.arange(
                    round_to(vmin, contour_interval),
                    round_to(vmax, contour_interval),
                    contour_interval,
                )
            cset_contours = ax.contour(x, lev, da.values, levels=levels, colors="black")
            ax.clabel(cset_contours, cset_contours.levels, inline=True, fontsize=10)

        cset.set_rasterized(True)

        if highlight_contour is not None:
            c_highlight = ax.contour(
                da.lat,
                da.lev,
                da.values,
                [highlight_contour],
                transform=ccrs.PlateCarree(),
                colors="C1",
            )

    if colorbar:
        cb = plt.colorbar(cset, ax=axs, orientation=cbar_orientation, label=cbar_label)

        if highlight_contour:
            cb.add_lines(c_highlight)
        if rotate_cbar_ticks:
            cb.ax.tick_params(rotation=cbar_tick_angle)


def plot_hovmoeller(
    axs,
    das,
    /,
    time_horizontal=True,
    colorbar=True,
    cbar_label=None,
    vmin=None,
    vmax=None,
    cmap=None,
    plot_method="pcolormesh",
    twoslopenorm=False,
    rotate_cbar_ticks=False,
    cbar_orientation="vertical",
    cbar_tick_angle=18,
    cbar_aspect=None,
    cbar_ticks=None,
    cbar_kwargs=None,
    cbar_cax=None,
    n_level=50,
    same_limits=True,
    **kwargs,
):
    if not isinstance(axs, list | np.ndarray):
        axs = np.array([axs])
    if not isinstance(das, list | np.ndarray):
        das = [das]

    assert all(da.ndim == 2 for da in das)
    das = [da.load() for da in das]

    same_limits = same_limits and ((vmax is None and vmin is None) or (vmax == -vmin))
    vmin = float(vmin if vmin is not None else min([da.min() for da in das]))
    vmax = float(vmax if vmax is not None else max([da.max() for da in das]))

    if same_limits:
        max_v = max(abs(vmin), abs(vmax))
        vmax = max_v
        vmin = -max_v

    if cmap is None:
        cmap = "cmc.vik" if same_limits else "cmo.amp"

    for ax, da in zip(axs.flatten(), das):
        kwargs_plot = {
            "cmap": cmap,
            "extend": "both",
        } | kwargs

        if not twoslopenorm:
            kwargs_plot |= {
                "vmin": vmin,
                "vmax": vmax,
            }
            # # Use our own locator because the default locator does not respect vmin/vmax
            levels = mpl.ticker.MaxNLocator(n_level + 1).tick_values(vmin, vmax)
        else:
            kwargs_plot |= {"norm": mpl.colors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)}
            levels = np.concatenate(
                [
                    mpl.ticker.MaxNLocator(n_level // 2).tick_values(vmin, 0),
                    mpl.ticker.MaxNLocator(n_level // 2).tick_values(0, vmax)[1:],
                ]
            )

        time = da.time.values
        if has_cftime_timedim(da):
            time = time.astype("datetime64[s]")

        da = da.transpose("time", ...)
        other_dim_name = da.dims[1]
        if other_dim_name == "lon":
            da, other_dim = add_cyclic_point(da.values, coord=da["lon"])
        else:
            other_dim = da[other_dim_name]

        if time_horizontal:
            x = time
            y = other_dim
            da = da.T
        else:
            x = other_dim
            y = time

        if plot_method == "contourf":
            if "levels" not in kwargs_plot:
                kwargs_plot |= {
                    "levels": levels,
                }
            cset = ax.contourf(
                x,
                y,
                da,
                **kwargs_plot,
            )
            # Rasterized kwarg does not work (https://github.com/matplotlib/matplotlib/issues/27669)
            # Therefore do it after plotting
            # This prints a warning but does, in fact, work
            cset.set_rasterized(True)
        elif plot_method == "pcolormesh":
            kwargs_pcolormesh = kwargs_plot.copy()
            kwargs_pcolormesh.pop("extend", None)
            cset = ax.pcolormesh(
                x,
                y,
                da,
                rasterized=True,
                **kwargs_pcolormesh,
            )
        else:
            raise ValueError("Invalid plotting method")

    if colorbar:
        if cbar_aspect is None:
            if cbar_orientation == "horizontal":
                cbar_aspect = 20 * np.atleast_2d(axs).shape[1]
            else:
                cbar_aspect = 20
        # if cbar_ticks is None:
        #     # Imitate AutoLocator but prune both sides, necessary of extend="both"
        #     cbar_ticks = mpl.ticker.MaxNLocator(
        #         "auto", steps=[1, 2, 2.5, 5, 10], prune="both"
        #     ).tick_values(vmin, vmax)
        cb = plt.colorbar(
            cset,
            ax=axs if cbar_cax is None else None,
            orientation=cbar_orientation,
            extend=kwargs_plot.get("extend"),
            label=cbar_label,
            aspect=cbar_aspect,
            ticks=cbar_ticks,
            cax=cbar_cax,
            **(cbar_kwargs or {}),
        )

        if rotate_cbar_ticks:
            cb.ax.tick_params(rotation=cbar_tick_angle)


def add_hatches(
    axs,
    das,
    style="/////",
    c="black",
    lw=0.5,
    fc="none",
):
    if not isinstance(axs, list | np.ndarray):
        axs = np.array([axs])
    if not isinstance(das, list | np.ndarray):
        das = [das]

    das = [to_cf_order(da).load() for da in das]

    for ax, da in zip(np.array(axs).flatten(), das):
        lat = da.lat
        if lat.ndim == 1:
            da, lon = add_cyclic_point(da.values, coord=da.lon)
        else:
            lon = da.lon
            da = da.values

        with mpl.rc_context({"hatch.color": c, "patch.edgecolor": c, "hatch.linewidth": lw}):
            ax.contourf(
                lon,
                lat,
                da,
                hatches=[None, style],
                colors=["none", fc],
                transform=ccrs.PlateCarree(),
            )


def shade_ci(
    ax,
    da: xr.DataArray | tuple[xr.DataArray, xr.DataArray],
    ci=0.90,
    dim="ens",
    x_dim="time",
    alpha=0.15,
    c=None,
    lw=0,
    **kwargs,
) -> tuple[xr.DataArray, xr.DataArray]:
    if isinstance(da, tuple):
        assert len(da) == 2
        lower, upper = da
    else:
        lower = da.quantile((1 - ci) / 2, dim)
        upper = da.quantile(1 - (1 - ci) / 2, dim)
    ax.fill_between(
        lower[x_dim],
        lower,
        upper,
        alpha=alpha,
        fc=c,
        lw=lw,
        **kwargs,
    )
    return lower, upper


def shade_hdi(
    ax, da: xr.DataArray, ci=0.90, dim="ens", x_dim="time", alpha=0.15, c=None, **kwargs
) -> tuple[xr.DataArray, xr.DataArray]:
    import arviz

    hdi = (
        arviz.hdi(da.compute(), hdi_prob=ci, input_core_dims=[[dim]])
        .to_dataarray()
        .squeeze("variable")
    )
    lower = hdi.sel(hdi="lower")
    upper = hdi.sel(hdi="higher")
    ax.fill_between(
        da[x_dim],
        lower,
        upper,
        alpha=alpha,
        fc=c,
        **kwargs,
    )
    return lower, upper


def plot_ensemble_members(
    ax, da, dim="time", ens_dim="ens", label=None, alpha=0.15, c="black", zorder=-1, **kwargs
):
    assert da.ndim == 2
    da = da.transpose(..., ens_dim)

    if not isinstance(label, list):
        # Show label only for first ensemble member
        label = [label] + [None] * (len(da[ens_dim]) - 1)

    ax.plot(
        da[dim],
        da,
        alpha=alpha,
        c=c,
        zorder=zorder,
        label=label,
        **kwargs,
    )


def subplots_cartopy(
    nrows=1,
    ncols=1,
    projection=None,
    central_longitude=None,
    pole_lowest_lat=None,
    cbar_vertical=False,
    **kwargs,
):
    if projection is None or projection == "equal_earth":
        if central_longitude is None:
            central_longitude = 200
        projection = ccrs.EqualEarth(central_longitude=central_longitude)
    elif projection == "arctic":
        if central_longitude is None:
            central_longitude = 0
        projection = ccrs.NorthPolarStereo(central_longitude=central_longitude)
    elif projection == "antarctic":
        if central_longitude is None:
            central_longitude = 0
        projection = ccrs.SouthPolarStereo(central_longitude=central_longitude)

    if "subplot_kw" not in kwargs:
        kwargs["subplot_kw"] = {}
    kwargs["subplot_kw"]["projection"] = projection

    if "figsize" not in kwargs:
        if isinstance(projection, ccrs.NorthPolarStereo | ccrs.SouthPolarStereo):
            figsize = (1.7 * ncols, 1.7 * nrows)
        else:
            figsize = (1.75 * 1.7 * ncols, 1.7 * nrows)
        if cbar_vertical:
            figsize = (figsize[0] + 1, figsize[1])
        kwargs["figsize"] = figsize

    fig, axs = plt.subplots(nrows, ncols, **kwargs)

    if isinstance(projection, ccrs.NorthPolarStereo):
        if pole_lowest_lat is None:
            pole_lowest_lat = 49
        for ax in np.atleast_1d(axs).flat:
            ax.set_extent([0, 360, pole_lowest_lat, 90], crs=ccrs.PlateCarree())
    elif isinstance(projection, ccrs.SouthPolarStereo):
        if pole_lowest_lat is None:
            pole_lowest_lat = 53
        for ax in np.atleast_1d(axs).flat:
            ax.set_extent([0, 360, -90, -pole_lowest_lat], crs=ccrs.PlateCarree())

    return fig, axs


def add_subplot_headers(
    row_headers=None,
    col_headers=None,
    row_pad=5,
    col_pad=10,
    rotate_row_headers=True,
    **text_kwargs,
):
    # Adapted from https://stackoverflow.com/a/71887460

    _text_kwargs = {
        "fontweight": "bold",
        "fontsize": "large",
    }
    if text_kwargs:
        _text_kwargs |= text_kwargs

    for ax in plt.gcf().axes:
        if hasattr(ax, "_colorbar"):
            continue

        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **_text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            row_header = str(row_headers[sbs.rowspan.start])
            n_lines = len(row_header.splitlines())
            # Keep rotated multiline labels from drifting into the axes.
            multiline_extra_pad = (
                (n_lines - 1) * mpl.rcParams["font.size"] * 1.2 if rotate_row_headers else 0
            )

            try:
                # May fail if cartopy is not imported
                is_geoaxes = isinstance(ax, cartopy.mpl.geoaxes.GeoAxes)
            except AttributeError:
                is_geoaxes = False

            if is_geoaxes:
                xycoords = "axes fraction"
                xytext = (-row_pad - multiline_extra_pad, 0)
            else:
                xycoords = ax.yaxis.label
                xytext = (-ax.yaxis.labelpad - row_pad - multiline_extra_pad, 0)

            ax.annotate(
                row_header,
                xy=(0, 0.5),
                xytext=xytext,
                xycoords=xycoords,
                textcoords="offset points",
                ha="center" if rotate_row_headers else "right",
                va="center",
                rotation=90 if rotate_row_headers else 0,
                rotation_mode="anchor" if rotate_row_headers else None,
                multialignment="center" if rotate_row_headers else None,
                **_text_kwargs,
            )


def add_panel_label(
    ax,
    label,
    xytext=None,
    ha="right",
    va="baseline",
    parentheses=False,
    title_has_superscript=False,
):
    if isinstance(label, int):
        label = string.ascii_lowercase[label]
    if parentheses:
        label = "(" + label + ")"
    else:
        label = label + ")"
    if xytext is None:
        x = -3 if ha == "right" else 0
        if title_has_superscript:
            # Title is shifted in pdf, but not in png, if there is a superscript (e.g., $^{-2}$)
            xytext = (x, mpl.rcParams["axes.titlepad"] * 1.16)
        else:
            xytext = (x, mpl.rcParams["axes.titlepad"])
    ax.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=xytext,
        textcoords="offset points",
        fontweight="bold",
        fontsize=mpl.rcParams["axes.titlesize"],
        ha=ha,
        va=va,
    )


def invert_ticks(x):
    # From https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.secondary_xaxis.html
    # 1/x with special treatment of x == 0
    # Useful for secondary axis with period/frequency
    x = np.array(x).astype(float)
    near_zero = np.isclose(x, 0)
    x[near_zero] = np.inf
    x[~near_zero] = 1 / x[~near_zero]
    return x


def get_field_unit(field, scaling=1, thin_space=False):
    field = map_cesm_to_cf_field(field)
    if field in ["ohc300_total", "ohc_total"] and np.isclose(scaling, 1e21):
        return "ZJ"
    if field in ["siconc", "siconcs", "siconcn", "clt", "cll", "clm", "clh"] and np.isclose(
        scaling, 1e-2
    ):
        return "%"

    unit = {
        "tas": "K",
        "tos": "K",
        "zg500": "m",
        "ohc300": "J m$^{-2}$",
        "ohc700": "J m$^{-2}$",
        "ddtohc300": "W m$^{-2}$",
        "rsdt": "W m$^{-2}$",
        "rsut": "W m$^{-2}$",
        "rlut": "W m$^{-2}$",
        "rsnt": "W m$^{-2}$",
        "eei": "W m$^{-2}$",
        "eei_inferred": "W m$^{-2}$",
        "eei_direct": "W m$^{-2}$",
        "R_eei": "W m$^{-2}$",
        "F_eei": "W m$^{-2}$",
        "CRE": "W m$^{-2}$",
        "clt": "1",
        "cll": "1",
        "clm": "1",
        "clh": "1",
        "clwvi": "kg m$^{-2}$",
        "siconc": "1",
        "siconcn": "1",
        "siconcs": "1",
        "siarea": "km²",
        "siarean": "km²",
        "siareas": "km²",
        "siextentn": "km²",
        "siextents": "km²",
        "nino34_index": "K",
        "pdo_index": "1",
        "amo_index": "K",
        "ipo_index": "K",
        "lambda": "W m$^{-2}$ K$^{-1}$",
        "lambda_sw": "W m$^{-2}$ K$^{-1}$",
        "lambda_lw": "W m$^{-2}$ K$^{-1}$",
    }.get(field, "??")
    if not np.isclose(scaling, 1):
        exp = np.log10(scaling)
        unit = f"×10$^{{{exp:.0f}}}$ {unit}"
    if thin_space:
        unit = unit.replace(" ", " ")
    return unit


def get_field_label(field, add_units=False, scaling=1):
    field = map_cesm_to_cf_field(field)
    names = {
        "tas": "Surface air temperature",
        "tos": "Sea surface temperature",
        "zg500": "500-hPa geopotential height",
        "ohc300": "OHC 300 m",
        "ohc300_total": "OHC 300 m",
        "ohc_total": "Integrated energy imbalance",
        "ohc700": "OHC 700 m",
        "ddtohc300": "d/dt OHC 300 m",
        "rsdt": "Insolation",
        "rsut": "Reflected SW",
        "rsnt": "Absorbed SW",
        "rlut": "Outgoing LW",
        "eei": "Energy imbalance",
        "eei_inferred": "Inferred energy imbalance",
        "eei_direct": "Directly reconstructed energy imbalance",
        "R_eei": "Response",
        "F_eei": "Forcing",
        "CRE": "Cloud-radiative effect",
        "clt": "Total cloud fraction",
        "cll": "Low-cloud fraction",
        "clm": "Mid-level-cloud fraction",
        "clh": "High-cloud fraction",
        "clwvi": "Condensed water path",
        "siconc": "Sea-ice concentration",
        "siconcn": "Arctic sea-ice concentration",
        "siconcs": "Antarctic sea-ice concentration",
        "siarea": "Sea-ice area",
        "siarean": "Arctic sea-ice area",
        "siareas": "Antarctic sea-ice area",
        "siextentn": "Arctic sea-ice extent",
        "siextents": "Antarctic sea-ice extent",
        "nino34_index": "Niño 3.4 index",
        "pdo_index": "PDO index",
        "amo_index": "AMO index",
        "ipo_index": "IPO index",
        "lambda": r"Feedback parameter $\lambda$",
        "lambda_sw": r"Shortwave feedback parameter $\lambda_\text{SW}$",
        "lambda_lw": r"Longwave feedback parameter $\lambda_\text{LW}$",
        "lambda_cs": r"Clear-sky feedback parameter $\lambda_\text{CS}$",
        "lambda_cre": r"Cloud-radiative effect feedback parameter $\lambda_\text{CRE}$",
    }

    if field not in names:
        return field

    label = names[field]
    if add_units:
        label += f" ({get_field_unit(field, scaling)})"
    return label


def get_field_shorthand_label(field, add_units=False, scaling=1):
    field = map_cesm_to_cf_field(field)
    names = {
        "tas": "SAT",
        "tos": "SST",
        "zg500": "ZG 500",
        "ohc300": "OHC 300 m",
        "ohc300_total": "OHC 300 m",
        "ohc700": "OHC 700 m",
        "ddtohc300": "d/dt OHC 300 m",
        "rsdt": "Solar",
        "rsut": "RSR",
        "rsnt": "ASR",
        "rlut": "OLR",
        "eei": "EEI",
        "eei_inferred": "Inferred EEI",
        "eei_direct": "Direct EEI",
        "CRE": "CRE",
        "clt": "Total cloud fraction",
        "cll": "Low-cloud fraction",
        "clm": "Medium-cloud fraction",
        "clh": "High-cloud fraction",
        "clwvi": "Condensed water path",
        "siconc": "SIC",
        "siconcn": "SIC (Arctic)",
        "siconcs": "SIC (Antarctic)",
        "siarea": "SI area",
        "siarean": "SI area (Arctic)",
        "siareas": "SI area (Antarctic)",
        "siextentn": "SI extent (Arctic)",
        "siextents": "SI extent (Antarctic)",
        "lambda": r"$\lambda$",
        "lambda_sw": r"$\lambda_\text{SW}$",
        "lambda_lw": r"$\lambda_\text{LW}$",
        "lambda_cs": r"$\lambda_\text{CS}$",
        "lambda_cre": r"$\lambda_\text{CRE}$",
    }

    if field not in names:
        return get_field_label(field, add_units)

    label = names[field]
    if add_units:
        label += f" ({get_field_unit(field, scaling)})"
    return label


def get_field_scaling(field: str) -> float:
    field = map_cesm_to_cf_field(field)
    return {
        "ohc300": 1e6,
        "ohc300_total": 1e21,
        "ohc_total": 1e21,
        "siconc": 1e-2,
        "siconcn": 1e-2,
        "siconcs": 1e-2,
        "clt": 1e-2,
        "cll": 1e-2,
        "clm": 1e-2,
        "clh": 1e-2,
        "siarean": 1e6,
        "siareas": 1e6,
        "siextentn": 1e6,
        "siextents": 1e6,
    }.get(field, 1)


def use_zero_ylim(ax, lower=True):
    if lower:
        ax.set_ylim([0, ax.get_ylim()[1]])
    else:
        ax.set_ylim([ax.get_ylim()[0], 0])


def use_zero_xlim(ax, lower=True):
    if lower:
        ax.set_xlim([0, ax.get_xlim()[1]])
    else:
        ax.set_xlim([ax.get_xlim()[0], 0])


def adjust_color_lightness(color, amount):
    """
    Adjust the lightness of a matplotlib color

    From https://stackoverflow.com/a/49601444

    Args:
        color: color
        amount: lighten if >1, darken if <1

    Returns:
        Lightened/darkened color
    """
    import colorsys

    import matplotlib.colors as mc

    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def set_yaxis_color(ax, c, side="right"):
    ax.tick_params(axis="y", colors=c)
    ax.spines[side].set_color(c)
    ax.yaxis.label.set_color(c)
