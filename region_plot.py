"""
region_plot.py

Two map styles, both usable for a single state OR for all of India
(pass state_name=None or "India"):

  - plot_region_pattern: a smoothed, continuous rainfall/temperature/wind/
    humidity gradient, interpolated onto a fine mesh and masked exactly to
    the region's true shape. District boundaries are drawn on top (kept,
    not erased), with the outer region boundary emphasized.

  - plot_region_spatial: a district-wise choropleth -- each district is
    filled with a single color representing its own average value, which
    is what "spatial plot" means per the app's query vocabulary.
"""

import os
import sys

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Point

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nc_data import VARIABLE_META
from districts import (
    is_india_scope, get_state_geometry, district_variable_series,
    compute_day_variable_grid,
)
from date_utils import day_label_for_date


def _scope_title(state_name):
    return "India" if is_india_scope(state_name) else state_name.title()


def plot_region_pattern(ds, gdf, id_grid, state_name, target_date, variable="rainfall", grid_res=220):
    if variable not in VARIABLE_META:
        raise ValueError(f"Unknown variable '{variable}'. Choose from {list(VARIABLE_META)}.")

    grid = compute_day_variable_grid(ds, target_date, variable)
    if grid is None:
        raise ValueError(f"No forecast data available for {target_date}.")

    if is_india_scope(state_name):
        rows = gdf
    else:
        rows = gdf[gdf["ST_NM_UP"] == state_name.strip().upper()]
        if rows.empty:
            raise ValueError(f"No district data found for '{state_name}'.")

    scope_idx = rows.index.values
    cell_mask = np.isin(id_grid, scope_idx)
    if not cell_mask.any():
        raise ValueError(f"No forecast grid cells fall inside '{state_name}'.")

    lats = ds["latitude"].values
    lons = ds["longitude"].values
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)

    valid_lat = lat_mesh[cell_mask]
    valid_lon = lon_mesh[cell_mask]
    valid_val = grid[cell_mask]

    lat_idx, lon_idx = np.where(cell_mask)
    lat_min, lat_max = lats[lat_idx].min(), lats[lat_idx].max()
    lon_min, lon_max = lons[lon_idx].min(), lons[lon_idx].max()
    pad = 0.15

    fine_lon = np.linspace(lon_min, lon_max, grid_res)
    fine_lat = np.linspace(lat_min, lat_max, grid_res)
    flon, flat = np.meshgrid(fine_lon, fine_lat)

    fine_val = griddata((valid_lon, valid_lat), valid_val, (flon, flat), method="linear")

    region_geom = get_state_geometry(gdf, state_name)
    pts = gpd.GeoSeries([Point(xy) for xy in zip(flon.ravel(), flat.ravel())], crs="EPSG:4326")
    inside = pts.within(region_geom).values.reshape(flon.shape)
    fine_val = np.where(inside, fine_val, np.nan)

    meta = VARIABLE_META[variable]
    is_india = is_india_scope(state_name)
    fig, ax = plt.subplots(figsize=(8, 9) if is_india else (8, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    cf = ax.contourf(flon, flat, fine_val, levels=20, cmap=meta["cmap"])
    plt.colorbar(cf, ax=ax, label=f"{meta['label']} ({meta['unit']})")

    # district boundaries kept (not erased), outer region boundary emphasized
    rows.boundary.plot(ax=ax, color="black", linewidth=0.35, alpha=0.6)
    gpd.GeoSeries([region_geom], crs="EPSG:4326").boundary.plot(ax=ax, color="black", linewidth=1.3)

    ax.set_xlim(lon_min - pad, lon_max + pad)
    ax.set_ylim(lat_min - pad, lat_max + pad)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    label = day_label_for_date(target_date)
    ax.set_title(f"{meta['label']} \u2014 {_scope_title(state_name)} \u2014 {label} ({target_date})")
    plt.tight_layout()
    return fig


def plot_region_spatial(ds, gdf, id_grid, state_name, target_date, variable="rainfall"):
    """District-wise choropleth: one flat color per district = its average value."""
    if variable not in VARIABLE_META:
        raise ValueError(f"Unknown variable '{variable}'. Choose from {list(VARIABLE_META)}.")

    table = district_variable_series(
        ds, gdf, id_grid, target_date, variable,
        state_name=None if is_india_scope(state_name) else state_name,
    )
    if table is None or table.empty:
        raise ValueError(f"No forecast data available for '{state_name}' on {target_date}.")

    meta = VARIABLE_META[variable]
    is_india = is_india_scope(state_name)
    fig, ax = plt.subplots(figsize=(8, 9) if is_india else (8, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    table.plot(
        column="value", ax=ax, cmap=meta["cmap"], edgecolor="black", linewidth=0.4,
        legend=True, legend_kwds={"label": f"{meta['label']} ({meta['unit']})", "shrink": 0.6},
        missing_kwds={"color": "white", "hatch": "///"},
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    label = day_label_for_date(target_date)
    ax.set_title(
        f"District-wise {meta['label']} \u2014 {_scope_title(state_name)} \u2014 {label} ({target_date})"
    )
    plt.tight_layout()
    return fig
