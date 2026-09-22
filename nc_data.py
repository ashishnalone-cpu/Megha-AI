"""
nc_data.py

Loads the ECMWF AIFS India NetCDF forecast file and provides:
  - auto-discovery of the latest ecmwf_aifs_india_YYYYMMDD_00z_merged.nc
  - IC (initialization cycle) parsing from the filename
  - UTC -> IST conversion for all forecast valid times
  - a fix for `tp` being accumulated-since-init (see TP_IS_CUMULATIVE below)
  - point (lat/lon) time series extraction via bilinear interpolation
  - "remaining day" forecasts that respect the current IST time
  - daily / day-part aggregation, IMD-style rainfall classification,
    and IMD-style multi-day tables
  - natural-language forecast text generation

ASSUMPTIONS (documented so you can adjust if your data convention differs):
  1. `tp` in the raw file is ACCUMULATED PRECIPITATION SINCE THE FORECAST
     INIT TIME (T0), which is the standard ECMWF convention. On load, we
     difference consecutive steps to recover the true 6-hourly increment
     (tp[0] is taken as-is, tp[i] = raw_tp[i] - raw_tp[i-1] for i>0,
     clipped at 0 to guard against tiny negative noise). Daily totals are
     then a simple SUM of that day's four (already-corrected) 6-hourly
     increments -- e.g. today's steps 10/15/10/5 mm -> 40 mm, tomorrow's
     steps 10/20/10/10 mm -> 50 mm, with NO carry-over between days.
     If your `tp` is already per-interval (not cumulative), set
     TP_IS_CUMULATIVE = False below (or pass tp_is_cumulative=False to
     open_forecast).
  2. `q1000` is used directly as a humidity percentage, per your
     description ("specific humidity in %"), even though the variable
     name suggests kg/kg. No conversion is applied.
  3. Wind direction is the meteorological "from" direction computed from
     u10/v10 (eastward/northward components).
"""

import glob
import os
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from date_utils import now_ist, day_label_for_date

IST_OFFSET = timedelta(hours=5, minutes=30)
STEP_HOURS = 6  # forecast output cadence
TP_IS_CUMULATIVE = True  # see assumption (1) above -- flip if your data differs

# --- IMD-style daily (24h) rainfall classification thresholds (mm) ---
# Values scale proportionally for shorter windows (e.g. 6h day-parts).
_RAIN_STEPS_24H = [
    (0.1, "No rain"),
    (2.5, "Very light rain"),
    (7.6, "Light rain"),
    (35.6, "Moderate rain"),
    (64.5, "Heavy rain"),
    (124.5, "Very heavy rain"),
]
RAIN_CATEGORY_ORDER = [
    "No rain", "Very light rain", "Light rain", "Moderate rain",
    "Heavy rain", "Very heavy rain", "Extremely heavy rain",
]
_THRESHOLD_ALIASES = {
    "light": "Light rain",
    "moderate": "Moderate rain",
    "heavy": "Heavy rain",
    "very heavy": "Very heavy rain",
    "extremely heavy": "Extremely heavy rain",
}

DAYPART_ORDER = ["Morning", "Afternoon", "Evening", "Night"]

VARIABLE_META = {
    "rainfall": {"label": "Rainfall", "unit": "mm", "cmap": "viridis_r"},
    "temperature_max": {"label": "Max Temperature", "unit": "\u00b0C", "cmap": "jet"},
    "temperature_min": {"label": "Min Temperature", "unit": "\u00b0C", "cmap": "jet"},
    "wind": {"label": "Wind Speed", "unit": "km/h", "cmap": "Greens"},
    "humidity": {"label": "Humidity", "unit": "%", "cmap": "PuBu"},
}


# ---------------------------------------------------------------- discovery

def find_latest_nc_file(directory="."):
    """
    Finds the most recent ecmwf_aifs_india_YYYYMMDD_00z_merged.nc file in
    `directory` (by the YYYYMMDD in the filename). Returns None if none found.
    """
    pattern = os.path.join(directory, "ecmwf_aifs_india_*_00z_merged.nc")
    candidates = []
    for path in glob.glob(pattern):
        m = re.search(r"ecmwf_aifs_india_(\d{8})_00z_merged\.nc$", os.path.basename(path))
        if m:
            candidates.append((m.group(1), path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def parse_ic_from_filename(nc_path):
    """
    Extract the model initialization (IC) time in UTC from a filename like
    'ecmwf_aifs_india_20260907_00z_merged.nc'.
    """
    fname = os.path.basename(nc_path)
    m = re.search(r"(\d{8})_(\d{2})z", fname)
    if not m:
        raise ValueError(f"Could not parse IC date/hour from filename: {fname}")
    date_str, hour_str = m.groups()
    return datetime.strptime(date_str + hour_str, "%Y%m%d%H")


# ---------------------------------------------------------------- loading

def open_forecast(nc_path, tp_is_cumulative=TP_IS_CUMULATIVE):
    """Opens the NetCDF file, attaches UTC/IST valid-time coords, and fixes
    `tp` if it's accumulated-since-init (see module docstring)."""
    ds = xr.open_dataset(nc_path)

    try:
        ic_utc = parse_ic_from_filename(nc_path)
    except ValueError:
        run_date = ds.attrs.get("run_date")
        run_hour = ds.attrs.get("run_hour", "00z").replace("z", "")
        if not run_date:
            raise
        ic_utc = datetime.strptime(f"{run_date}{run_hour}", "%Y%m%d%H")

    n_steps = ds.sizes["time"]
    step_hours = np.arange(n_steps) * STEP_HOURS
    valid_utc = pd.to_datetime([ic_utc + timedelta(hours=int(h)) for h in step_hours])
    valid_ist = valid_utc + IST_OFFSET

    if tp_is_cumulative and "tp" in ds.data_vars:
        tp_vals = ds["tp"].values
        tp_interval = np.empty_like(tp_vals)
        tp_interval[0] = tp_vals[0]
        tp_interval[1:] = np.diff(tp_vals, axis=0)
        tp_interval = np.clip(tp_interval, 0, None)
        ds["tp_cumulative"] = (ds["tp"].dims, tp_vals)
        ds["tp"] = (ds["tp"].dims, tp_interval.astype(np.float32))

    ds = ds.assign_coords(
        valid_time_utc=("time", valid_utc),
        valid_time_ist=("time", valid_ist),
    )
    ds.attrs["ic_utc"] = ic_utc.isoformat()
    ds.attrs["nc_path"] = nc_path
    return ds


def get_point_timeseries(ds, lat, lon):
    """Bilinearly interpolates all variables to a point; returns a DataFrame."""
    lat_min, lat_max = float(ds["latitude"].min()), float(ds["latitude"].max())
    lon_min, lon_max = float(ds["longitude"].min()), float(ds["longitude"].max())
    if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
        raise ValueError(
            f"Point ({lat:.3f}, {lon:.3f}) is outside the forecast domain "
            f"(lat {lat_min}-{lat_max}, lon {lon_min}-{lon_max})."
        )
    point = ds.interp(latitude=lat, longitude=lon, method="linear")
    return point.to_dataframe().reset_index()


# ------------------------------------------------------------ classification

def classify_rainfall(mm, hours=24):
    """IMD-style rainfall category, thresholds scaled to the given window."""
    scale = hours / 24.0
    for limit, label in _RAIN_STEPS_24H:
        if mm <= limit * scale:
            return label
    return "Extremely heavy rain"


def category_at_least(category, threshold_word):
    key = _THRESHOLD_ALIASES.get((threshold_word or "").strip().lower(), "Heavy rain")
    if category not in RAIN_CATEGORY_ORDER or key not in RAIN_CATEGORY_ORDER:
        return False
    return RAIN_CATEGORY_ORDER.index(category) >= RAIN_CATEGORY_ORDER.index(key)


def classify_cloud(pct):
    if pct <= 10:
        return "clear skies"
    if pct <= 40:
        return "mostly clear skies"
    if pct <= 70:
        return "partly cloudy skies"
    if pct <= 90:
        return "generally cloudy skies"
    return "overcast skies"


def classify_humidity(pct):
    if pct < 40:
        return "low"
    if pct < 70:
        return "moderate"
    return "high"


def wind_dir_label(u, v):
    """Meteorological 'from' direction, 8-point compass label."""
    deg = (270 - np.degrees(np.arctan2(v, u))) % 360
    dirs = [
        "Northerly", "North-Easterly", "Easterly", "South-Easterly",
        "Southerly", "South-Westerly", "Westerly", "North-Westerly",
    ]
    idx = int(((deg + 22.5) % 360) // 45)
    return dirs[idx], deg


def get_daypart(hour):
    if 4 <= hour < 10:
        return "Morning"
    if 10 <= hour < 16:
        return "Afternoon"
    if 16 <= hour < 20:
        return "Evening"
    return "Night"


# ------------------------------------------------------------ aggregation

def aggregate_day(df, target_date, only_future=False, now=None):
    """Rows of a point time series falling on target_date (IST calendar date).
    If only_future, drops rows before `now` (used for "today" forecasts so
    only the remaining part of the day is reported)."""
    day_df = df[df["valid_time_ist"].dt.date == target_date].copy()
    if only_future:
        now = now or now_ist()
        day_df = day_df[day_df["valid_time_ist"] >= now]
    if day_df.empty:
        return None
    day_df["daypart"] = day_df["valid_time_ist"].dt.hour.apply(get_daypart)
    return day_df


def summarize_period(sub_df):
    tp_total = float(sub_df["tp"].sum())
    tmax = float((sub_df["t2m"] - 273.15).max())
    tmin = float((sub_df["t2m"] - 273.15).min())
    tcc_avg = float(sub_df["tcc"].mean())
    q_avg = float(sub_df["q1000"].mean())
    u_avg = float(sub_df["u10"].mean())
    v_avg = float(sub_df["v10"].mean())
    wind_speed_kmh = float(np.hypot(u_avg, v_avg) * 3.6)
    wind_label, wind_deg = wind_dir_label(u_avg, v_avg)
    hours = len(sub_df) * STEP_HOURS
    return {
        "tp_mm": round(tp_total, 1),
        "tmax_c": round(tmax, 1),
        "tmin_c": round(tmin, 1),
        "cloud_pct": round(tcc_avg, 1),
        "humidity_pct": round(q_avg, 1),
        "wind_kmh": round(wind_speed_kmh, 1),
        "wind_dir": wind_label,
        "rain_category": classify_rainfall(tp_total, hours=hours),
        "hours": hours,
    }


def should_split_daypart(day_df):
    parts = day_df.groupby("daypart")["tp"].sum()
    if len(parts) < 2:
        return False
    cats = {classify_rainfall(v, hours=STEP_HOURS) for v in parts.values}
    return len(cats) > 1 and parts.max() >= 3.0  # heuristic: meaningful mm in a 6h window


def format_period_text(label, stats, show_minmax=True):
    cloud_desc = classify_cloud(stats["cloud_pct"])
    humidity_desc = classify_humidity(stats["humidity_pct"])
    if stats["rain_category"] == "No rain":
        rain_phrase = "No significant rainfall is expected"
    else:
        rain_phrase = f"{stats['rain_category']} of {stats['tp_mm']:.0f} mm is likely"

    text = f"{label}: {rain_phrase} with {cloud_desc}. "
    if show_minmax:
        text += (
            f"Maximum temperature may be around {stats['tmax_c']:.0f}\u00b0C, "
            f"while minimum temperature may be around {stats['tmin_c']:.0f}\u00b0C. "
        )
    else:
        avg_t = (stats["tmax_c"] + stats["tmin_c"]) / 2
        text += f"Temperature may be around {avg_t:.0f}\u00b0C. "
    text += (
        f"Winds will be from the {stats['wind_dir']} direction at around "
        f"{stats['wind_kmh']:.0f} km/h. "
        f"Humidity will remain {humidity_desc} at around {stats['humidity_pct']:.0f}%."
    )
    return text


def build_daily_forecast_text(day_label, df, target_date):
    """Full natural-language forecast for one calendar day. If target_date is
    today (IST), only the remaining part of the day is reported, split into
    Morning/Afternoon/Evening/Night if conditions vary meaningfully."""
    now = now_ist()
    is_today = target_date == now.date()
    day_df = aggregate_day(df, target_date, only_future=is_today, now=now)

    if day_df is None:
        if is_today:
            return "No further forecast periods remain for today (all 6-hourly windows have passed)."
        return None

    prefix = ""
    if is_today and len(day_df) < 4:
        prefix = "_(Showing the remaining forecast periods for today.)_\n\n"

    if should_split_daypart(day_df):
        lines = []
        for part in DAYPART_ORDER:
            part_df = day_df[day_df["daypart"] == part]
            if part_df.empty:
                continue
            stats = summarize_period(part_df)
            lines.append(format_period_text(f"{day_label} {part}", stats, show_minmax=False))
        return prefix + "\n\n".join(lines)

    stats = summarize_period(day_df)
    return prefix + format_period_text(day_label, stats, show_minmax=True)


def build_compact_day_summary(df, target_date):
    """Full-day stats dict (not filtered to remaining periods), used for
    multi-day tables/charts where each day should show its own complete total."""
    day_df = aggregate_day(df, target_date)
    if day_df is None:
        return None
    return summarize_period(day_df)


def build_multiday_table(df, dates):
    """IMD-style outlook table: Date, Day, Rain Category, Rainfall (mm),
    Max/Min Temp, Wind, Humidity -- one row per day. The first row (if it's
    today) reflects only the remaining periods of today."""
    now = now_ist()
    today = now.date()
    rows = []
    for d in dates:
        is_today = d == today
        day_df = aggregate_day(df, d, only_future=is_today, now=now)
        label = day_label_for_date(d, today)
        if day_df is None:
            rows.append({
                "Date": d.strftime("%d-%m-%Y"), "Day": label,
                "Rain Category": "No data", "Rainfall (mm)": None,
                "Max Temp (\u00b0C)": None, "Min Temp (\u00b0C)": None,
                "Wind": None, "Humidity (%)": None,
            })
            continue
        stats = summarize_period(day_df)
        rows.append({
            "Date": d.strftime("%d-%m-%Y"),
            "Day": label + (" (remaining)" if is_today and len(day_df) < 4 else ""),
            "Rain Category": stats["rain_category"],
            "Rainfall (mm)": stats["tp_mm"],
            "Max Temp (\u00b0C)": stats["tmax_c"],
            "Min Temp (\u00b0C)": stats["tmin_c"],
            "Wind": f"{stats['wind_dir']} {stats['wind_kmh']:.0f} km/h",
            "Humidity (%)": stats["humidity_pct"],
        })
    return pd.DataFrame(rows)
