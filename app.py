"""
app.py

Interactive rainfall/weather forecast chat, served as a local web page.

Run with:
    streamlit run app.py

Streamlit will print a local address (usually http://localhost:8501)
that opens in your browser.
"""

import json
import os
import re
import sys

import matplotlib.pyplot as plt
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Make sure this file's own directory is importable regardless of the
# working directory the process was launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nc_data import (
    open_forecast, get_point_timeseries, build_daily_forecast_text,
    build_compact_day_summary, build_multiday_table, category_at_least,
    find_latest_nc_file, ensure_nc_file, describe_directory, VARIABLE_META,
)
from districts import (
    load_district_gdf, build_grid_district_index, district_rainfall_table,
    state_rainfall_table, district_full_table, summarize_district_table,
    is_india_scope,
)
from region_plot import plot_region_pattern, plot_region_spatial
from geo_places import IndiaGeoAgent
from date_utils import resolve_date_term, day_label_for_date, get_date_range, now_ist

# --------------------------------------------------------------- CONFIG
# NC_PATH=None -> auto-detect the newest ecmwf_aifs_india_YYYYMMDD_00z_merged.nc
# in NC_DIR. Set NC_PATH explicitly to override auto-detection.
NC_DIR = "."
NC_PATH = None
GEONAMES_PATH = "IN.txt"
DISTRICT_GEOJSON_PATH = "IND-DIS-732.json"

st.set_page_config(page_title="MEGHA-AI", page_icon="\U0001f326\ufe0f", layout="wide")

PLOT_VAR_META = {
    "rainfall": {"ylabel": "Rainfall (mm)", "title": "Daily Rainfall", "color": "#2b6cb0"},
    "temperature_max": {"ylabel": "\u00b0C", "title": "Max Temperature", "color": "#e53e3e"},
    "temperature_min": {"ylabel": "\u00b0C", "title": "Min Temperature", "color": "#3182ce"},
    "wind": {"ylabel": "km/h", "title": "Wind Speed", "color": "#38a169"},
    "humidity": {"ylabel": "%", "title": "Humidity", "color": "#805ad5"},
}


# --------------------------------------------------------------- cached resources

def _get_secret(key, default=None):
    """Reads from Streamlit secrets if available, else falls back to env vars.
    st.secrets raises if no secrets.toml exists at all, so this is wrapped."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


@st.cache_resource
def get_model():
    """
    Local dev (Ollama running on your machine): leave LLM_PROVIDER unset,
    or set it to "ollama". Requires `ollama serve` running with the
    `llama3.2` model pulled -- this only works on a machine you control,
    NOT on Streamlit Community Cloud (no local model server there).

    Cloud deployment (Streamlit Community Cloud or similar): set
    LLM_PROVIDER="anthropic" and ANTHROPIC_API_KEY="sk-ant-..." in the
    app's Secrets (Streamlit Cloud -> App settings -> Secrets), or as
    environment variables for another host.
    """
    provider = (_get_secret("LLM_PROVIDER", "ollama") or "ollama").lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = _get_secret("ANTHROPIC_API_KEY")
        if not api_key:
            st.error(
                "LLM_PROVIDER is set to 'anthropic' but ANTHROPIC_API_KEY is "
                "missing from Streamlit secrets. Add it under App settings -> Secrets."
            )
            st.stop()
        return ChatAnthropic(model="claude-3-5-haiku-20241022", api_key=api_key, temperature=0)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = _get_secret("OPENAI_API_KEY")
        if not api_key:
            st.error(
                "LLM_PROVIDER is set to 'openai' but OPENAI_API_KEY is "
                "missing from Streamlit secrets. Add it under App settings -> Secrets."
            )
            st.stop()
        return ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)

    # default: local Ollama (dev machine only)
    from langchain_ollama.llms import OllamaLLM
    return OllamaLLM(model="llama3.2")


@st.cache_resource
def get_forecast():
    download_url = _get_secret("NC_FILE_URL")  # optional external fallback, see README
    path = ensure_nc_file(nc_dir=NC_DIR, nc_path=NC_PATH, download_url=download_url)
    return open_forecast(path)


@st.cache_resource
def get_geo_agent():
    return IndiaGeoAgent(GEONAMES_PATH)


@st.cache_resource
def get_district_data():
    gdf = load_district_gdf(DISTRICT_GEOJSON_PATH)
    ds = get_forecast()
    id_grid = build_grid_district_index(ds, gdf)
    return gdf, id_grid


# --------------------------------------------------------------- intent parsing

INTENT_TEMPLATE = """
You are an intent parser for a rainfall/weather forecast assistant covering India.
Classify the question into JSON with these fields:
- "intent": one of "city_forecast", "state_summary", "state_district_table",
  "district_rain_query", "state_threshold_query", "region_plot", "other"
  * city_forecast: forecast for a city, village, town, or landmark (may not
    exist in the district file -- that's fine, it's geocoded separately)
  * state_summary: a short overall rainfall narrative for one state
  * state_district_table: a full weather TABLE (rainfall, temp, wind,
    humidity) for every district of a state
  * district_rain_query: which districts of a state meet a rainfall threshold
  * state_threshold_query: which states nationally meet a rainfall threshold
  * region_plot: a map/plot for a state or for all of India
- "location": the city/village/landmark/state name, or "India" for a
  country-wide region_plot, or null
- "date": one of "today", "tomorrow", "day after tomorrow", "yesterday",
  an explicit date (any format, e.g. "15 August", "15-08-2026"), or null
  if no date is mentioned. Never compute the actual date yourself -- pass
  the word/phrase through literally. If no date is mentioned, use null
  (the app defaults to today).
- "range_days": integer if the user asks for a multi-day range like
  "next 7 days" or "next 15 days", else null
- "threshold": one of "light", "moderate", "heavy", "very heavy" if the
  user names a minimum rainfall level, else null
- "plot": true if the user explicitly asks to plot/chart/graph/map
  something, else false
- "plot_variables": a list with any of "rainfall", "temperature", "wind",
  "humidity" that the user wants plotted. Default ["rainfall"] if plot is
  true but nothing specific is named.
- "plot_style": "spatial" if the user says "spatial plot" or "district-wise"
  / "districtwise", else "pattern" (a smooth gradient map). Only relevant
  when intent is "region_plot".

Only output valid JSON. No explanation, no markdown fences.

Examples:
Q: "What is the rainfall forecast for Pune?"
A: {{"intent":"city_forecast","location":"Pune","date":null,"range_days":null,"threshold":null,"plot":false,"plot_variables":["rainfall"],"plot_style":"pattern"}}

Q: "What's the weather right now?"
A: {{"intent":"city_forecast","location":null,"date":null,"range_days":null,"threshold":null,"plot":false,"plot_variables":["rainfall"],"plot_style":"pattern"}}

Q: "Will it rain in Mumbai tomorrow?"
A: {{"intent":"city_forecast","location":"Mumbai","date":"tomorrow","range_days":null,"threshold":null,"plot":false,"plot_variables":["rainfall"],"plot_style":"pattern"}}

Q: "Weather in Nashik on 15 August"
A: {{"intent":"city_forecast","location":"Nashik","date":"15 August","range_days":null,"threshold":null,"plot":false,"plot_variables":["rainfall"],"plot_style":"pattern"}}

Q: "Show me rainfall forecast for Delhi for the next 15 days."
A: {{"intent":"city_forecast","location":"Delhi","date":null,"range_days":15,"threshold":null,"plot":false,"plot_variables":["rainfall"],"plot_style":"pattern"}}

Q: "Plot temperature and rainfall for Bengaluru for the next 7 days."
A: {{"intent":"city_forecast","location":"Bengaluru","date":null,"range_days":7,"threshold":null,"plot":true,"plot_variables":["temperature","rainfall"],"plot_style":"pattern"}}

Q: "What is the rainfall situation in Maharashtra?"
A: {{"intent":"state_summary","location":"Maharashtra","date":"today","range_days":null,"threshold":null,"plot":false,"plot_variables":["rainfall"],"plot_style":"pattern"}}

Q: "Show weather of all districts in Maharashtra"
A: {{"intent":"state_district_table","location":"Maharashtra","date":"today","range_days":null,"threshold":null,"plot":false,"plot_variables":["rainfall"],"plot_style":"pattern"}}

Q: "Which districts of Maharashtra are expected to receive heavy rainfall?"
A: {{"intent":"district_rain_query","location":"Maharashtra","date":"today","range_days":null,"threshold":"heavy","plot":false,"plot_variables":["rainfall"],"plot_style":"pattern"}}

Q: "Show me the rainfall forecast around Mumbai Airport."
A: {{"intent":"city_forecast","location":"Mumbai Airport","date":null,"range_days":null,"threshold":null,"plot":false,"plot_variables":["rainfall"],"plot_style":"pattern"}}

Q: "Which states are expected to receive moderate or higher rainfall tomorrow?"
A: {{"intent":"state_threshold_query","location":null,"date":"tomorrow","range_days":null,"threshold":"moderate","plot":false,"plot_variables":["rainfall"],"plot_style":"pattern"}}

Q: "Show rainfall forecast map of Maharashtra."
A: {{"intent":"region_plot","location":"Maharashtra","date":"today","range_days":null,"threshold":null,"plot":true,"plot_variables":["rainfall"],"plot_style":"pattern"}}

Q: "Show temperature and wind map of Gujarat tomorrow"
A: {{"intent":"region_plot","location":"Gujarat","date":"tomorrow","range_days":null,"threshold":null,"plot":true,"plot_variables":["temperature","wind"],"plot_style":"pattern"}}

Q: "Show a spatial plot of rainfall for Maharashtra"
A: {{"intent":"region_plot","location":"Maharashtra","date":"today","range_days":null,"threshold":null,"plot":true,"plot_variables":["rainfall"],"plot_style":"spatial"}}

Q: "Show district-wise rainfall map of India"
A: {{"intent":"region_plot","location":"India","date":"today","range_days":null,"threshold":null,"plot":true,"plot_variables":["rainfall"],"plot_style":"spatial"}}

Q: "Plot rainfall map of India"
A: {{"intent":"region_plot","location":"India","date":"today","range_days":null,"threshold":null,"plot":true,"plot_variables":["rainfall"],"plot_style":"pattern"}}

Now classify:
Q: "{question}"
A:
"""
intent_prompt = ChatPromptTemplate.from_template(INTENT_TEMPLATE)

DEFAULT_PARSED = {
    "intent": "other", "location": None, "date": None, "range_days": None,
    "threshold": None, "plot": False, "plot_variables": ["rainfall"], "plot_style": "pattern",
}


class LLMBackendError(RuntimeError):
    """Raised when the configured LLM backend can't be reached, with an
    actionable message for the person running the app (not a raw traceback)."""


def parse_intent(question):
    # StrOutputParser normalizes output whether the model returns a plain
    # string (Ollama) or a chat message object (Anthropic/OpenAI).
    provider = (_get_secret("LLM_PROVIDER", "ollama") or "ollama").lower()
    chain = intent_prompt | get_model() | StrOutputParser()
    try:
        raw = chain.invoke({"question": question})
    except Exception as e:
        if provider == "ollama":
            raise LLMBackendError(
                "Could not reach a local Ollama server (provider is currently "
                "**'ollama'**, which only works on a machine where `ollama serve` "
                "is actually running -- it will never work on Streamlit Community "
                "Cloud or similar hosts).\n\n"
                "**Fix:** in this app's Streamlit Cloud secrets (App settings -> "
                "Secrets), add:\n```toml\nLLM_PROVIDER = \"anthropic\"\n"
                "ANTHROPIC_API_KEY = \"sk-ant-your-real-key\"\n```\n"
                "then reboot the app from the Streamlit Cloud dashboard so it "
                "picks up the new secrets."
            ) from e
        raise LLMBackendError(
            f"Could not reach the '{provider}' LLM backend: {e}\n\n"
            "Check that the matching API key secret is set correctly and that "
            "your account has access to the model, then reboot the app."
        ) from e
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return dict(DEFAULT_PARSED)
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return dict(DEFAULT_PARSED)

    merged = dict(DEFAULT_PARSED)
    merged.update({k: v for k, v in parsed.items() if v is not None})

    # sensible default: "plot a city forecast" with no explicit range -> 7 days
    if merged["intent"] == "city_forecast" and merged["plot"] and not merged.get("range_days"):
        merged["range_days"] = 7
    return merged


def expand_plot_variables(vars_list):
    """'temperature' -> both max and min; dedupe; default to rainfall."""
    out = []
    for v in vars_list or []:
        if v == "temperature":
            out += ["temperature_max", "temperature_min"]
        elif v in ("rainfall", "wind", "humidity", "temperature_max", "temperature_min"):
            out.append(v)
    seen, result = set(), []
    for v in out:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result or ["rainfall"]


# --------------------------------------------------------------- charts

def build_multiday_chart(df, dates, variables, location_label):
    n = len(variables)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.8 * n), sharex=True)
    if n == 1:
        axes = [axes]
    x_labels = [day_label_for_date(d) for d in dates]

    field_map = {
        "rainfall": "tp_mm", "temperature_max": "tmax_c", "temperature_min": "tmin_c",
        "wind": "wind_kmh", "humidity": "humidity_pct",
    }

    for ax, var in zip(axes, variables):
        y = []
        for d in dates:
            stats = build_compact_day_summary(df, d)
            y.append(stats[field_map[var]] if stats else None)
        meta = PLOT_VAR_META[var]
        ax.plot(x_labels, y, marker="o", linewidth=2, color=meta["color"])
        ax.set_ylabel(meta["ylabel"])
        ax.set_title(meta["title"])
        ax.grid(alpha=0.3)

    plt.xticks(rotation=30, ha="right")
    fig.suptitle(f"{len(dates)}-day outlook \u2014 {location_label.title()}")
    plt.tight_layout()
    return fig


# --------------------------------------------------------------- handlers
# Each handler returns a dict: {"text": str, "table": DataFrame|None, "figs": [Figure, ...]}

def _result(text, table=None, figs=None):
    return {"text": text, "table": table, "figs": figs or []}


def geocode(location):
    return get_geo_agent().get_lat_lon(location)


def handle_city_forecast(location, date_term, range_days, plot, plot_variables):
    if not location:
        location = "your location"
        return _result("Please tell me which city, village, or landmark you'd like the forecast for.")

    coords = geocode(location)
    if coords is None:
        return _result(f"Could not find '{location}' in the places database.")
    lat, lon = coords

    try:
        df = get_point_timeseries(get_forecast(), lat, lon)
    except ValueError as e:
        return _result(str(e))

    header = f"**{location}** (lat {lat:.3f}, lon {lon:.3f})\n\n"

    if range_days:
        dates = get_date_range(now_ist().date(), range_days)
        table = build_multiday_table(df, dates)
        figs = []
        if plot:
            variables = expand_plot_variables(plot_variables)
            figs = [build_multiday_chart(df, dates, variables, location)]
        return _result(header + f"{range_days}-day outlook (IMD categories):", table=table, figs=figs)

    target_date = resolve_date_term(date_term) or now_ist().date()
    if target_date is None:
        return _result(f"Couldn't understand the date '{date_term}'.")
    label = day_label_for_date(target_date)
    text = build_daily_forecast_text(label, df, target_date)
    if text is None:
        return _result(header + f"No forecast data available for {target_date}.")
    return _result(header + text)


def handle_state_summary(state_name, date_term):
    target_date = resolve_date_term(date_term) or now_ist().date()
    label = day_label_for_date(target_date)
    gdf, id_grid = get_district_data()

    stab = state_rainfall_table(get_forecast(), gdf, id_grid, target_date)
    if stab is None:
        return _result(f"No forecast data available for {target_date}.")
    row = stab[stab["state"].str.upper() == state_name.strip().upper()]
    if row.empty:
        return _result(f"No district data found for '{state_name}'.")

    dtab = district_rainfall_table(get_forecast(), gdf, id_grid, state_name, target_date)
    top = dtab.head(3)
    top_text = ", ".join(f"{r.district.title()} ({r.avg_rain_mm:.0f} mm)" for r in top.itertuples())

    avg_rain = row.iloc[0]["avg_rain_mm"]
    category = row.iloc[0]["category"]
    text = (
        f"**{state_name.title()}** \u2014 {label} ({target_date}): state-wide average is "
        f"**{category}** (~{avg_rain:.0f} mm). Highest expected rainfall: {top_text}."
    )
    return _result(text)


def handle_state_district_table(state_name, date_term):
    target_date = resolve_date_term(date_term) or now_ist().date()
    gdf, id_grid = get_district_data()
    table = district_full_table(get_forecast(), gdf, id_grid, target_date, state_name=state_name)
    if table is None or table.empty:
        return _result(f"No district data found for '{state_name}'.")
    summary = summarize_district_table(table, state_name.title(), target_date)
    display_table = table.drop(columns=["State", "Category"])
    return _result(summary, table=display_table)


def handle_district_query(state_name, date_term, threshold):
    threshold = (threshold or "heavy").lower()
    target_date = resolve_date_term(date_term) or now_ist().date()
    gdf, id_grid = get_district_data()

    table = district_rainfall_table(get_forecast(), gdf, id_grid, state_name, target_date)
    if table is None or table.empty:
        return _result(f"No district data found for '{state_name}'.")

    filtered = table[table["category"].apply(lambda c: category_at_least(c, threshold))]
    label = day_label_for_date(target_date)
    if filtered.empty:
        return _result(
            f"No districts in {state_name.title()} are expected to see "
            f"{threshold}-or-higher rainfall {label.lower()} ({target_date})."
        )
    lines = [
        f"Districts in **{state_name.title()}** expected to see **{threshold} or higher** "
        f"rainfall \u2014 {label} ({target_date}):\n"
    ]
    for r in filtered.itertuples():
        lines.append(f"- {r.district.title()}: {r.category} (~{r.avg_rain_mm:.0f} mm)")
    return _result("\n".join(lines))


def handle_state_threshold_query(date_term, threshold):
    threshold = (threshold or "moderate").lower()
    target_date = resolve_date_term(date_term) or now_ist().date()
    gdf, id_grid = get_district_data()

    table = state_rainfall_table(get_forecast(), gdf, id_grid, target_date)
    if table is None or table.empty:
        return _result("No forecast data available for that date.")

    filtered = table[table["category"].apply(lambda c: category_at_least(c, threshold))]
    label = day_label_for_date(target_date)
    if filtered.empty:
        return _result(f"No states are expected to see {threshold}-or-higher rainfall {label.lower()} ({target_date}).")
    lines = [f"States expected to see **{threshold} or higher** rainfall \u2014 {label} ({target_date}):\n"]
    for r in filtered.itertuples():
        lines.append(f"- {r.state}: {r.category} (~{r.avg_rain_mm:.0f} mm)")
    return _result("\n".join(lines))


def handle_region_plot(location, date_term, plot_variables, plot_style):
    target_date = resolve_date_term(date_term) or now_ist().date()
    gdf, id_grid = get_district_data()
    variables = expand_plot_variables(plot_variables)
    scope_label = "India" if is_india_scope(location) else location.title()
    label = day_label_for_date(target_date)

    plot_fn = plot_region_spatial if plot_style == "spatial" else plot_region_pattern
    figs = []
    for var in variables:
        try:
            figs.append(plot_fn(get_forecast(), gdf, id_grid, location, target_date, variable=var))
        except ValueError as e:
            return _result(str(e))

    style_word = "District-wise spatial" if plot_style == "spatial" else "Smoothed pattern"
    text = f"{style_word} map(s) for **{scope_label}** \u2014 {label} ({target_date})"
    return _result(text, figs=figs)


def route(parsed):
    intent = parsed.get("intent")
    location = parsed.get("location")
    date_term = parsed.get("date")
    range_days = parsed.get("range_days")
    threshold = parsed.get("threshold")
    plot = parsed.get("plot")
    plot_variables = parsed.get("plot_variables")
    plot_style = parsed.get("plot_style")

    if intent == "city_forecast":
        return handle_city_forecast(location, date_term, range_days, plot, plot_variables)
    if intent == "state_summary" and location:
        return handle_state_summary(location, date_term)
    if intent == "state_district_table" and location:
        return handle_state_district_table(location, date_term)
    if intent == "district_rain_query" and location:
        return handle_district_query(location, date_term, threshold)
    if intent == "state_threshold_query":
        return handle_state_threshold_query(date_term, threshold)
    if intent == "region_plot" and location:
        return handle_region_plot(location, date_term, plot_variables, plot_style)

    return _result(
        "I couldn't tell what you meant. Try something like:\n"
        "- 'weather in Pune' / 'rainfall forecast for Pune tomorrow'\n"
        "- 'rainfall for Delhi for the next 15 days'\n"
        "- 'weather of all districts in Maharashtra'\n"
        "- 'which districts of Maharashtra expect heavy rainfall'\n"
        "- 'which states expect moderate or higher rainfall tomorrow'\n"
        "- 'rainfall map of Maharashtra' / 'spatial plot of rainfall for India'"
    )


# --------------------------------------------------------------- Streamlit UI

st.title("\U0001f326\ufe0f MEGHA-AI")
st.title("(Multi-model Ensemble for Geospatial & Hyperlocal Agentic Atmospheric Intelligence)")
st.caption("Design and Developed By - Ashish Alone \n")
st.caption("Guided By - Prof. Anoop Kumar Shukla, Dr D. R. Pattanaik & Prof. Gopal Nandan")
st.caption(
    "Ask things like *'weather in Pune'*, *'rainfall for Delhi next 15 days'*, "
    "*'weather of all districts in Maharashtra'*, or *'spatial plot of rainfall for India'*."
)

with st.sidebar:
    st.markdown("**Assistant brain**")
    _active_provider = (_get_secret("LLM_PROVIDER", "ollama") or "ollama").lower()
    if _active_provider == "ollama":
        st.warning(
            "LLM_PROVIDER = 'ollama' (default). This only works if a local "
            "Ollama server is running on THIS machine -- it will always fail "
            "on Streamlit Community Cloud. Set LLM_PROVIDER + an API key "
            "under App settings -> Secrets to fix."
        )
    else:
        st.caption(f"Provider: {_active_provider}")

    st.markdown("**Forecast source**")
    try:
        ds = get_forecast()
        st.caption(f"File: {ds.attrs.get('nc_path', 'unknown')}")
        st.caption(f"Model init (UTC): {ds.attrs.get('ic_utc', 'unknown')}")
        st.caption(f"Current IST time: {now_ist().strftime('%Y-%m-%d %H:%M')}")
    except Exception as e:
        st.error("Could not load the NetCDF forecast file.")
        st.code(str(e))

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("table") is not None:
            st.dataframe(msg["table"], use_container_width=True)
        for fig in msg.get("figs", []):
            st.pyplot(fig)

question = st.chat_input("Ask about weather forecast... (Rainfall, Temperature, Wind etc..)")

if question:
    st.session_state.messages.append({"role": "user", "content": question, "table": None, "figs": []})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                parsed = parse_intent(question)
                result = route(parsed)
            except LLMBackendError as e:
                result = _result(str(e))
        st.markdown(result["text"])
        if result.get("table") is not None:
            st.dataframe(result["table"], use_container_width=True)
        for fig in result.get("figs", []):
            st.pyplot(fig)

    st.session_state.messages.append({
        "role": "assistant", "content": result["text"],
        "table": result.get("table"), "figs": result.get("figs", []),
    })
