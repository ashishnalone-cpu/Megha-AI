# Deploying MEGHA-AI (Rainfall Forecast Assistant) on Streamlit Community Cloud

## 1. Files to upload to GitHub

Put all of these at the **root** of the repo (flat, no subfolders needed --
the code already uses relative paths like `"IN.txt"`, `"IND-DIS-732.json"`,
and auto-detects the `.nc` file in the current directory):

```
app.py
date_utils.py
nc_data.py
districts.py
region_plot.py
geo_places.py
requirements.txt
packages.txt
.gitignore
.streamlit/config.toml
.streamlit/secrets.toml.example      (template only -- see step 4)

IND-DIS-732.json
IN.txt
ecmwf_aifs_india_20260920_00z_merged.nc
```

**Do not upload:**
- `Local-Agent-2.ipynb` -- it was only a way to run things locally; Streamlit
  Cloud runs `app.py` directly and doesn't need the notebook at all.
- `main.py` (if you have it) -- the console-only version, not used by the web app.
- `.streamlit/secrets.toml` (the *real* one, if you make one for local testing)
  -- never commit real API keys. It's already in `.gitignore`.

### A note on file sizes
GitHub hard-rejects any single file over 100 MB (and warns above ~50 MB).
Check yours before pushing:
```bash
ls -lh ecmwf_aifs_india_20260920_00z_merged.nc IN.txt IND-DIS-732.json
```
A full-India domain at the resolution you described (129 x 137 x 61 steps x
6 variables, float32) is roughly 25 MB, and a full GeoNames `IN.txt` /
732-district GeoJSON are typically a few MB to a few tens of MB -- all
normally fine as regular files. If any single file is over 100 MB, use
[Git LFS](https://git-lfs.com/) for that file instead of a plain `git add`.

## 2. The one required code change: the LLM backend

Your app currently uses `OllamaLLM` for parsing natural-language questions
into structured intent. **Ollama needs a local model server running on your
machine (`ollama serve`) -- there is no equivalent on Streamlit Community
Cloud**, so this is the one part of the app that cannot run online as-is.

I've patched `app.py` (attached) so `get_model()` picks the LLM backend from
Streamlit secrets / environment variables:

- `LLM_PROVIDER` unset, or `"ollama"` -> local Ollama (dev machine only)
- `LLM_PROVIDER = "anthropic"` -> Anthropic Claude API (needs `ANTHROPIC_API_KEY`)
- `LLM_PROVIDER = "openai"` -> OpenAI API (needs `OPENAI_API_KEY`)

Nothing else in the app changes -- `parse_intent()` now runs the same chain
through `StrOutputParser()` so it gets a plain string back regardless of
which provider answered. I tested that both the Ollama and Anthropic
branches construct correctly (see the conversation for the test output);
I couldn't call a real API from here without a key, so please do one live
smoke test after deploying.

I also added a small `sys.path` safety line to the top of each module, so
imports resolve correctly no matter what working directory the process
was started from -- this is what fixed the "ModuleNotFoundError: No module
named 'date_utils'" you hit earlier, and it's cheap insurance for any
hosting environment.

## 3. requirements.txt / packages.txt

`requirements.txt` (attached) adds `langchain-anthropic` and
`langchain-openai` alongside your existing packages, and keeps
`langchain-ollama` too (harmless to install; only used if you ever set
`LLM_PROVIDER=ollama` from your own machine against this same repo).

`packages.txt` (attached) lists system libraries (`gdal-bin`, `libgdal-dev`,
`libgeos-dev`, `libproj-dev`) for Streamlit Cloud to `apt install` before
your Python packages build. Modern `geopandas`/`shapely` wheels usually
bundle what they need, so you may not strictly need this -- but it's cheap
insurance against a build failure on the geospatial stack, which is the
most common thing that breaks on first deploy.

## 4. Setting your API key (secrets)

**Never commit a real API key to GitHub.** Two places to put it instead:

- **On Streamlit Community Cloud:** after creating the app, go to
  **App settings -> Secrets** and paste:
  ```toml
  LLM_PROVIDER = "anthropic"
  ANTHROPIC_API_KEY = "sk-ant-your-real-key"
  ```
  Save -- the app restarts automatically with these available via `st.secrets`.

- **For local testing of the same cloud path:** copy
  `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` (already
  gitignored) and fill in the real key there instead.

## 5. Step-by-step: GitHub -> Streamlit Cloud

1. Create a new GitHub repository (public or private both work).
2. Add all the files listed in step 1 to the repo root, commit, push.
   ```bash
   git init
   git add .
   git commit -m "Initial deploy: MEGHA-AI rainfall forecast assistant"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
3. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
4. Click **New app**, pick your repo/branch, and set **Main file path** to
   `app.py`.
5. Before clicking Deploy, open **Advanced settings** and paste your secrets
   (step 4) into the Secrets box.
6. Click **Deploy**. First build takes a few minutes (installing
   `geopandas`'s dependency chain is the slow part).
7. Once it's up, do a smoke test: ask it *"weather in Pune"* or whatever
   city you have in your `IN.txt`, and confirm the LLM call actually reaches
   Anthropic/OpenAI (check for a clear error banner if the key is missing --
   the patched `get_model()` calls `st.error(...)` + `st.stop()` rather than
   crashing opaquely if so).

## 6. Operational limitation worth knowing

This deploys **one fixed forecast file** (`ecmwf_aifs_india_20260920_00z_merged.nc`).
Streamlit Cloud only reads what's in the repo -- it won't automatically fetch
a newer IC run for you. When a new forecast file is generated, you'll need to
either:
- replace the `.nc` file in the repo and push again (simplest), or
- change the app to download the latest file from external storage
  (S3/GCS/a URL) at startup instead of reading a committed file -- a
  reasonable next step if you want this genuinely "live," but out of scope
  for just getting the current app online, so I haven't implemented it here.

## 7. Free-tier resource note

Streamlit Community Cloud's free tier caps each app around 1 GB RAM. The
one-time district-to-grid spatial join (732 districts x ~17.6k grid cells)
and the ~25 MB NetCDF should comfortably fit, and `st.cache_resource` means
that join only runs once per app instance rather than per request -- but if
you see the app get OOM-killed under real usage, that's the first place
to look.
