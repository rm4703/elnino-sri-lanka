# Deploying to GitHub → Streamlit Community Cloud

This app needs **no API keys** — every data source (NOAA ONI, CHIRPS, MODIS,
geoBoundaries) is public and key-less, so deployment is straightforward.

---

## 1. Prerequisites

- A free **GitHub** account.
- A free **Streamlit Community Cloud** account (sign in at
  <https://share.streamlit.io> with your GitHub login).
- `git` installed locally. (Optional: the GitHub CLI, `gh`.)

Files the deployment relies on (already in this project):

| File | Purpose |
|------|---------|
| `app.py` | the Streamlit entry point |
| `requirements.txt` | pip dependencies (all have Linux wheels) |
| `.streamlit/config.toml` | theme + minimal toolbar |
| `enso_lk/` | the analysis package |
| `.gitignore` | excludes `.cache/`, virtualenvs, etc. |

> **Note:** `.cache/` is git-ignored, so the cloud app starts with an empty
> cache and downloads CHIRPS on first load (~1–2 min behind the status spinner).
> See *Tips* below if you want to pre-bundle the cache.

---

## 2. Push the project to GitHub

### Option A — GitHub CLI (fastest)

```bash
cd "/media/dinesh/Local Disk/Antigravity"
git init
git add .
git commit -m "El Niño × Sri Lanka impact dashboard"
gh repo create elnino-sri-lanka --public --source=. --remote=origin --push
```

### Option B — plain git + GitHub website

1. On GitHub click **New repository** → name it e.g. `elnino-sri-lanka` →
   **Create repository** (leave it empty, no README).
2. Locally:

```bash
cd "/media/dinesh/Local Disk/Antigravity"
git init
git add .
git commit -m "El Niño × Sri Lanka impact dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/elnino-sri-lanka.git
git push -u origin main
```

Confirm on GitHub that `app.py` and `requirements.txt` are at the repo root.

---

## 3. Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and **sign in with GitHub** (authorise it
   to read your repos the first time).
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `<your-username>/elnino-sri-lanka`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Open **Advanced settings** and set **Python version → 3.12**
   (matches the local environment; 3.11+ also works). No secrets are needed.
5. Click **Deploy**. The first build installs the dependencies and the first
   page load fetches the live data — give it a couple of minutes.

Your app gets a public URL like
`https://elnino-sri-lanka.streamlit.app`.

---

## 4. Updating the live app

Streamlit Cloud auto-redeploys on every push to the deployed branch:

```bash
git add -A
git commit -m "Update: <what changed>"
git push
```

Use the app's **⋮ → Reboot** in the Streamlit Cloud dashboard if you change
dependencies and want a clean rebuild.

---

## Tips & gotchas

- **Cold-start speed.** To make the cloud app fast on first load you can bundle
  the data cache: remove the `.cache/` line from `.gitignore`, then
  `git add .cache && git commit`. This adds a few MB (CHIRPS) plus the MODIS
  NDVI JSONs — handy, but it bloats the repo. Leaving it ignored keeps the repo
  small at the cost of a slower first load.
- **Resource limits.** Community Cloud gives ~1 GB RAM; this app stays well
  under that. Heavy work (CHIRPS, MODIS) is cached after first use.
- **No secrets required.** If you later add a keyed data source, put keys in the
  Streamlit Cloud **app → Settings → Secrets** (TOML), never in the repo.
- **Outbound network.** The app calls NOAA, the IRI Data Library (CHIRPS), the
  ORNL DAAC (MODIS) and GitHub (geoBoundaries) — all allowed on Streamlit Cloud.
- **Python deps are pip-only** (no conda needed in the cloud); `requirements.txt`
  is all that's read.
