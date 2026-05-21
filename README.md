# Timeries

Time series toolkit for hourly meter data: ingestion, cleaning, EDA, and forecasting (Prophet, NeuralProphet, SARIMAX).

## Install

```bash
pip install -e .
```

## Documentation

### Published site (GitHub Pages)

URL: `https://<your-github-username>.github.io/Timeries/`

1. Push to **`main`** or **`master`** (workflow runs on both).
2. Repo **Settings → Pages → Build and deployment → Source**: choose **GitHub Actions** (not “Deploy from a branch”).
3. After the [docs workflow](.github/workflows/docs.yml) succeeds, open the Pages URL above.

If you only see **README.md**, Pages is still serving the repo branch instead of the MkDocs build—switch the source to **GitHub Actions** and re-run the workflow (**Actions → Deploy docs to GitHub Pages → Run workflow**).

### Local docs (includes REST API)

The REST API page embeds Swagger at `http://127.0.0.1:8000/docs` and is **not** published to GitHub Pages.

```bash
pip install -e ".[docs]"
mkdocs serve -f mkdocs.local.yml -a 127.0.0.1:8001
```

Start your FastAPI app on port **8000**, then open the MkDocs URL (e.g. `http://127.0.0.1:8001`) and open **REST API** in the sidebar.

### Local docs (Python API only, matches GitHub)

```bash
mkdocs serve -a 127.0.0.1:8001
```

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/<your-username>/Timeries.git
git push -u origin master
# or: git branch -M main && git push -u origin main
```
