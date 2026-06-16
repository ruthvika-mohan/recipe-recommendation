# Garden Guide

Garden Guide is a recipe recommendation app with a FastAPI backend and a React/Tailwind frontend.

The backend keeps the existing TF-IDF/vectorizer artifacts, but improves ranking by enforcing diet as a hard filter, treating course as a strong filter, scoring ingredient coverage separately, and layering feedback boosts on top.

## Data Catalog

The app loads `data/final_ingredient_list_cleaned.csv` when present. That cleaned catalog keeps the same row count and order as `data/final_ingredient_list_created.csv` so `concatenated_features.pkl` stays aligned with recipe ids.

To rebuild it after changing raw data or cleaning rules:

```bash
python scripts/build_clean_catalog.py
```

The rebuild also writes `reports/clean_catalog_changes.csv`, which lists the diet and ingredient-list changes applied to the raw catalog.

## Local Development

Backend:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## API

- `GET /health`
- `GET /metadata`
- `POST /recommendations`
- `GET /recipes/{recipe_id}`
- `POST /feedback`

## Supabase

Run `supabase/schema.sql` in Supabase SQL editor, then set:

- Backend Render env:
  - `GARDEN_SUPABASE_URL`
  - `GARDEN_SUPABASE_SERVICE_ROLE_KEY`
  - `GARDEN_ALLOWED_ORIGINS`
- Frontend Vercel env:
  - `VITE_API_BASE_URL`
  - `VITE_SUPABASE_URL`
  - `VITE_SUPABASE_ANON_KEY`

Without Supabase env vars, local development uses a browser demo identity and an in-memory feedback store.

## Deploy

Backend on Render:

```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
```

Frontend on Vercel:

```bash
cd frontend
npm install
npm run build
```

Set `VITE_API_BASE_URL` to the Render API URL.
