# Search API — Tridev Cyber Warrior

FastAPI + DuckDB search API streaming parquet indexes from HuggingFace.

## Deploy
1. Push this repo to GitHub.
2. Import the repo on Vercel → framework preset: **Other**.
3. Deploy. Endpoint: `https://<project>.vercel.app/search?q=<number>`

## Endpoints
- `GET /` — status
- `GET /health` — health
- `GET /search?q=<phone|aadhar>` — unified search
- `GET /search?q=X&field=phoneNumber&mode=exact` — field search
- `GET /fields` — supported fields

## Env vars (Vercel → Settings → Environment Variables)
- `ICMR_HF_INDEX_BASE` — HuggingFace bucket base URL
- `ICMR_THREADS` — DuckDB threads (default 2)
- `ICMR_TIMEOUT` — per-request timeout seconds (default 25)

## Notes
- Vercel serverless has a 60s max duration on Pro (10s on Hobby). Keep `limit` small.
- First request per cold start may be slow — indexes stream over HTTP.

**Develop by Tridev Cyber Warrior**
