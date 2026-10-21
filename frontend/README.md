# CommInt frontend

The React workspace discovers communities and public discussion evidence for a
product or problem description. It also surfaces alternative products and can
draft evidence-based, help-first outreach for review.

## Run locally

Start the backend at `http://127.0.0.1:8000` (see [`../backend/README.md`](../backend/README.md)),
then run:

```powershell
npm install
npm run dev
```

Vite proxies `/api` to the local backend. For a separately hosted API, set
`VITE_API_BASE_URL` in a local `.env` file. `VITE_API_PROXY_TARGET` can override
the local proxy target. Only the API base URL belongs in frontend environment
variables; Supabase credentials stay in `backend/.env`.

## Supabase cache

Supabase is optional and is used only for sharing analysis-cache entries across
backend restarts or instances. Configure it using the instructions in the
[backend setup guide](../backend/README.md#optional-supabase-cache). The header
shows the current API and cache connection status.
