# Enrichment Runbook

## Before you start
- Ensure `.env` includes:
  - `MONGO_URI` (or `MONGODB_URI`)
  - `JOBS_DB_NAME=refjobs`
  - `LLM_PROVIDER`, `LLAMA_BASE_URL`, `LLAMA_MODEL`
- Confirm llama.cpp is running at `$LLAMA_BASE_URL` with `$LLAMA_MODEL`.
- Run the **refresh** first so `last_seen_at` is current.

## What this step does
- Selects jobs where **`enriched_at` is missing** OR **`last_seen_at > enriched_at`**.
- For each selected job, generates and writes back:
  - `normalized_title`, `seniority`, `skills`, `remote_mode`, `lang`
  - `summary_tldr`, `responsibilities`, `requirements`
  - `salary.min|max|currency|period`, `salary.estimated`
  - `quality_score`, `spam_flag`
  - `enriched_at` timestamp

> Enrichment only `$set`s new fields. Provider data and `raw` are not overwritten.

## Run (manual)
1. Start/verify llama.cpp server is up.
2. Execute the enrichment process (your worker/script when added).
3. Monitor logs for batch progress and any JSON-parse skips.

## Verify in MongoDB Compass
- Filter enriched docs and sort by newest:
  ```json
  { "enriched_at": { "$exists": true } }
