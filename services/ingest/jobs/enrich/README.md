# LLM Enrichment (Design Doc)

**Purpose:** read jobs from `refjobs.jobs`, add structured fields the UI + matcher need, and write them back to the same docs. This step is **idempotent** and **additive** (does not overwrite provider data).

## Inputs
- MongoDB: database `refjobs`, collection `jobs`
- Documents that meet the **selection rule**:
  - `enriched_at` is **missing**, or
  - `last_seen_at` **>** `enriched_at`

## Outputs (fields written back)
- Timestamps: `first_seen_at` (set once), `last_seen_at` (set by refresh), `enriched_at`
- Understanding: `normalized_title`, `seniority`, `skills[]`, `remote_mode`, `lang`
- Compensation: `salary.min|max|currency|period`, `salary.estimated` (true if inferred)
- UX helpers: `summary_tldr`, `responsibilities[]`, `requirements[]`
- Quality: `quality_score` (0–100), `spam_flag` (boolean)

> **Never** remove or rewrite `raw` and provider-native fields; enrichment **only** `$set`s the fields above.

## Environment (from `.env`)
- `MONGO_URI` (or `MONGODB_URI`)
- `JOBS_DB_NAME=refjobs`
- LLM:
  - `LLM_PROVIDER=LLAMACPP`
  - `LLAMA_BASE_URL=http://127.0.0.1:8080`
  - `LLAMA_MODEL=Phi-3.5-mini-instruct-Q5_K_S.gguf`

## Run Order
1. **Refresh** (fetch + upsert) — you already run this.
   - Also set:
     - `first_seen_at` if missing
     - `last_seen_at` to now
2. **Enrichment** (this step)
   - For each selected job: apply prompts → parse JSON → `$set` only enrichment fields
   - Set `enriched_at` to now

## Idempotency & Safety
- Only `$set` enrichment fields; **do not** `$unset` provider fields.
- If a prompt fails to parse JSON, **skip** the write for that job and continue.
- For salary: if not explicit in text, either leave blank or set `salary.estimated: true` and be conservative.

## Verification (manual)
- In MongoDB Compass:
  - Filter: `{"enriched_at": {"$exists": true}}`, sort by `enriched_at` desc
  - Inspect fields: `normalized_title`, `skills`, `remote_mode`, `summary_tldr`
- Counts by source:
  - `db.jobs.aggregate([{ $match: { enriched_at: { $exists: true } } }, { $group: { _id: "$source", n: { $count: {} } } }])`

## Performance & Limits
- Configure batch size and concurrency in `config/enrichment.yaml`.
- Respect llama.cpp throughput (start low: 2–4 concurrent jobs with small max tokens).
