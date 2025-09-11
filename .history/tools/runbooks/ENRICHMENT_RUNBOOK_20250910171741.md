# Enrichment Runbook

## Before you start
- Confirm `.env` has MONGO_URI (or MONGODB_URI) and JOBS_DB_NAME=refjobs
- LLM server up: llama.cpp at $LLAMA_BASE_URL with $LLAMA_MODEL
- Refresh has been run recently (so last_seen_at is up to date)

## What this step does
- Select jobs where enriched_at is missing OR last_seen_at > enriched_at
- For each job, run prompts and write back:
  normalized_title, seniority, skills, remote_mode, summary_tldr, responsibilities, requirements, salary.*, quality_score, spam_flag, enriched_at

## Run (manual)
1) Start/ensure llama.cpp is running locally.
2) Execute the enrichment process (your script/worker when you add it).
3) Watch logs for per-batch progress and parse skips.

## Verify in MongoDB Compass
- Filter: `{"enriched_at": {"$exists": true}}`, sort by `enriched_at` desc
- Spot-check fields: `normalized_title`, `skills[0..]`, `remote_mode`, `summary_tldr`
- Per-source counts:
