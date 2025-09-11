# Enrichment Fields (Canonical Definitions)

## Timestamps
- **first_seen_at**: ISO string — set once when the job first appears.
- **last_seen_at**: ISO string — set by refresh each time the job is seen.
- **enriched_at**: ISO string — last time enrichment was written.

## Understanding
- **normalized_title**: short, canonical snake_case title (e.g., `software_engineer`, `data_scientist`).
- **seniority**: one of `intern|junior|mid|senior|lead|manager|director|vp|cxo|null`.
- **skills**: array of lowercase tokens; must-have skills should appear earlier.
- **remote_mode**: `remote|hybrid|onsite` (refined from text).
- **lang**: ISO 2-letter language code of the description (e.g., `en`, `de`).

## Compensation
- **salary.min|max**: numbers (same currency units).
- **salary.currency**: ISO-4217 (e.g., `USD`, `EUR`).
- **salary.period**: `year|month|week|day|hour|null`.
- **salary.estimated**: boolean (true if inferred).

## UX helpers
- **summary_tldr**: 3–4 factual lines for quick reading.
- **responsibilities**: 5–10 bullet strings.
- **requirements**: 5–10 bullet strings.

## Quality
- **quality_score**: integer 0–100 (see rubric.md).
- **spam_flag**: boolean (true only if clearly spam/irrelevant).
