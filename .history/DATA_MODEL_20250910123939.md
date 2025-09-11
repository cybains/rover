# Unified Job Schema (v1)

This schema is the single normalized shape we store in MongoDB, no matter which provider we pull from. Preserve the original payload for traceability.

## Collection: `jobs`

### Document shape
```json
{
  "_id": "string",                       // stable fingerprint (see Identity)
  "source": "remotive|arbeitnow|jobicy|usajobs|jooble|adzuna",
  "source_id": "string|null",            // provider’s native id if present
  "title": "string",
  "company": {
    "name": "string|null",
    "domain": "string|null"
  },
  "locations": [
    {
      "type": "remote|onsite|hybrid",
      "city": "string|null",
      "region": "string|null",
      "country": "ISO-3166-1 alpha-2|null",
      "lat": null,
      "lon": null
    }
  ],
  "employment_type": "full_time|part_time|contract|internship|temp|null",
  "remote": true,
  "salary": {
    "min": null,
    "max": null,
    "currency": "ISO-4217|null",
    "period": "year|month|week|day|hour|null"
  },
  "description_html": "<p>…</p>",
  "tags": ["python","ml"],
  "apply_url": "https://…",
  "posted_at": "2025-09-03T12:34:56Z",
  "fetched_at": "2025-09-10T10:00:00Z",
  "duplicate_of": null,                  // if clustered as cross-source duplicate
  "raw": { /* original provider item */ }
}
