# Rover Learn

## Phase 1: Run

### Terminal 1 (backend)
```
uvicorn backend.app:app --host 0.0.0.0 --port 4000 --reload
```

### Terminal 2 (frontend)
```
cd frontend
npm run dev
```

Open <http://localhost:3000>

## Phase 2: Sessions & Latency

- Live page has a title input before starting a session.
- A status badge shows **● Live** while streaming and **● Stopped** after stopping.
- Top-right latency indicator displays milliseconds since the last segment; it turns orange when over 2000 ms.

### API

`POST /sessions/start` →
```
{ _id, title, createdAt, updatedAt, status, segmentsCount }
```

`GET /sessions` returns for each session:
```
{ _id, title, createdAt, updatedAt, status, segmentsCount }
```

## Phase 3: Exports & Glossary

### Exports

From Session detail, click **Export** → files are written under
`exports/sessions/<date>/<id>/`.

- `transcript_src.txt` – concatenated `textSrc`
- `translation_en.txt` – concatenated `textEn`
- `segments.jsonl` – one JSON segment per line

### Glossary

View terms at `/glossary`. Source: `config/glossary.csv`.

## Phase 4: Translator

Phase 4 adds a local Marian de→en translator. Start it via:

```
uvicorn services.mt.server:app --host 0.0.0.0 --port 4002 --reload
```

On first run, the model will be downloaded to your HF cache.

### Dev run order reminder

1. llama-server (optional / unused this phase)
2. `uvicorn services.mt.server:app --port 4002`
3. `uvicorn backend.app:app --port 4000`
4. `cd frontend && npm run dev`

## Phase 5: ASR stub

Start a tiny ASR service:

```
uvicorn services.asr.server:app --host 0.0.0.0 --port 4001 --reload
```

### Dev run order reminder

1. `uvicorn services.asr.server:app --port 4001`
2. `uvicorn services.mt.server:app --port 4002`
3. `uvicorn backend.app:app --port 4000`
4. `cd frontend && npm run dev`
