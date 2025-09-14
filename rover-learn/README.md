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

## Phase 6: Live audio → ASR → Backend

```
# Terminal A: ASR (real)
uvicorn services.asr.server:app --host 0.0.0.0 --port 4001 --reload

# Terminal B: MT
uvicorn services.mt.server:app --host 0.0.0.0 --port 4002 --reload

# Terminal C: Backend
uvicorn backend.app:app --host 0.0.0.0 --port 4000 --reload

# Terminal D: Frontend
cd frontend && npm run dev

# Start a session in the UI (note the returned sessionId)
# Terminal E: Capture agent (Mic -> fallback Loopback)
python services/capture/agent.py --session <SESSION_ID>
```

Notes:

- No audio files are saved; only segments are stored.
- If mic is not available, the agent will switch to loopback (system audio).

## Phase 7: Integrated capture

### New endpoints

- `POST /capture/start` `{ sessionId, source }`
- `POST /capture/stop` `{ sessionId }`

### Dev run order reminder

1. ASR (4001)
2. MT (4002)
3. Backend (4000)
4. Frontend (3000)
5. Click **Start** in the UI

> Note: "Loopback" requires Windows WASAPI; behavior on other OSes may vary.

## Latency tips

Set ASR env for German seminars:

```
set ASR_DEVICE=cuda          # if CUDA works on your 1650; else leave cpu
set ASR_COMPUTE=int8
set ASR_FORCE_LANG=de
```

Smaller chunks = lower latency: 500 ms default.

Silence gate saves CPU & RTT; overlap tail keeps readability.

Order to start is unchanged (ASR→MT→Backend→Frontend).

Expect latency ≤ 1s on typical speech; if consistently higher:

- switch ASR_DEVICE=cpu (sometimes more stable than CUDA on 1650),
- keep ASR_FORCE_LANG=de,
- close background apps using the mic.

## German-first & paragraphs

German is recommended:

```
set ASR_FORCE_LANG=de
```

Live uses 500 ms chunks with WebRTC VAD and ~0.6–0.8 s endpointing so typical speech appears with English translation in under a second. Non‑German speech auto-routes to an NLLB-200 translator. The UI shows partial English immediately and finalizes into paragraphs after pauses.

## Phase 8: Q↔A & Highlights

- Final segments flag questions (German heuristics) and link to top answers via TF-IDF.
- Bookmark any paragraph from Live or Session detail.
- `/sessions/{id}/qa/recompute` rebuilds Q/A links.
- `/sessions/{id}/highlights` lists questions, answers, and bookmarks.
- `/segments/{segmentId}/bookmark` toggles bookmarks.
- Exports now include `captions_src.vtt` and `captions_en.vtt`.

## Phase 9: Diarization & Glossary Enforcement

- ASR adds lightweight diarization, tagging segments as `Speaker 1`, `Speaker 2`, etc.
- Rename speakers per session via `POST /sessions/{id}/speakers/rename { from, to }`.
- English translations enforce glossary terms from `config/glossary.csv` and record hits.
- UI shows speaker tags and underlines enforced terms; Glossary hits appear in Highlights.

## Phase 10: Generators

- Start a local llama-server (Phi-3.5 mini) on `http://127.0.0.1:8080`.
- Endpoints:
  - `POST /generate/summary`
  - `POST /generate/flashcards`
  - `POST /generate/quiz`
  - `POST /generate/explain`
  - `GET /sessions/{id}/generations?type=summary|flashcards|quiz|explain`
- Select paragraphs and choose **Generate…** to open a right-side drawer with results.
- Summaries and Flashcards persist under `/sessions/{id}/summaries` and `/sessions/{id}/flashcards`.

### llama-server

```
llama-server --model phi-3.5
```

The backend looks for the binary either in your `PATH` or under
`llama/llama-server(.exe)` at the project root. On Windows keep the
executable as `llama-server.exe` in that folder if it's not globally
available.

Requests use strict prompts and expect JSON-only responses.

## Phase 11: Hotkeys & Settings

- Global hotkeys:
  - **Alt+S** – Start/Stop capture
  - **Alt+P** – Pause/Resume capture
  - **Alt+B** – Bookmark last paragraph
  - **Alt+L** – Jump to Live
- Settings persisted in `config/runtime_settings.json` via `GET/POST /settings`.
- Storage guardrail: `GET /storage/status` warns when exports + DB exceed 2 GB.
- Bulk delete old sessions with `POST /sessions/purge { before: ISO timestamp }`.
