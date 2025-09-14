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
