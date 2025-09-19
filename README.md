## MVP-0 (Live UI + Sessions + Mock Stream)

### Prereqs
- Python 3.10+
- Node 18+
- MongoDB running
- set `MONGO_URI`

### Steps
1. `powershell ./scripts/dev_backend.ps1`
2. `powershell ./scripts/dev_frontend.ps1`
3. Open http://localhost:3000/app/live
4. Click **Start Session**.
5. In another shell: `curl -X POST http://localhost:4000/mock/stream -H "Content-Type: application/json" -d '{"sessionId":"<id>"}'`
6. Click **Stop** and check `exports/sessions/...` for files.
