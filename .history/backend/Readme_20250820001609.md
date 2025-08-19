# Backend (FastAPI + llama.cpp)

Loads a 3–4B GGUF model (e.g., Phi-3.5-mini-instruct Q4_K_M) with partial GPU offload.

## 1) Environment
```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install fastapi "uvicorn[standard]" pydantic python-multipart
pip install --upgrade llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
pip install sentence-transformers faiss-cpu
