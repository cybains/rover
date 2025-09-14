$root = Split-Path -Parent $MyInvocation.MyCommand.Path
python "$root\preflight.py"

Start-Process powershell -ArgumentList "-NoExit", "cd $root\services\asr; uvicorn services.asr.server:app --host 0.0.0.0 --port 4001 --reload"
Start-Process powershell -ArgumentList "-NoExit", "cd $root\services\mt; uvicorn services.mt.server:app --host 0.0.0.0 --port 4002 --reload"
Start-Process powershell -ArgumentList "-NoExit", "cd $root\backend; uvicorn backend.app:app --host 0.0.0.0 --port 4000 --reload"
Start-Process powershell -ArgumentList "-NoExit", "cd $root\frontend; npm run dev"
Start-Process "http://localhost:3000"
