$cfg = Get-Content "config/app.yaml" | ConvertFrom-Yaml
$port = $cfg.api_port
uvicorn backend.app:app --port $port --reload
