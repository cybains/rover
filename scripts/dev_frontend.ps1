$cfg = Get-Content "config/app.yaml" | ConvertFrom-Yaml
$port = $cfg.web_port
cd frontend
npm run dev -- --port $port
