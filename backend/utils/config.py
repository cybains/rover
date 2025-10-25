import os
import yaml
from functools import lru_cache

DEFAULTS = {
    "web_port": 3000,
    "api_port": 4000,
    "autosave_sec": 5,
    "export_dir": "./exports/sessions",
}

@lru_cache()
def get_config():
    path = os.path.join("config", "app.yaml")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    cfg = DEFAULTS.copy()
    cfg.update(data)
    return cfg
