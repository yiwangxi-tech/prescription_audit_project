import json
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_model_registry(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    registry: Dict[str, Dict[str, Any]] = {}
    model_extra_body = config.get("model_extra_body", {})

    for model_name, item in config.get("local_models", {}).items():
        model_path = item["path"]
        registry[model_name] = {
            "source_type": "local_path",
            "path": model_path,
            "model": item.get("model", model_name),
            "stream": False,
            "extra_body": model_extra_body.get(model_name, item.get("extra_body", {}))
        }

    for model_name, item in config.get("remote_models", {}).items():
        registry[model_name] = {
            "source_type": "remote_server",
            **item
        }

    for model_name, item in config.get("api_models", {}).items():
        registry[model_name] = {
            "source_type": "third_party_api",
            **item
        }

    return registry
