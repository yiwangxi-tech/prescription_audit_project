import argparse
import json
import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from prescription_audit.config import build_model_registry, load_config
from prescription_audit.models import build_model


def check_tcp(base_url: str, timeout: float = 3.0):
    parsed = urlparse(base_url)
    host = parsed.hostname
    port = parsed.port
    if host is None or port is None:
        return False, "invalid_host_or_port"
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, "tcp_ok"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config json")
    parser.add_argument("--model", default="", help="Only check one model key")
    args = parser.parse_args()

    config = load_config(args.config)
    model_registry = build_model_registry(config)
    model_keys = [args.model] if args.model else config.get("models_to_run", [])

    rows = []
    for model_key in model_keys:
        cfg = model_registry[model_key]
        row = {
            "model_key": model_key,
            "source_type": cfg.get("source_type", ""),
            "base_url": cfg.get("base_url", ""),
            "path": cfg.get("path", ""),
            "configured_model": cfg.get("model", ""),
            "tcp_ok": False,
            "tcp_msg": "",
            "path_exists": False,
            "models_list_ok": False,
            "models_list_msg": "",
            "chat_ok": False,
            "chat_msg": "",
        }

        if cfg.get("source_type") == "local_path":
            try:
                row["path_exists"] = Path(cfg["path"]).exists()
                client = build_model(
                    model_key=model_key,
                    model_registry=model_registry,
                    temperature=0,
                    max_tokens=16,
                )
                model_list = client.list_models()
                ids = []
                for item in getattr(model_list, "data", []):
                    item_id = getattr(item, "id", None)
                    if item_id:
                        ids.append(item_id)
                row["models_list_ok"] = True
                row["models_list_msg"] = ",".join(ids[:20])
            except Exception as e:
                row["models_list_msg"] = str(e)

            try:
                finish_reason, text = client.call(
                    "You are a helpful assistant.",
                    "Reply with exactly: OK",
                )
                row["chat_ok"] = True
                row["chat_msg"] = f"finish_reason={finish_reason}; text={text[:200]}"
            except Exception as e:
                row["chat_msg"] = str(e)
        else:
            base_url = cfg["base_url"]
            tcp_ok, tcp_msg = check_tcp(base_url)
            row["tcp_ok"] = tcp_ok
            row["tcp_msg"] = tcp_msg

            if tcp_ok:
                try:
                    client = build_model(
                        model_key=model_key,
                        model_registry=model_registry,
                        temperature=0,
                        max_tokens=32,
                    )
                    model_list = client.list_models()
                    ids = []
                    for item in getattr(model_list, "data", []):
                        item_id = getattr(item, "id", None)
                        if item_id:
                            ids.append(item_id)
                    row["models_list_ok"] = True
                    row["models_list_msg"] = ",".join(ids[:20])
                except Exception as e:
                    row["models_list_msg"] = str(e)

                try:
                    finish_reason, text = client.call(
                        "You are a helpful assistant.",
                        "Reply with exactly: OK",
                    )
                    row["chat_ok"] = True
                    row["chat_msg"] = f"finish_reason={finish_reason}; text={text[:200]}"
                except Exception as e:
                    row["chat_msg"] = str(e)

        rows.append(row)

    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
