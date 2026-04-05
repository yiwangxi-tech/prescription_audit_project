import json
import os

import pandas as pd


def safe_json_dumps(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_csv_row(path, row_dict):
    pd.DataFrame([row_dict]).to_csv(
        path,
        mode="a",
        index=False,
        header=not os.path.exists(path),
        encoding="utf-8-sig",
    )


def load_progress(progress_csv):
    if not os.path.exists(progress_csv):
        return pd.DataFrame()
    df = pd.read_csv(progress_csv, encoding="utf-8-sig")
    return pd.DataFrame() if df.empty else df


def loads_list(x):
    if pd.isna(x) or x == "":
        return []
    return json.loads(x)
