import json
import os
import time
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from .io_utils import append_csv_row, append_jsonl, load_progress, loads_list, safe_json_dumps
from .metrics import compute_metrics
from .models import build_model
from .parsing import (
    build_gold_relation_edges,
    build_user_prompt,
    clean_model_text,
    format_medications_for_excel,
    get_gold_prescription_labels,
    get_gold_prescription_reasonable,
    parse_audit_result,
)
from .prompts import SYSTEM_PROMPT


def evaluate_single_model(
    model_key: str,
    data: List[Dict[str, Any]],
    config: Dict[str, Any],
    model_registry: Dict[str, Dict[str, Any]],
    output_root: str,
    force_rerun: bool = False,
):
    model = build_model(
        model_key=model_key,
        model_registry=model_registry,
        temperature=float(config.get("temperature", 0)),
        max_tokens=int(config.get("max_tokens", 512)),
    )
    model_dir = os.path.join(output_root, model_key)
    os.makedirs(model_dir, exist_ok=True)

    result_excel = os.path.join(model_dir, "prescription_results.xlsx")
    metric_excel = os.path.join(model_dir, "metrics.xlsx")
    raw_jsonl = os.path.join(model_dir, "raw_outputs.jsonl")
    progress_csv = os.path.join(model_dir, "progress.csv")

    if force_rerun and os.path.exists(progress_csv):
        os.remove(progress_csv)
    progress_df = load_progress(progress_csv)
    processed_ids = set(progress_df["prescription_id"].astype(str).tolist()) if not progress_df.empty else set()
    print(f"[{model_key}] loaded processed_ids: {len(processed_ids)}")

    for item in tqdm(data, desc=f"[{model_key}] eval"):
        prescription_id = str(item.get("prescription_id", ""))
        if prescription_id in processed_ids:
            continue

        patient = item.get("patient_info", {})
        meds = item.get("medications", [])
        gold_labels = get_gold_prescription_labels(item)
        gold_is_reasonable = get_gold_prescription_reasonable(item)
        gold_relation_edges = build_gold_relation_edges(item)

        model_text = ""
        finish_reason = None
        error_msg = ""
        parsed = {"is_reasonable": None, "labels": [], "relation_edges": []}

        for attempt in range(1, int(config.get("max_retry", 2)) + 2):
            try:
                finish_reason, model_text = model.call(SYSTEM_PROMPT, build_user_prompt(item))
                model_text = clean_model_text(model_text)
                append_jsonl(raw_jsonl, {
                    "prescription_id": prescription_id,
                    "finish_reason": finish_reason,
                    "model_text": model_text,
                })
                parsed = parse_audit_result(model_text)
                break
            except Exception as e:
                error_msg = str(e)
                if attempt < int(config.get("max_retry", 2)) + 1:
                    time.sleep(float(config.get("request_interval", 0.5)))

        pred_is_reasonable = parsed["is_reasonable"]
        pred_labels = parsed["labels"]
        pred_relation_edges = parsed["relation_edges"]
        if pred_is_reasonable is None:
            pred_is_reasonable = False
            pred_labels = []
            pred_relation_edges = []

        row = {
            "prescription_id": prescription_id,
            "dept": patient.get("dept", ""),
            "age": patient.get("age", ""),
            "gender": patient.get("gender", ""),
            "diagnosis": patient.get("diagnosis", ""),
            "medications": format_medications_for_excel(meds),
            "gold_standard_labels": safe_json_dumps(gold_labels),
            "audit_result": model_text if model_text else f"call_failed_or_empty: {error_msg}",
            "_gold_is_reasonable": gold_is_reasonable,
            "_pred_is_reasonable": pred_is_reasonable,
            "_gold_labels": safe_json_dumps(gold_labels),
            "_pred_labels": safe_json_dumps(pred_labels),
            "_gold_relation_edges": safe_json_dumps(gold_relation_edges),
            "_pred_relation_edges": safe_json_dumps(pred_relation_edges),
            "_finish_reason": finish_reason,
        }
        append_csv_row(progress_csv, row)
        processed_ids.add(prescription_id)
        time.sleep(float(config.get("request_interval", 0.5)))

    result_df = pd.read_csv(progress_csv, encoding="utf-8-sig")
    result_df["_gold_labels_obj"] = result_df["_gold_labels"].apply(loads_list)
    result_df["_pred_labels_obj"] = result_df["_pred_labels"].apply(loads_list)
    result_df["_gold_relation_edges_obj"] = result_df["_gold_relation_edges"].apply(loads_list)
    result_df["_pred_relation_edges_obj"] = result_df["_pred_relation_edges"].apply(loads_list)

    export_df = result_df[
        ["prescription_id", "dept", "age", "gender", "diagnosis", "medications", "gold_standard_labels", "audit_result"]
    ].copy()
    export_df.to_excel(result_excel, index=False)

    metric_sheets, summary = compute_metrics(result_df)
    with pd.ExcelWriter(metric_excel, engine="openpyxl") as writer:
        metric_sheets["binary"].to_excel(writer, sheet_name="binary", index=False)
        metric_sheets["multilabel"].to_excel(writer, sheet_name="multilabel", index=False)
        metric_sheets["edge"].to_excel(writer, sheet_name="edge", index=False)

    return {
        "model_key": model_key,
        "n_samples": len(result_df),
        **summary,
        "result_excel": result_excel,
        "metric_excel": metric_excel,
        "progress_csv": progress_csv,
        "raw_jsonl": raw_jsonl,
    }


def run_all(config: Dict[str, Any], model_registry: Dict[str, Dict[str, Any]], repeats: int = 1, force_rerun: bool = False):
    base_output_root = config["output_root"]
    os.makedirs(base_output_root, exist_ok=True)
    with open(config["input_json"], "r", encoding="utf-8") as f:
        data = json.load(f)

    all_summaries = []
    for repeat_idx in range(1, repeats + 1):
        repeat_output_root = base_output_root if repeats == 1 else os.path.join(base_output_root, f"repeat_{repeat_idx}")
        os.makedirs(repeat_output_root, exist_ok=True)
        for model_key in config.get("models_to_run", []):
            summary = evaluate_single_model(
                model_key=model_key,
                data=data,
                config=config,
                model_registry=model_registry,
                output_root=repeat_output_root,
                force_rerun=force_rerun,
            )
            summary["repeat"] = repeat_idx
            all_summaries.append(summary)

    leaderboard = pd.DataFrame(all_summaries).sort_values(
        by=["multilabel_micro_f1", "edge_f1"], ascending=False
    )
    leaderboard_path = os.path.join(base_output_root, "leaderboard.csv")
    leaderboard.to_csv(leaderboard_path, index=False, encoding="utf-8-sig")

    if repeats > 1:
        grouped = leaderboard.groupby("model_key", as_index=False).agg({
            "n_samples": "mean",
            "binary_f1": ["mean", "std"],
            "multilabel_micro_f1": ["mean", "std"],
            "multilabel_macro_f1": ["mean", "std"],
            "edge_f1": ["mean", "std"],
        })
        grouped.columns = [
            "model_key",
            "n_samples_mean",
            "binary_f1_mean",
            "binary_f1_std",
            "multilabel_micro_f1_mean",
            "multilabel_micro_f1_std",
            "multilabel_macro_f1_mean",
            "multilabel_macro_f1_std",
            "edge_f1_mean",
            "edge_f1_std",
        ]
        repeat_summary_path = os.path.join(base_output_root, "leaderboard_repeat_summary.csv")
        grouped.sort_values(by=["multilabel_micro_f1_mean", "edge_f1_mean"], ascending=False).to_csv(
            repeat_summary_path, index=False, encoding="utf-8-sig"
        )
    return leaderboard, leaderboard_path
