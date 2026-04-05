import itertools
import re

from .io_utils import safe_json_dumps
from .labels import ALL_ERROR_LABELS, RELATION_LABELS, canonicalize_label


def clean_model_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    fenced = re.search(r"```(?:text|json)?\s*(.*?)\s*```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text.strip()


def extract_final_answer_block(text: str) -> str:
    if not text:
        return ""

    text = text.strip()
    if text == "合理":
        return text

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    answer_like = []
    for line in lines:
        if line == "合理":
            answer_like.append(line)
            continue
        if "：" in line and "-" in line:
            answer_like.append(line)
            continue
        if "：" in line and "（" in line and "）" in line:
            answer_like.append(line)

    if answer_like:
        return "\n".join(answer_like)

    # Fallback: take the suffix after the last think marker or the last answer-like line.
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if line == "合理" or "：" in line:
            return "\n".join(lines[i:])

    return text


def extract_final_audit_lines(text: str):
    if not text:
        return []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    result_lines = []
    for line in lines:
        if line == "合理":
            result_lines.append(line)
            continue
        if "（" in line and "）" in line and "：" in line:
            result_lines.append(line)

    if result_lines:
        return result_lines

    tail_lines = []
    for line in reversed(lines):
        if line == "合理" or ("（" in line and "）" in line and "：" in line):
            tail_lines.append(line)
        elif tail_lines:
            break
    return list(reversed(tail_lines))


def extract_label_from_line(line: str) -> str:
    start = line.find("（")
    end = line.find("）", start + 1) if start != -1 else -1
    if start != -1 and end != -1:
        raw_label = line[start + 1:end].strip()
        label = canonicalize_label(raw_label)
        if label in ALL_ERROR_LABELS or label in RELATION_LABELS:
            return label

    fallback_candidates = list(ALL_ERROR_LABELS) + list(RELATION_LABELS) + [
        "遴选的药品不适宜",
        "有配伍禁忌或不良相互作用",
        "给药途径",
    ]
    for candidate in fallback_candidates:
        if candidate in line:
            label = canonicalize_label(candidate)
            if label in ALL_ERROR_LABELS or label in RELATION_LABELS:
                return label

    return ""


def format_medications_for_prompt(medications):
    lines = []
    for i, med in enumerate(medications, start=1):
        lines.append(
            f"{i}. 药品名称：{med.get('drug_name','')}\n"
            f"   规格：{med.get('specification','')}\n"
            f"   生产企业：{med.get('manufacturer','')}\n"
            f"   用法用量：{med.get('usage_dosage','')}\n"
            f"   给药途径：{med.get('administration_route','')}"
        )
    return "\n".join(lines)


def format_medications_for_excel(medications):
    simplified = []
    for med in medications:
        simplified.append({
            "drug_name": med.get("drug_name", ""),
            "specification": med.get("specification", ""),
            "manufacturer": med.get("manufacturer", ""),
            "usage_dosage": med.get("usage_dosage", ""),
            "administration_route": med.get("administration_route", "")
        })
    return safe_json_dumps(simplified)


def build_user_prompt(item):
    patient = item.get("patient_info", {})
    meds = item.get("medications", [])
    return (
        f"请审核以下处方：\n\n"
        f"处方ID：{item.get('prescription_id', '')}\n"
        f"科室：{patient.get('dept', '')}\n"
        f"年龄：{patient.get('age', '')}\n"
        f"性别：{patient.get('gender', '')}\n"
        f"诊断：{patient.get('diagnosis', '')}\n\n"
        f"处方药品：\n{format_medications_for_prompt(meds)}"
    )


def get_gold_prescription_reasonable(item):
    meds = item.get("medications", [])
    if not meds:
        return True
    return all(bool(m.get("is_reasonable", True)) for m in meds)


def get_gold_prescription_labels(item):
    labels = set()
    for med in item.get("medications", []):
        for label in med.get("gold_standard_labels", []) or []:
            label = canonicalize_label(label)
            if label and label != "合理":
                labels.add(label)
    return sorted(labels)


def build_gold_relation_edges(item):
    meds = item.get("medications", [])
    edges = set()
    for relation_label in RELATION_LABELS:
        idxs = []
        for i, med in enumerate(meds):
            med_labels = {canonicalize_label(x) for x in (med.get("gold_standard_labels", []) or [])}
            if relation_label in med_labels:
                idxs.append(i)
        for i, j in itertools.combinations(idxs, 2):
            drug_a = str(meds[i].get("drug_name", "")).strip()
            drug_b = str(meds[j].get("drug_name", "")).strip()
            if drug_a and drug_b:
                pair = tuple(sorted([drug_a, drug_b]))
                edges.add((pair[0], pair[1], relation_label))
    return sorted(edges)


def parse_audit_result(text):
    text = clean_model_text(text)
    if not text:
        return {"is_reasonable": None, "labels": [], "relation_edges": []}
    text = extract_final_answer_block(text)
    final_lines = extract_final_audit_lines(text)
    if text == "合理" or final_lines == ["合理"]:
        return {"is_reasonable": True, "labels": [], "relation_edges": []}

    labels = []
    relation_edges = set()

    lines_to_parse = final_lines if final_lines else [line for line in text.splitlines() if line.strip()]
    for raw_line in lines_to_parse:
        line = raw_line.strip()
        if not line:
            continue
        label = extract_label_from_line(line)
        if label in ALL_ERROR_LABELS:
            labels.append(label)
        if label in RELATION_LABELS:
            m = re.match(r"^\s*(.+?)\s*\+\s*(.+?)\s*：", line)
            if m:
                drug_a = m.group(1).strip()
                drug_b = m.group(2).strip()
                if drug_a and drug_b:
                    pair = tuple(sorted([drug_a, drug_b]))
                    relation_edges.add((pair[0], pair[1], label))

    return {
        "is_reasonable": len(set(labels)) == 0,
        "labels": sorted(set(labels)),
        "relation_edges": sorted(relation_edges),
    }
