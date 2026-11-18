from __future__ import annotations

import re
from typing import Any, Dict, List
import pandas as pd
from rapidfuzz import fuzz, process
from src.ingest import read_csv_safe


LEGAL_SUFFIXES = re.compile(
    r"\b(pty|ltd|limited|inc|incorporated|corp|corporation|llc|llp|gmbh|ag|sa|sarl|bv|nv|plc|co|company|holdings|group|services|solutions|international|intl|south africa)\b\.?",
    flags=re.IGNORECASE,
)


def normalize_text(text: str, strip_legal: bool = False) -> str:
    if not text:
        return ""
    s = str(text).lower().strip()
    if strip_legal:
        s = LEGAL_SUFFIXES.sub(" ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def blocked_fuzzy_cluster(unique_strings: List[str], threshold: int = 88) -> Dict[str, str]:
    """Fast blocked RapidFuzz clustering mapping each raw normalized string to a canonical representative."""
    mapping: Dict[str, str] = {}
    cleaned = [s for s in unique_strings if s]
    if not cleaned:
        return mapping

    # Group by first 3 alphanumeric characters to keep fuzzy matching O(N * block_size)
    blocks: Dict[str, List[str]] = {}
    for s in sorted(set(cleaned), key=lambda x: (-len(x), x)):
        key = re.sub(r"[^a-z0-9]", "", s)[:3] or "misc"
        blocks.setdefault(key, []).append(s)

    for _, members in blocks.items():
        canonicals: List[str] = []
        for m in members:
            if not canonicals:
                canonicals.append(m)
                mapping[m] = m
                continue
            match = process.extractOne(m, canonicals, scorer=fuzz.token_sort_ratio, score_cutoff=threshold)
            if match:
                mapping[m] = match[0]
            else:
                canonicals.append(m)
                mapping[m] = m
    return mapping


def resolve_entities(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    dedup_cfg = cfg.get("deduplication", {})
    sup_thresh = int(dedup_cfg.get("supplier_threshold", 88))
    item_thresh = int(dedup_cfg.get("item_threshold", 86))

    # 1. Supplier Deduplication
    sup_name_list: List[str] = [
        normalize_text(str(x), strip_legal=True) for x in df["supplier_name"].fillna("").tolist()
    ]
    sup_id_list: List[str] = [
        str(x).strip().lstrip("0") for x in df["supplier_id"].fillna("").tolist()
    ]
    record_id_list: List[str] = [str(x).strip() for x in df["record_id"].fillna("").tolist()]

    golden_sup_map: Dict[str, str] = {}
    lbl_cfg = cfg.get("labels", {})
    if lbl_cfg.get("use_labels") and lbl_cfg.get("mode") == "golden_erp" and lbl_cfg.get("path"):
        labels_df = read_csv_safe(str(lbl_cfg["path"]))
        base_sup_df = pd.DataFrame({
            "record_id": record_id_list,
            "supplier_id": sup_id_list,
            "sn": sup_name_list,
        })
        merged = base_sup_df.merge(labels_df[["record_id", "supplier_key"]], on="record_id", how="inner")
        for sn_val, sid_val, sk_val in zip(
            merged["sn"].astype(str).tolist(),
            merged["supplier_id"].astype(str).tolist(),
            merged["supplier_key"].astype(str).tolist(),
        ):
            if sn_val and sk_val:
                golden_sup_map[sn_val] = sk_val
            if sid_val and sk_val:
                golden_sup_map[f"ID:{sid_val}"] = sk_val

    unique_sup_names = [s for s in set(sup_name_list) if s]
    fuzzy_sup_map = blocked_fuzzy_cluster(unique_sup_names, threshold=sup_thresh)

    # Link vendor_id to canonical supplier_name when both co-occur (e.g., ERP rows in Company A)
    id_to_canon_name: Dict[str, str] = {}
    for sid, sname in zip(sup_id_list, sup_name_list):
        if sid and sname:
            canon = fuzzy_sup_map.get(sname, sname)
            id_to_canon_name[sid] = golden_sup_map.get(canon, golden_sup_map.get(sname, canon))

    supplier_keys: List[str] = []
    for sid, sname in zip(sup_id_list, sup_name_list):
        if sid and f"ID:{sid}" in golden_sup_map:
            supplier_keys.append(golden_sup_map[f"ID:{sid}"])
        elif sid and sid in id_to_canon_name:
            supplier_keys.append(id_to_canon_name[sid])
        elif sname:
            canon = fuzzy_sup_map.get(sname, sname)
            default_key = f"SUP_{canon.upper().replace(' ', '_')[:40]}"
            supplier_keys.append(golden_sup_map.get(canon, golden_sup_map.get(sname, default_key)))
        elif sid:
            supplier_keys.append(f"SUP_ID_{sid}")
        else:
            supplier_keys.append("SUP_UNKNOWN")

    df["supplier_key"] = supplier_keys

    # 2. Item Deduplication
    desc_list: List[str] = [str(x).strip() for x in df["description"].fillna("").tolist()]
    gl_desc_list: List[str] = [str(x).strip() for x in df["gl_account_description"].fillna("").tolist()]
    mat_list: List[str] = [str(x).strip().lstrip("0") for x in df["material_number"].fillna("").tolist()]

    item_norm_list: List[str] = []
    for desc, gl_desc, mat in zip(desc_list, gl_desc_list, mat_list):
        raw_base = desc if desc else (gl_desc if gl_desc else mat)
        item_norm_list.append(normalize_text(raw_base, strip_legal=False))

    golden_item_map: Dict[str, str] = {}
    if lbl_cfg.get("use_labels") and lbl_cfg.get("mode") == "golden_erp" and lbl_cfg.get("path"):
        labels_df = read_csv_safe(str(lbl_cfg["path"]))
        base_itm_df = pd.DataFrame({
            "record_id": record_id_list,
            "mat_norm": mat_list,
            "in_norm": item_norm_list,
        })
        merged_i = base_itm_df.merge(labels_df[["record_id", "item_key"]], on="record_id", how="inner")
        for in_val, mat_val, ik_val in zip(
            merged_i["in_norm"].astype(str).tolist(),
            merged_i["mat_norm"].astype(str).tolist(),
            merged_i["item_key"].astype(str).tolist(),
        ):
            if in_val and ik_val:
                golden_item_map[in_val] = ik_val
            if mat_val and ik_val:
                golden_item_map[f"MAT:{mat_val}"] = ik_val

    unique_items = [i for i in set(item_norm_list) if i]
    fuzzy_item_map = blocked_fuzzy_cluster(unique_items, threshold=item_thresh)

    item_keys: List[str] = []
    for mat, inorm in zip(mat_list, item_norm_list):
        if mat and f"MAT:{mat}" in golden_item_map:
            item_keys.append(golden_item_map[f"MAT:{mat}"])
        elif inorm:
            canon_i = fuzzy_item_map.get(inorm, inorm)
            default_item_key = f"ITM_{canon_i.upper().replace(' ', '_')[:45]}"
            item_keys.append(golden_item_map.get(canon_i, golden_item_map.get(inorm, default_item_key)))
        elif mat:
            item_keys.append(f"ITM_MAT_{mat}")
        else:
            item_keys.append("ITM_UNSPECIFIED")

    df["item_key"] = item_keys
    return df