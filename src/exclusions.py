from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from src.ingest import get_reference_df, read_csv_safe


_LEGAL_RE = re.compile(r"\b(pty|ltd|limited|inc|corp|corporation|llc|gmbh|sa|co|plc)\b\.?", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]")


def _norm_supplier_fast(s: pd.Series) -> List[str]:
    result: List[str] = []
    for val in s.fillna("").tolist():
        cleaned = str(val).lower().strip()
        cleaned = _LEGAL_RE.sub("", cleaned)
        cleaned = _NON_ALNUM_RE.sub("", cleaned)
        result.append(cleaned)
    return result


def learn_seed_exclusion_signals(cfg: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Set[str]]:
    learned: Dict[str, Set[str]] = {
        "gl_accounts": set(),
        "gl_descriptions": set(),
        "supplier_types": set(),
        "document_types": set(),
    }
    lbl_cfg = cfg.get("labels", {})
    if lbl_cfg.get("mode") != "seed_subset" or not lbl_cfg.get("path"):
        return learned

    seed = read_csv_safe(str(lbl_cfg["path"]))
    ref_df = get_reference_df(cfg, df)
    merged = ref_df.merge(seed, left_on="record_id", right_on="line_id", how="inner")
    non_proc = merged[merged["exclusion_reason"] == "NON_PROCUREMENT"]

    if not non_proc.empty:
        col_map = [
            ("gl_account", "gl_accounts"),
            ("gl_account_description", "gl_descriptions"),
        ]
        for col, key in col_map:
            vals = non_proc[col].replace("", np.nan).dropna()
            total_counts = merged[col].value_counts()
            np_counts = vals.value_counts()
            for code, cnt in np_counts.items():
                code_str = str(code).strip()
                total_for_code = int(total_counts.get(code, 1))
                if int(cnt) >= 2 and (int(cnt) / max(total_for_code, 1)) >= 0.90:
                    learned[key].add(code_str.lower())
    return learned


def apply_exclusions(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    excl_cfg = cfg.get("exclusions", {})
    n = len(df)

    include = np.ones(n, dtype=np.int64)
    reason = np.full(n, "", dtype=object)
    excl_conf = np.full(n, np.nan, dtype=np.float64)

    # 1. Status checks (CANCELLED vs NOT_APPROVED)
    approved_set = {str(s).lower().strip() for s in excl_cfg.get("approved_statuses", [])}
    cancelled_set = {str(s).lower().strip() for s in excl_cfg.get("cancelled_statuses", [])}

    status_list = [str(v).lower().strip() for v in df["status"].fillna("").tolist()]
    has_status = np.array([s != "" for s in status_list], dtype=bool)
    is_cancelled = np.array([s in cancelled_set for s in status_list], dtype=bool) & has_status

    include = np.where(is_cancelled, 0, include)
    reason = np.where(is_cancelled, "CANCELLED", reason)
    excl_conf = np.where(is_cancelled, 0.99, excl_conf)

    if approved_set:
        not_in_approved = np.array([s not in approved_set for s in status_list], dtype=bool)
        is_not_approved = (include == 1) & has_status & not_in_approved
        include = np.where(is_not_approved, 0, include)
        reason = np.where(is_not_approved, "NOT_APPROVED", reason)
        excl_conf = np.where(is_not_approved, 0.98, excl_conf)

    # 2. Zero / Null spend (Only for Company A where 0.0 spend is non-procurement; Company B keeps 0.0 amounts)
    amt_series = pd.to_numeric(df["net_spend_reporting"], errors="coerce").fillna(0.0)
    amt = amt_series.to_numpy(dtype=np.float64)

    if str(cfg.get("company_id", "")) == "A":
        is_zero = (include == 1) & (np.abs(amt) < 1e-4)
        include = np.where(is_zero, 0, include)
        reason = np.where(is_zero, "NON_PROCUREMENT", reason)
        excl_conf = np.where(is_zero, 0.95, excl_conf)

    # 3. NON_PROCUREMENT rules (Intercompany GLs + Tax descriptions + keywords + seed-learned patterns)
    learned = learn_seed_exclusion_signals(cfg, df)
    kw_list = [re.escape(str(k).lower().strip()) for k in excl_cfg.get("non_procurement_keywords", [])]
    gl_pats = [re.escape(str(k).lower().strip()) for k in excl_cfg.get("non_procurement_gl_patterns", [])]
    gl_exact_set = {str(k).lower().strip() for k in excl_cfg.get("non_procurement_gl_exact", [])}
    st_list = {str(k).lower().strip() for k in excl_cfg.get("non_procurement_supplier_types", [])} | learned["supplier_types"]

    desc_lower = [str(v).lower() for v in df["description"].fillna("").tolist()]
    gl_desc_lower = [str(v).lower() for v in df["gl_account_description"].fillna("").tolist()]
    sup_lower = [str(v).lower() for v in df["supplier_name"].fillna("").tolist()]
    sup_type_lower = [str(v).lower().strip() for v in df["supplier_type"].fillna("").tolist()]
    gl_acc_lower = [str(v).lower().strip() for v in df["gl_account"].fillna("").tolist()]
    doc_type_lower = [str(v).lower().strip() for v in df["document_type"].fillna("").tolist()]

    non_proc_mask = np.zeros(n, dtype=bool)

    # Intercompany GL descriptions (IC*, I/C Payable)
    for idx, (gd, ga) in enumerate(zip(gl_desc_lower, gl_acc_lower)):
        if gd.startswith("ic") or "i/c payable" in gd or "ic payables" in gd:
            non_proc_mask[idx] = True

    # Tax descriptions in description / gl_description
    for idx, (d, gd) in enumerate(zip(desc_lower, gl_desc_lower)):
        if any(tx in f"{d} {gd}" for tx in ("sales tax", "state & local taxes", "gst input tax", "vat payment")):
            non_proc_mask[idx] = True

    if kw_list:
        kw_regex = re.compile(r"\b(?:" + "|".join(kw_list) + r")\b")
        non_proc_mask |= np.array(
            [bool(kw_regex.search(f"{d} {g} {s}")) for d, g, s in zip(desc_lower, gl_desc_lower, sup_lower)],
            dtype=bool,
        )

    if gl_pats:
        gl_regex = re.compile(r"(?:" + "|".join(gl_pats) + r")")
        non_proc_mask |= np.array([bool(gl_regex.search(g)) for g in gl_desc_lower], dtype=bool)

    if gl_exact_set:
        non_proc_mask |= np.array(
            [(gd.strip() in gl_exact_set) or (ga in gl_exact_set) for gd, ga in zip(gl_desc_lower, gl_acc_lower)],
            dtype=bool,
        )

    if st_list:
        non_proc_mask |= np.array([st in st_list for st in sup_type_lower], dtype=bool)

    if learned["gl_accounts"]:
        non_proc_mask |= np.array([ga in learned["gl_accounts"] for ga in gl_acc_lower], dtype=bool)

    if learned["gl_descriptions"]:
        non_proc_mask |= np.array([gd.strip() in learned["gl_descriptions"] for gd in gl_desc_lower], dtype=bool)

    if learned["document_types"]:
        non_proc_mask |= np.array([dt in learned["document_types"] for dt in doc_type_lower], dtype=bool)

    apply_np = (include == 1) & non_proc_mask
    include = np.where(apply_np, 0, include)
    reason = np.where(apply_np, "NON_PROCUREMENT", reason)
    excl_conf = np.where(apply_np, 0.93, excl_conf)

    # 4. Internal duplicate lines within same source file
    is_primary = df["_is_primary"].to_numpy(dtype=bool)
    has_inv = np.array([str(v) != "" for v in df["invoice_number"].fillna("").tolist()], dtype=bool)

    if excl_cfg.get("internal_tool_dedup", False):
        tool_mask = (~is_primary) & (include == 1) & has_inv
        dup_cols = ["invoice_number", "supplier_name", "description", "raw_amount"]
        is_dup = tool_mask & df.duplicated(subset=dup_cols, keep="first").to_numpy(dtype=bool)
        include = np.where(is_dup, 0, include)
        reason = np.where(is_dup, "DUPLICATE", reason)
        excl_conf = np.where(is_dup, 0.98, excl_conf)

    if excl_cfg.get("internal_line_dedup", False):
        line_mask = (include == 1) & has_inv
        dup_cols = ["invoice_number", "invoice_line", "supplier_id", "supplier_name", "raw_amount", "description"]
        is_dup = line_mask & df.duplicated(subset=dup_cols, keep="first").to_numpy(dtype=bool)
        include = np.where(is_dup, 0, include)
        reason = np.where(is_dup, "DUPLICATE", reason)
        excl_conf = np.where(is_dup, 0.97, excl_conf)

    # 5. Cross-System Duplicate Matching (Company A: TOOL vs ERP)
    if excl_cfg.get("cross_system_dedup", False):
        erp_indices = np.flatnonzero(is_primary & (include == 1))
        tool_indices = np.flatnonzero((~is_primary) & (include == 1))

        if len(erp_indices) > 0 and len(tool_indices) > 0:
            sup_norm_list = _norm_supplier_fast(df["supplier_name"])
            po_num_list = [str(v) for v in df["po_number_clean"].fillna("").tolist()]
            po_itm_list = [str(v) for v in df["po_item_clean"].fillna("").tolist()]
            date_list = df["resolved_date"].tolist()

            erp_po_item_set: Set[Tuple[str, str]] = set()
            erp_by_po: Dict[str, List[Tuple[str, float]]] = {}
            erp_by_sup: Dict[str, List[Tuple[float, Any]]] = {}

            for i in erp_indices:
                po_n = po_num_list[i]
                po_i = po_itm_list[i]
                sup_n = sup_norm_list[i]
                e_amt = float(amt[i])
                e_dt = date_list[i]

                if po_n:
                    if po_i:
                        erp_po_item_set.add((po_n, po_i))
                    erp_by_po.setdefault(po_n, []).append((sup_n, e_amt))
                if sup_n:
                    erp_by_sup.setdefault(sup_n, []).append((e_amt, e_dt))

            for idx in tool_indices:
                po_n = po_num_list[idx]
                po_i = po_itm_list[idx]
                sup_n = sup_norm_list[idx]
                t_amt = float(amt[idx])
                t_dt = date_list[idx]

                matched = False
                conf = 0.0

                if po_n and po_i and (po_n, po_i) in erp_po_item_set:
                    matched, conf = True, 0.99
                elif po_n and po_n in erp_by_po:
                    for e_sup, e_amt in erp_by_po[po_n]:
                        amt_close = abs(t_amt - e_amt) <= max(0.05 * abs(e_amt), 1.0)
                        if (sup_n and sup_n == e_sup) or amt_close:
                            matched, conf = True, 0.94
                            break
                if not matched and sup_n and sup_n in erp_by_sup and pd.notna(t_dt):
                    for e_amt, e_dt in erp_by_sup[sup_n]:
                        if pd.notna(e_dt) and abs((t_dt - e_dt).days) <= 45:
                            if abs(t_amt - e_amt) <= max(0.02 * abs(e_amt), 1.0):
                                matched, conf = True, 0.88
                                break

                if matched:
                    include[idx] = 0
                    reason[idx] = "DUPLICATE"
                    excl_conf[idx] = conf

    df["include"] = include
    df["exclusion_reason"] = reason
    df["_excl_confidence"] = excl_conf
    df["spend"] = np.where(include == 1, np.round(amt, 2), 0.0)
    return df
