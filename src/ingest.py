from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import yaml


_REF_CACHE: Dict[str, pd.DataFrame] = {}


def _to_str_list(series: pd.Series) -> List[str]:
    return [str(v).strip() for v in series.fillna("").tolist()]


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_csv_safe(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Required CSV not found: {file_path}")
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(file_path, dtype=str, keep_default_na=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode CSV file: {file_path}")


def load_taxonomy(taxonomy_cfg: Dict[str, Any]) -> pd.DataFrame:
    df = read_csv_safe(str(taxonomy_cfg["path"]))
    configured_cols = [str(c) for c in taxonomy_cfg.get("level_cols", [])]
    available_cols = [c for c in configured_cols if c in df.columns]
    if not available_cols:
        available_cols = [
            str(c)
            for c in df.columns
            if str(c).lower().startswith(("category_l", "l1", "l2", "l3", "l4"))
        ]
    tax = pd.DataFrame({c: _to_str_list(df[c]) for c in available_cols})
    tax = tax.drop_duplicates().reset_index(drop=True)
    return tax


def _ingest_from_sources(sources_cfg: List[Dict[str, Any]], input_dir_override: Optional[str] = None) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for src in sources_cfg:
        orig_path = Path(str(src["path"]))
        if input_dir_override:
            candidate = Path(input_dir_override) / orig_path.name
            src_path = str(candidate) if candidate.exists() else str(orig_path)
        else:
            src_path = str(orig_path)

        raw = read_csv_safe(src_path)
        n = len(raw)
        prefix = str(src.get("id_prefix", ""))
        id_col = str(src["id_col"])

        def get_list(col_name: Optional[str]) -> List[str]:
            if col_name and col_name in raw.columns:
                return _to_str_list(raw[col_name])
            return [""] * n

        id_vals = get_list(id_col)
        date_cols = [str(c) for c in src.get("date_cols", [])]

        desc_cols = [str(c) for c in src.get("desc_cols", []) if str(c) in raw.columns]
        if desc_cols:
            desc_lists = [_to_str_list(raw[c]) for c in desc_cols]
            descriptions = [
                " | ".join([val for val in row_vals if val])
                for row_vals in zip(*desc_lists)
            ]
        else:
            descriptions = [""] * n

        ctx_cols = [str(c) for c in src.get("context_cols", []) if str(c) in raw.columns]
        if ctx_cols:
            ctx_lists = [_to_str_list(raw[c]) for c in ctx_cols]
            contexts = [
                " | ".join([f"{col}:{val}" for col, val in zip(ctx_cols, row_vals) if val])
                for row_vals in zip(*ctx_lists)
            ]
        else:
            contexts = [""] * n

        fx_orig = str(src.get("fx_rates_path") or "")
        if fx_orig and input_dir_override:
            fx_cand = Path(input_dir_override) / Path(fx_orig).name
            fx_resolved = str(fx_cand) if fx_cand.exists() else fx_orig
        else:
            fx_resolved = fx_orig

        data_dict: Dict[str, Any] = {
            "_row_order": list(range(n)),
            "_source": [str(src["name"])] * n,
            "_is_primary": [bool(src.get("is_primary", True))] * n,
            "record_id": [f"{prefix}{v}" for v in id_vals],
            "raw_amount": get_list(src.get("amount_col")),
            "is_gross": [bool(src.get("is_gross", False))] * n,
            "raw_vat_rate": get_list(src.get("vat_col")),
            "default_vat_rate": [float(src.get("default_vat_rate", 0.0))] * n,
            "currency": [v.upper() for v in get_list(src.get("currency_col"))],
            "currency_rate": get_list(src.get("currency_rate_col")),
            "fx_rates_path": [fx_resolved] * n,
            "primary_date_raw": get_list(date_cols[0]) if len(date_cols) > 0 else [""] * n,
            "fallback_date_raw": get_list(date_cols[1]) if len(date_cols) > 1 else [""] * n,
            "supplier_id": get_list(src.get("supplier_id_col")),
            "supplier_name": get_list(src.get("supplier_name_col")),
            "po_number_raw": get_list(src.get("po_col")),
            "po_item_raw": get_list(src.get("po_item_col")),
            "po_ref_raw": get_list(src.get("po_ref_col")),
            "invoice_number": get_list(src.get("invoice_col")),
            "invoice_line": get_list(src.get("invoice_line_col")),
            "status": get_list(src.get("status_col")),
            "description": descriptions,
            "context_text": contexts,
        }

        extra_cols = [
            "material_number",
            "material_type",
            "spend_type",
            "opex_capex",
            "gl_account",
            "gl_account_description",
            "supplier_type",
            "document_type",
            "po_type",
            "transport_flag",
            "cost_centre",
        ]
        for extra in extra_cols:
            data_dict[extra] = get_list(extra)

        frames.append(pd.DataFrame(data_dict))

    combined = pd.concat(frames, ignore_index=True)
    combined["_global_idx"] = list(range(len(combined)))
    return combined


def ingest_sources(cfg: Dict[str, Any], input_dir_override: Optional[str] = None) -> pd.DataFrame:
    return _ingest_from_sources(cfg["sources"], input_dir_override=input_dir_override)


def get_reference_df(cfg: Dict[str, Any], current_df: pd.DataFrame) -> pd.DataFrame:
    """Return historical training dataframe if current_df is a blind test split (A2/B2) with new record_ids."""
    lbl_cfg = cfg.get("labels", {})
    lbl_path = str(lbl_cfg.get("path") or "")
    if not lbl_path or not Path(lbl_path).exists():
        return current_df

    lbl_df = read_csv_safe(lbl_path)
    id_col = "record_id" if "record_id" in lbl_df.columns else "line_id"
    lbl_ids = set(_to_str_list(lbl_df[id_col]))
    cur_ids = set(_to_str_list(current_df["record_id"]))

    if len(cur_ids & lbl_ids) > 0:
        return current_df

    cache_key = str(cfg.get("company_id", ""))
    if cache_key not in _REF_CACHE:
        try:
            _REF_CACHE[cache_key] = _ingest_from_sources(cfg["sources"], input_dir_override=None)
        except Exception:
            return current_df
    return _REF_CACHE[cache_key]
