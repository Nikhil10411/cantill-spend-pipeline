from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import yaml


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
            # Read all columns as str to protect leading zeros and large numeric IDs
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


def ingest_sources(cfg: Dict[str, Any]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for src in cfg["sources"]:
        raw = read_csv_safe(str(src["path"]))
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
            "fx_rates_path": [str(src.get("fx_rates_path") or "")] * n,
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