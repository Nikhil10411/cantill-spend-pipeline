from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from src.ingest import read_csv_safe


_CURRENCY_SYM_RE = re.compile(r"[\s,$€£₹]")
_PAREN_NEG_RE = re.compile(r"^\((.+)\)$")
_PO_EXTRACT_RE = re.compile(r"(?:PO[\s\-_:#/]*)?(\d{7,12})(?:[\s\-_/:]+(\d{1,5}))?", re.IGNORECASE)
_NON_DIGIT_RE = re.compile(r"\D")


def _to_str_list(series: pd.Series) -> List[str]:
    return [str(v).strip() for v in series.fillna("").tolist()]


def parse_numeric(series: pd.Series) -> pd.Series:
    cleaned: List[str] = []
    for raw in _to_str_list(series):
        s = _CURRENCY_SYM_RE.sub("", raw)
        s = _PAREN_NEG_RE.sub(r"-\1", s)
        cleaned.append(s)
    return pd.to_numeric(pd.Series(cleaned, index=series.index), errors="coerce")


def parse_dates_with_fallback(primary: pd.Series, fallback: pd.Series) -> Tuple[pd.Series, pd.Series]:
    p1_raw = [v if v != "" else None for v in _to_str_list(primary)]
    p2_raw = [v if v != "" else None for v in _to_str_list(fallback)]

    p1 = pd.to_datetime(pd.Series(p1_raw, index=primary.index), errors="coerce", format="mixed")
    invalid_p1 = p1.isna() | (p1.dt.year < 2020) | (p1.dt.year > 2027)

    p2 = pd.to_datetime(pd.Series(p2_raw, index=fallback.index), errors="coerce", format="mixed")
    invalid_p2 = p2.isna() | (p2.dt.year < 2020) | (p2.dt.year > 2027)

    resolved = p1.where(~invalid_p1, p2.where(~invalid_p2, pd.NaT))
    date_valid_flag = resolved.notna()
    return resolved, date_valid_flag


def parse_po_fields(df: pd.DataFrame) -> pd.DataFrame:
    po_nums = _to_str_list(df["po_number_raw"])
    po_itms = _to_str_list(df["po_item_raw"])
    po_refs = _to_str_list(df["po_ref_raw"])

    clean_po_list: List[str] = []
    clean_itm_list: List[str] = []

    for p_num, p_itm, p_ref in zip(po_nums, po_itms, po_refs):
        if (not p_num or not p_itm) and p_ref:
            m = _PO_EXTRACT_RE.search(p_ref)
            if m:
                if not p_num and m.group(1):
                    p_num = m.group(1)
                if not p_itm and m.group(2):
                    p_itm = m.group(2)

        c_num = _NON_DIGIT_RE.sub("", p_num).lstrip("0")
        c_itm = _NON_DIGIT_RE.sub("", p_itm).lstrip("0")
        clean_po_list.append(c_num)
        clean_itm_list.append(c_itm)

    df["po_number_clean"] = clean_po_list
    df["po_item_clean"] = clean_itm_list
    return df


def apply_fx_conversion(df: pd.DataFrame, reporting_currency: str) -> pd.Series:
    target_curr = reporting_currency.strip().upper()
    amt_arr = pd.to_numeric(df["net_local_amount"], errors="coerce").to_numpy(dtype=np.float64)
    rate_arr = parse_numeric(df["currency_rate"]).to_numpy(dtype=np.float64)
    curr_list = [v.upper() if v else target_curr for v in _to_str_list(df["currency"])]

    converted = amt_arr.copy()
    need_fx_indices: List[int] = []

    for i, c in enumerate(curr_list):
        if c == target_curr:
            continue
        r = rate_arr[i]
        if np.isfinite(r) and r > 0.0:
            converted[i] = amt_arr[i] / r
        else:
            need_fx_indices.append(i)

    if not need_fx_indices:
        return pd.Series(converted, index=df.index)

    fx_paths = [p for p in _to_str_list(df["fx_rates_path"]) if p and Path(p).exists()]
    if not fx_paths:
        return pd.Series(converted, index=df.index)

    fx = read_csv_safe(fx_paths[0])
    fx["currency"] = [v.upper() for v in _to_str_list(fx["currency"])]
    fx["month"] = _to_str_list(fx["month"])
    fx["units_per_usd"] = parse_numeric(fx["units_per_usd"])
    fx = fx.dropna(subset=["units_per_usd"])
    fx = fx[fx["units_per_usd"] > 0]

    exact_map: Dict[Tuple[str, str], float] = {}
    for c_val, m_val, u_val in zip(
        fx["currency"].tolist(),
        fx["month"].tolist(),
        fx["units_per_usd"].tolist(),
    ):
        exact_map[(str(c_val), str(m_val))] = float(u_val)

    median_series = fx.groupby("currency")["units_per_usd"].median()
    median_map: Dict[str, float] = {str(k): float(v) for k, v in median_series.items()}

    ym_list = _to_str_list(df["resolved_date"].dt.strftime("%Y-%m"))

    for idx in need_fx_indices:
        c = curr_list[idx]
        m = ym_list[idx]
        fx_rate = exact_map.get((c, m)) or median_map.get(c)
        if fx_rate and fx_rate > 0.0 and np.isfinite(amt_arr[idx]):
            converted[idx] = amt_arr[idx] / fx_rate

    return pd.Series(converted, index=df.index)


def clean_amounts_and_dates(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    reporting_currency = str(cfg["reporting_currency"]).strip().upper()

    # 1. Dates
    df["resolved_date"], df["date_valid"] = parse_dates_with_fallback(
        df["primary_date_raw"], df["fallback_date_raw"]
    )

    # 2. Raw amount & VAT removal
    raw_num = parse_numeric(df["raw_amount"]).fillna(0.0).to_numpy(dtype=np.float64)
    vat_num = parse_numeric(df["raw_vat_rate"]).to_numpy(dtype=np.float64)
    default_vat = pd.to_numeric(df["default_vat_rate"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    is_gross = df["is_gross"].to_numpy(dtype=bool)

    # Normalize whole-number percentage (e.g., 15 -> 0.15) and fill blanks/negatives with default VAT
    vat_norm = np.where(np.isnan(vat_num) | (vat_num <= 1.0), vat_num, vat_num / 100.0)
    vat_filled = np.where(np.isnan(vat_norm) | (vat_norm < 0.0), default_vat, vat_norm)

    net_local = np.where(is_gross, raw_num / (1.0 + vat_filled), raw_num)
    df["net_local_amount"] = net_local

    # 3. Currency conversion to reporting currency
    df["net_spend_reporting"] = apply_fx_conversion(df, reporting_currency)

    # 4. PO field normalization
    df = parse_po_fields(df)

    return df