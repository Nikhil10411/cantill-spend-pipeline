from __future__ import annotations
from typing import List
import pandas as pd


REQUIRED_COLUMNS: List[str] = [
    "record_id",
    "include",
    "exclusion_reason",
    "supplier_key",
    "item_key",
    "spend",
    "l1",
    "l2",
    "l3",
    "l4",
    "confidence",
]

VALID_EXCLUSION_REASONS = {"CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE"}


def validate_and_format_output(
    df: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    expected_row_count: int,
) -> pd.DataFrame:
    out = df[REQUIRED_COLUMNS].copy()

    # 1. Exact row count & unique record_id check
    assert len(out) == expected_row_count, f"Row count mismatch: got {len(out)}, expected {expected_row_count}"
    assert out["record_id"].nunique() == expected_row_count, "Duplicate record_id values detected in output!"

    # 2. Enforce string cleanliness & inclusion/exclusion invariants
    for col in ["record_id", "exclusion_reason", "supplier_key", "item_key", "l1", "l2", "l3", "l4"]:
        out[col] = out[col].fillna("").astype(str).str.strip()

    assert (out["supplier_key"] != "").all(), "Blank supplier_key detected!"
    assert (out["item_key"] != "").all(), "Blank item_key detected!"

    inc_mask = out["include"] == 1
    exc_mask = out["include"] == 0
    assert (inc_mask | exc_mask).all(), "Column 'include' must strictly contain 1 or 0."

    # Excluded row checks
    assert (out.loc[exc_mask, "spend"] == 0).all(), "Excluded rows must have spend == 0."
    assert out.loc[exc_mask, "exclusion_reason"].isin(VALID_EXCLUSION_REASONS).all(), (
        f"Invalid exclusion_reason found: {set(out.loc[exc_mask, 'exclusion_reason']) - VALID_EXCLUSION_REASONS}"
    )
    for lvl in ["l1", "l2", "l3", "l4"]:
        assert (out.loc[exc_mask, lvl] == "").all(), f"Excluded rows must have blank {lvl}."

    # Included row checks
    assert (out.loc[inc_mask, "exclusion_reason"] == "").all(), "Included rows must have blank exclusion_reason."
    assert out["confidence"].between(0.0, 1.0).all(), "Confidence values must be in [0.0, 1.0]."

    # Taxonomy exact string membership check
    num_levels = len(taxonomy_df.columns)
    valid_tuples = {
        tuple(str(v).strip() for v in row)
        for row in taxonomy_df.itertuples(index=False)
    }
    used_level_cols = [f"l{k+1}" for k in range(num_levels)]
    pred_tuples = set(zip(*(out.loc[inc_mask, c] for c in used_level_cols)))
    invalid_preds = pred_tuples - valid_tuples
    assert not invalid_preds, f"Out-of-taxonomy predictions found: {list(invalid_preds)[:5]}"

    # Ensure levels below tree depth remain blank
    for k in range(num_levels, 4):
        assert (out[f"l{k+1}"] == "").all(), f"Column l{k+1} must be blank for {num_levels}-level taxonomy."

    return out