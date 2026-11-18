from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional
import pandas as pd
from src.ingest import ingest_sources, load_config, load_taxonomy, read_csv_safe
from src.cleaner import clean_amounts_and_dates
from src.exclusions import apply_exclusions
from src.dedup import resolve_entities
from src.classifier import classify_taxonomy
from src.validator import validate_and_format_output


def evaluate_zero_shot_holdout(out_df: pd.DataFrame, labels_path: str, company_id: str) -> None:
    labels = read_csv_safe(labels_path)
    id_col = "record_id" if "record_id" in labels.columns else "line_id"

    if "include" in labels.columns:
        merged_all = out_df.merge(labels, left_on="record_id", right_on=id_col, suffixes=("_pred", "_true"))
        if merged_all.empty:
            print("[EVAL] No matching records found for evaluation.")
            return
        inc_acc = (merged_all["include_pred"].astype(str) == merged_all["include_true"].astype(str)).mean() * 100
        merged = merged_all[merged_all["include_true"].astype(str) == "1"].copy()
        print(f"\n--- Zero-Shot Validation on Company {company_id} (Labels Hidden) ---")
        print(f"Evaluated Seed Rows:  {len(merged_all):,}")
        print(f"Include/Exclude Acc:  {inc_acc:.2f}%")
    else:
        erp_out = out_df[out_df["record_id"].str.startswith("ERP:") & (out_df["include"] == 1)].copy()
        merged = erp_out.merge(labels, left_on="record_id", right_on=id_col, how="inner")
        if merged.empty:
            print("[EVAL] No matching ERP records found for evaluation.")
            return
        print(f"\n--- Zero-Shot Validation on Company {company_id} (Labels Hidden) ---")
        print(f"Evaluated ERP lines:  {len(merged):,}")

    l1_acc = (merged["l1"] == merged["category_l1"]).mean() * 100
    l2_acc = ((merged["l1"] == merged["category_l1"]) & (merged["l2"] == merged["category_l2"])).mean() * 100
    l3_acc = (
        (merged["l1"] == merged["category_l1"])
        & (merged["l2"] == merged["category_l2"])
        & (merged["l3"] == merged["category_l3"])
    ).mean() * 100

    total_spend = merged["spend"].sum()
    exact_mask = (
        (merged["l1"] == merged["category_l1"])
        & (merged["l2"] == merged["category_l2"])
        & (merged["l3"] == merged["category_l3"])
    )
    weighted_l3 = (merged.loc[exact_mask, "spend"].sum() / max(total_spend, 1.0)) * 100

    print(f"L1 Accuracy:          {l1_acc:.2f}%")
    print(f"L2 Accuracy:          {l2_acc:.2f}%")
    print(f"L3 Exact Accuracy:    {l3_acc:.2f}%")
    print(f"Spend-Weighted L3:    {weighted_l3:.2f}%")


def run_pipeline(
    config_path: str,
    eval_zero_shot: bool = False,
    output_override: Optional[str] = None,
    input_dir_override: Optional[str] = None,
) -> None:
    t0 = time.perf_counter()
    cfg = load_config(config_path)
    company_id = cfg["company_id"]
    print(f"[1/6] Ingesting sources for Company {company_id}...")

    raw_df = ingest_sources(cfg, input_dir_override=input_dir_override)
    expected_rows = len(raw_df)
    taxonomy_df = load_taxonomy(cfg["taxonomy"])
    print(f"      Loaded {expected_rows:,} total input rows | Taxonomy leaves: {len(taxonomy_df)}")

    print("[2/6] Cleaning amounts, VAT, FX conversions, and dates...")
    cleaned_df = clean_amounts_and_dates(raw_df, cfg)

    print("[3/6] Applying spend rules, exclusions, and cross-system duplicate matching...")
    filtered_df = apply_exclusions(cleaned_df, cfg)

    print("[4/6] Deduplicating suppliers and items...")
    dedup_df = resolve_entities(filtered_df, cfg)

    print("[5/6] Classifying included lines into hierarchical taxonomy...")
    classified_df = classify_taxonomy(dedup_df, taxonomy_df, cfg, force_zero_shot=eval_zero_shot)

    print("[6/6] Validating output schema and saving CSV...")
    final_df = validate_and_format_output(classified_df, taxonomy_df, expected_rows)

    out_path = Path(output_override or cfg["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)

    elapsed = time.perf_counter() - t0
    inc_cnt = int((final_df["include"] == 1).sum())
    tot_spend = float(final_df.loc[final_df["include"] == 1, "spend"].sum())
    print(f"\n[DONE] Saved {out_path} in {elapsed:.1f}s")
    print(f"       Included: {inc_cnt:,} / {expected_rows:,} | Total Net Spend ({cfg['reporting_currency']}): {tot_spend:,.2f}")
    print(f"       Unique Suppliers: {final_df['supplier_key'].nunique():,} | Unique Items: {final_df['item_key'].nunique():,}")
    print(f"       Exclusions breakdown: {final_df.loc[final_df['include'] == 0, 'exclusion_reason'].value_counts().to_dict()}")

    if eval_zero_shot and cfg.get("labels", {}).get("path"):
        evaluate_zero_shot_holdout(final_df, cfg["labels"]["path"], company_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cantill.ai Spend Data Pipeline & Taxonomy Classifier")
    parser.add_argument("--config", required=True, help="Path to company YAML config file")
    parser.add_argument("--eval-zero-shot", action="store_true", help="Hide labels and evaluate zero-shot accuracy")
    parser.add_argument("--output", default=None, help="Optional override path for output CSV")
    parser.add_argument("--input-dir", default=None, help="Optional directory override for blind test splits (A2, B2)")
    args = parser.parse_args()
    run_pipeline(
        args.config,
        eval_zero_shot=args.eval_zero_shot,
        output_override=args.output,
        input_dir_override=args.input_dir,
    )
