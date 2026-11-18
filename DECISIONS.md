# Engineering & Data Ambiguity Decisions

## 1. Company A: Cross-System Precedence & Tax Handling
- **ERP vs. Invoice Tool Precedence**: Per procurement accounting rules, when a purchase appears in both `erp_goods_receipts.csv` and `invoice_tool_export.csv`, the `ERP:` record is retained (`include=1`) because it contains validated goods-receipt net values, cost centres, and material codes, while the corresponding `TOOL:` line is marked `include=0, exclusion_reason=DUPLICATE`.
- **VAT Normalization**: `erp_goods_receipts.csv` (`value_zar`) is already net of tax. `invoice_tool_export.csv` (`gross_amount`) includes VAT; where `vat_rate` is blank/NaN, 15% (`0.15`) is applied (`gross / 1.15`). Values recorded as whole percentages (e.g., `15`) are normalized to `0.15` prior to division.
- **PO Reference Parsing**: `po_reference` in `invoice_tool_export.csv` contains prefixes (`PO `), slashes (`4510085877/10`), and leading zeros. Regex extraction isolates the numeric PO core and optional line item number, stripping leading zeros to align with SAP `po_number` and `po_item`.

## 2. Company B: Multi-ERP Currencies, Dates & Reversals
- **Missing or Out-of-Bounds Dates**: Where `invoice_date` is missing or contains implausible historical outliers (e.g., year `< 2020`), the pipeline falls back to `posting_date` for FX month lookup and temporal checks.
- **FX Conversion Fallback**: Non-USD lines first use the row-level `currency_rate` (`line_amount / currency_rate`). Where missing or zero, `fx_rates_usd.csv` is joined on `(currency, YYYY-MM)`. If a specific `(currency, month)` pair is absent from the FX table (e.g., `AED` in `2024-02`), the currency's median `units_per_usd` across available months is used as a deterministic fallback.
- **Offsetting Reversal Pairs**: Equal-and-opposite line amounts on the same `(vendor_id, invoice_number)` (e.g., `-13827.64` and `+13827.64`) represent cancelled/reversed postings; both lines are excluded with `exclusion_reason=CANCELLED` and `spend=0`.
- **Supplier Key Without Supplier Names**: Because `invoice_lines.csv` provides `vendor_id` without vendor names, `supplier_key` is derived from normalized `vendor_id` (stripping leading zeros).
- **Blank Line Descriptions**: Rows with empty `line_description` are enriched using `gl_account_description`, `material_number`, `po_type`, `supplier_type`, and `transport_flag` so both deduplication (`item_key`) and multilingual taxonomy classification remain accurate.

## 3. Generalization to Company C
- **Zero-Label Readiness**: No category strings or company-specific column names exist in `src/`. Company C requires only creating `submission/config_C.yaml` mapping its column names and up to 4 taxonomy levels (`category_l1`..`category_l4`).