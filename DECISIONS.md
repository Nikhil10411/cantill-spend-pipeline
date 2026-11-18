---

### 2. `DECISIONS.md`

```markdown
# Engineering & Data Ambiguity Decisions

This document records the key accounting, data-quality, and modeling decisions made while resolving real-world ERP/AP ambiguities across Company A, Company B, and blind unseen datasets (Company C).

---

## 1. Company A: Multi-System Reconciliation & Tax Normalization

### Decision 1.1: ERP Goods Receipts Take Precedence Over AP Invoice Tool Lines
* **Observation**: Company A provides two overlapping feeds—`erp_goods_receipts.csv` (37,029 rows) and `invoice_tool_export.csv` (28,520 rows)—where ~19,800 purchases appear in both systems.
* **Decision**: Retain the `ERP:` record (`include = 1`) as the system of record and mark the matched `TOOL:` record as `include = 0, exclusion_reason = DUPLICATE, spend = 0`.
* **Rationale**: SAP goods receipts carry audited net values (`value_zar`), validated cost centres, and material codes, whereas the AP invoice tool captures downstream gross invoice entries (including unmatched non-PO/services invoices that remain `include = 1` when no ERP counterpart exists).

### Decision 1.2: Gross-to-Net VAT Stripping with 15% South African Standard Default
* **Observation**: `erp_goods_receipts.csv` (`value_zar`) is recorded net of tax, whereas `invoice_tool_export.csv` (`gross_amount`) includes VAT, with `vat_rate` formatted variably (`0.15`, `15`, or blank).
* **Decision**: Configure `is_gross: false` for `ERP` and `is_gross: true` with `default_vat_rate: 0.15` for `TOOL`. Values $> 1.0$ (e.g., `15`) are divided by `100` prior to computing `net_spend = gross / (1 + vat_rate)`.

### Decision 1.3: Regex Extraction of Reformatted PO & Line-Item Keys
* **Observation**: `invoice_tool_export.csv` encodes purchase orders as free-text strings (`PO 4510109330`, `4510085877/10`, or zero-padded strings) that fail naive string joins against SAP `po_number` and `po_item`.
* **Decision**: Apply regex normalization during ingestion to split compound `PO/Item` references and strip prefixes and leading zeros, enabling exact Pass-1 `(po_number, po_item)` matching (`0.99` confidence) backed by tolerance-bounded Pass-2/Pass-3 fallbacks.

---

## 2. Company B: Multi-Currency Ledgers, Date Outliers & Precision Exclusions

### Decision 2.1: Posting-Date Fallback for Missing or Implausible Invoice Dates
* **Observation**: Several rows in `invoice_lines.csv` contain missing `invoice_date` values or corrupted historical years (`< 2020`) that break monthly FX rate lookups.
* **Decision**: Configure `date_cols: ["invoice_date", "posting_date"]` so `src/cleaner.py` automatically rejects out-of-bounds timestamps and resolves the valid accounting `posting_date` for FX month alignment.

### Decision 2.2: Three-Tier FX Resolution (`currency_rate` $\rightarrow$ Monthly FX Table $\rightarrow$ Currency Median)
* **Observation**: Non-USD invoice lines occasionally lack an inline `currency_rate` (or have `0`), and specific `(currency, YYYY-MM)` pairs (e.g., `AED` in `2024-02`) are absent from `fx_rates_usd.csv`.
* **Decision**: Resolve `net_spend_reporting` in strict priority order: (1) valid row-level `line_amount / currency_rate`, (2) exact `(currency, YYYY-MM)` lookup in `fx_rates_usd.csv`, and (3) currency-level median `units_per_usd` across available months so zero spend is dropped due to sparse FX coverage.

### Decision 2.3: Non-Destructive Negative Reversal Filtering (`exclude_reversal_offsets: false`)
* **Observation**: Negative line amounts represent credit memos/reversals, while positive lines on the same `(vendor_id, invoice_number)` frequently represent valid net procurement charges rather than cancelled pairs.
* **Decision**: Exclude negative reversal lines (`amt < -1e-4`) as `exclusion_reason = CANCELLED` while keeping positive invoice lines eligible unless explicitly flagged by status or non-procurement rules (`exclude_reversal_offsets: false`). This prevented false-positive exclusions on valid positive invoices.

### Decision 2.4: Preventing False Exclusions on Shared Entity Codes (`PEUS`, `FXW`) vs. True Intercompany GLs
* **Observation**: During error analysis on Company B's 3,000 seed rows, broad substring rules (`ic0..ic9`) and entity codes (`PEUS`, `FXW`) inside `gl_account_description` caused 917 false exclusions (dropping Include/Exclude accuracy to 68.40%) because `PEUS` and `FXW` are operating company codes shared by hundreds of valid procurement lines.
* **Decision**:
  1. Separate substring GL patterns (`blcn`, `interco`, `i/c`, `ic payable`, `royalt`) from exact non-procurement GL codes (`non_procurement_gl_exact`: `ic05`, `ic06`, `ic54`, `ic67`, `ppus`).
  2. Tighten seed-learned exclusion purity (`learn_seed_exclusion_signals`) to $\ge 98\%$ `NON_PROCUREMENT` concentration ($n \ge 3$) restricted to `gl_account` and `gl_account_description`.
  3. **Impact**: Restored Company B seed Include/Exclude accuracy from **68.40% $\rightarrow$ 94.63%** and Seed L3 classification accuracy from **63.22% $\rightarrow$ 97.22%**.

### Decision 2.5: Canonical Key Generation Without Supplier Names or Line Descriptions
* **Observation**: Company B provides `vendor_id` without a `supplier_name` column, and numerous rows have blank `line_description` fields.
* **Decision**: Derive `supplier_key` from zero-stripped normalized `vendor_id` when `supplier_name` is absent, and enrich blank descriptions with `gl_account_description`, `po_type`, and `transport_flag` (`"freight transport logistics shipping"`) prior to both item deduplication and multilingual embedding classification.

---

## 3. Generalization to Unseen Datasets (Company C)

### Decision 3.1: Leaf-First Multilingual Embeddings & Optional Local LLM Reranking
* **Observation**: Blind test datasets (Company C) provide no training labels and may contain unseen 3- or 4-level taxonomies and non-English descriptions.
* **Decision**: Encode taxonomy paths leaf-first (`Leaf | Parent | Root`) with `paraphrase-multilingual-MiniLM-L12-v2`, boost exact leaf/path lexical overlap, and enable optional local Ollama JSON re-ranking (`use_ollama_rerank: true`, `temperature=0.0`) on top-spend clusters—requiring zero paid API keys and zero code modifications to `src/`.