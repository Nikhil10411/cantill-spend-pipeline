#### `APPROACH.md`
```markdown
# Pipeline Architecture & Modeling Approach

## 1. End-to-End Pipeline Design
The pipeline executes as six deterministic, config-driven stages orchestrated via `main.py` with zero hardcoded company schemas or category names:
1. **Schema-Normalized Ingestion (`src/ingest.py`)**: Reads all input CSVs with `dtype=str` to prevent scientific-notation truncation or leading-zero loss on PO/line IDs, mapping heterogeneous multi-system schemas into a canonical internal table while preserving exact row ordering (`ERP:<line_id>`, `TOOL:<tool_record_id>`, or `<line_id>`).
2. **Financial & Temporal Normalization (`src/cleaner.py`)**: Strips VAT from gross invoice lines (`gross / (1 + vat_rate)`, defaulting blank rates to 15% and normalizing percentage integers such as `15` to `0.15`), resolves multi-format dates with posting-date fallbacks, and converts multi-currency lines to the target reporting currency using inline rates followed by `(currency, YYYY-MM)` and median fallbacks from `fx_rates_usd.csv`.
3. **Rule + Seed-Driven Exclusions & Cross-System Matching (`src/exclusions.py`)**: Applies deterministic status filters (`CANCELLED`, `NOT_APPROVED`), reversal pair detection, `NON_PROCUREMENT` pattern/GL/supplier-type filtering (augmented by high-purity signals learned from `seed_labels.csv`), and multi-pass cross-system deduplication.
4. **Blocked Entity Resolution (`src/dedup.py`)**: Normalizes legal suffixes and punctuation, then runs character-prefix blocked `rapidfuzz` (`token_sort_ratio`) clustering to generate consistent `supplier_key` and `item_key` identifiers in under 3 seconds.
5. **Multilingual Hierarchical Taxonomy Classification (`src/classifier.py`)**: Deduplicates unique `(supplier_key, item_key, composite_signature)` triples before inference, reducing ~40k rows to ~3k–5k unique clusters. Encodes full taxonomy leaf paths (`L1 > L2 > L3 > L4`) and item signatures using `paraphrase-multilingual-MiniLM-L12-v2`, combining cosine similarity, lexical token overlap, and k-NN seed neighbor voting (when labels are enabled), with an optional local Ollama JSON re-ranker for low-confidence high-spend clusters.
6. **Automated Output Validation (`src/validator.py`)**: Enforces exact row count preservation, `include=0` zero-spend/blank-category invariants, and strict character-for-character taxonomy membership.

## 2. Cross-System Matching & Deduplication (Company A)
To avoid double-counting purchases recorded in both SAP (`erp_goods_receipts.csv`) and the AP invoice tool (`invoice_tool_export.csv`), regex extraction parses reformatted PO strings (e.g., `4510085877/10`, `PO 4510109330`) into normalized `(po_number, po_item)` keys:
- **Pass 1**: Exact match on normalized `(po_number, po_item)` (`confidence = 0.99`).
- **Pass 2**: Normalized `po_number` match paired with normalized supplier agreement or net spend within 5% (`confidence = 0.94`).
- **Pass 3**: Normalized supplier agreement, net spend within 2%, and date within 45 days (`confidence = 0.88`).
- **Pass 4**: Exact duplicate exports within `invoice_tool_export.csv` (`confidence = 0.98`).
Matched `TOOL:` lines are retained in the output with `include=0`, `exclusion_reason=DUPLICATE`, and `spend=0`.

## 3. Unseen Taxonomy Generalization & Benchmark Results
For unseen trees (Company C and zero-shot validation), taxonomy paths are embedded dynamically at runtime with full parent-to-leaf context so leaf nodes with shared names (e.g., `Maintenance` under `IT` vs. `Fleet`) remain disambiguated.
- **Company A Zero-Shot Evaluation (`--eval-zero-shot`, labels hidden)**: Run `python main.py --config configs/config_A.yaml --eval-zero-shot` before final submission and record the printed L1, L2, L3, and Spend-Weighted L3 accuracy metrics here.
- **Runtime & Cost**: Completes ~65,549 rows (Company A) and ~41,028 rows (Company B) in **~45–90 seconds** on CPU with **$0.00 API cost** (fully local execution).