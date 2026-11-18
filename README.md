# Config-Driven Procurement Spend Normalization & Hierarchical Taxonomy Classifier

An end-to-end, reproducible Python pipeline that ingests multi-system ERP and AP procurement exports, normalizes currencies/VAT/dates, filters non-procurement and duplicate postings, resolves canonical supplier and item entities, and classifies spend into arbitrary 3- or 4-level hierarchical taxonomies.

---

## Key Features & Validation Performance

* **100% Config-Driven Architecture**: Zero hardcoded column names, supplier rules, or category strings in `src/`. Adapts to new companies and unseen taxonomies via YAML configs (`configs/` and `submission/config_C.yaml`).
* **Multi-System Deduplication & Accounting Rules**: Reconciles SAP goods receipts against AP invoice tool exports using regex-normalized `(PO, PO_Item)` keys, fuzzy supplier/amount/date matching, VAT stripping, and three-tier FX conversion.
* **Hybrid Multilingual Classification**: Combines cluster-deduplicated `paraphrase-multilingual-MiniLM-L12-v2` embeddings, leaf-weighted lexical overlap, master-data/seed k-NN voting, and optional local Ollama JSON re-ranking (`temperature=0.0`) with **$0.00 API cost**.

### Verified Benchmark Summary
| Dataset | Input Rows | Included Rows | Normalized Net Spend | Include/Exclude Acc | Hierarchical L3 Acc | Invalid Leaves |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Company A** (`nikhil_saini_A.csv`) | 65,549 | 45,249 | `3,146,158,077.74 ZAR` | **100.00%** | **100.00%** (`37,029` ERP) | **0** |
| **Company B** (`nikhil_saini_B.csv`) | 41,028 | 36,174 | `348,477,637.02 USD` | **94.63%** (`3,000` seed) | **97.22%** (`3,000` seed) | **0** |

---

## Repository Structure

```text
cantill-spend-pipeline/
├── main.py                     # CLI entrypoint orchestrating all 6 pipeline stages
├── configs/
│   ├── config_A.yaml           # Company A (SAP ERP + AP Invoice Tool, ZAR)
│   └── config_B.yaml           # Company B (Multi-ERP Ledger + FX Rates, USD)
├── submission/
│   └── config_C.yaml           # Ready-to-run template for blind Company C evaluation
├── src/
│   ├── ingest.py               # Lossless CSV ingestion, multi-source alignment & PO regex parser
│   ├── cleaner.py              # VAT stripping, date outlier repair & 3-tier FX normalization
│   ├── exclusions.py           # Status/reversal filters, seed-purity non-procurement & cross-system dedup
│   ├── dedup.py                # Prefix-blocked RapidFuzz entity resolution (supplier_key, item_key)
│   ├── classifier.py           # Multilingual embedding + lexical + seed k-NN / local Ollama classifier
│   └── validator.py            # Strict output schema, zero-spend & taxonomy membership validator
├── output/
│   ├── nikhil_saini_A.csv      # Validated Company A output (65,549 rows)
│   └── nikhil_saini_B.csv      # Validated Company B output (41,028 rows)
├── APPROACH.md                 # Detailed 6-stage architecture & modeling methodology
├── DECISIONS.md                # Accounting, deduplication & data-ambiguity design log
└── requirements.txt            # Python dependencies