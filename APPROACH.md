# Pipeline Architecture & Modeling Approach

## 1. System Overview
The pipeline executes as six deterministic, config-driven stages orchestrated via `main.py`, with **zero hardcoded company schemas, supplier names, GL codes, or taxonomy strings** inside `src/`. All dataset-specific mappings and thresholds live in declarative YAML configuration files (`configs/config_A.yaml`, `configs/config_B.yaml`, and `submission/config_C.yaml`).

```text
  Raw Multi-System CSVs + Taxonomy CSV + Optional Seed Labels
                             │
                             ▼
  [1] src/ingest.py      ──► Canonical Schema Alignment (dtype=str, ID Prefixing, PO Regex)
                             │
                             ▼
  [2] src/cleaner.py     ──► Net Spend Normalization (VAT Stripping, Multi-Tier FX, Date Repair)
                             │
                             ▼
  [3] src/exclusions.py  ──► Status, Reversal, Non-Procurement (Rule + Seed Purity) & Cross-System Dedup
                             │
                             ▼
  [4] src/dedup.py       ──► Prefix-Blocked Fuzzy Entity Resolution (supplier_key, item_key)
                             │
                             ▼
  [5] src/classifier.py  ──► Cluster-Deduplicated Multilingual Embedding + Lexical + Seed k-NN / LLM Rerank
                             │
                             ▼
  [6] src/validator.py   ──► Schema, Row-Count, Zero-Spend & Leaf Membership Verification