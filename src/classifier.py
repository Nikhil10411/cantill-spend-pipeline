from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from src.ingest import read_csv_safe


def _to_str_list(series: pd.Series) -> List[str]:
    return [str(v).strip() for v in series.fillna("").tolist()]


def build_taxonomic_item_text(df: pd.DataFrame) -> List[str]:
    """Build clean product/service text without noisy IDs or cost-centre codes for zero-shot taxonomy matching."""
    desc_l = _to_str_list(df["description"])
    gl_l = _to_str_list(df["gl_account_description"])
    po_l = _to_str_list(df["po_type"])
    tr_l = [
        "freight transport logistics shipping" if v.lower() in ("1", "y", "yes", "true") else ""
        for v in _to_str_list(df["transport_flag"])
    ]

    texts: List[str] = []
    for desc, gl, po, tr in zip(desc_l, gl_l, po_l, tr_l):
        parts: List[str] = []
        if desc:
            parts.append(desc)
        if gl and gl.lower() != desc.lower():
            parts.append(gl)
        if tr:
            parts.append(tr)
        if not parts and po:
            parts.append(po)
        texts.append(" | ".join(parts) if parts else "general industrial supplies and services")
    return texts


def build_full_signature(df: pd.DataFrame) -> List[str]:
    """Full signature including supplier & cost centre for k-NN matching against labelled rows."""
    item_texts = build_taxonomic_item_text(df)
    sup_l = _to_str_list(df["supplier_name"])
    stype_l = _to_str_list(df["supplier_type"])
    cc_l = _to_str_list(df["cost_centre"])

    sigs: List[str] = []
    for itm, sup, stype, cc in zip(item_texts, sup_l, stype_l, cc_l):
        extra = [x for x in (sup, stype, cc) if x]
        sigs.append(f"{itm} || {' | '.join(extra)}" if extra else itm)
    return sigs


def check_ollama_available(url: str) -> bool:
    try:
        base = url.split("/api/")[0]
        r = requests.get(f"{base}/api/tags", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def ollama_classify_top_candidate(
    item_text: str,
    supplier_hint: str,
    candidate_paths: List[str],
    url: str,
    model: str,
) -> Optional[Tuple[int, float]]:
    options_block = "\n".join([f"{i}: {p}" for i, p in enumerate(candidate_paths)])
    prompt = (
        "You are a senior procurement spend analyst. Pick the single most accurate category index (0-based) for this purchase.\n"
        f"Item Description: {item_text}\n"
        f"Supplier Context: {supplier_hint or 'N/A'}\n"
        f"Candidate Categories:\n{options_block}\n"
        'Return strictly valid JSON with no extra text: {"index": <int>, "confidence": <float 0.50-0.98>}'
    )
    try:
        resp = requests.post(
            url,
            json={
                "model": model,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 40},
            },
            timeout=12,
        )
        resp.raise_for_status()
        parsed = json.loads(str(resp.json().get("response", "{}")))
        idx = int(parsed.get("index", 0))
        conf = float(parsed.get("confidence", 0.78))
        if 0 <= idx < len(candidate_paths):
            return idx, max(0.50, min(0.98, conf))
    except Exception:
        return None
    return None


def classify_taxonomy(
    df: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    cfg: Dict[str, Any],
    force_zero_shot: bool = False,
) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    level_cols = [str(c) for c in taxonomy_df.columns]
    num_levels = len(level_cols)

    for lvl in ["l1", "l2", "l3", "l4"]:
        df[lvl] = ""
    df["confidence"] = pd.to_numeric(df["_excl_confidence"], errors="coerce").fillna(0.95)

    inc_mask = (df["include"] == 1).to_numpy(dtype=bool)
    if not np.any(inc_mask):
        return df

    # 1. Build leaf-focused taxonomy representations (deduplicating repeated parent-child words)
    tax_embed_texts: List[str] = []
    tax_display_paths: List[str] = []
    tax_tuples: List[Tuple[str, ...]] = []

    tax_cols_lists: List[List[str]] = [_to_str_list(taxonomy_df[c]) for c in level_cols]
    for row_vals in zip(*tax_cols_lists):
        vals = tuple(str(v).strip() for v in row_vals)
        clean_vals = [v for v in vals if v]
        tax_display_paths.append(" > ".join(clean_vals))
        tax_tuples.append(vals)

        unique_ordered: List[str] = []
        for v in reversed(clean_vals):
            if v not in unique_ordered:
                unique_ordered.append(v)
        tax_embed_texts.append(" | ".join(unique_ordered))

    valid_tax_set = set(tax_tuples)

    # 2. Load multilingual sentence-transformer
    model_name = str(cfg.get("classification", {}).get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2"))
    model = SentenceTransformer(model_name)

    tax_embeddings = model.encode(tax_embed_texts, normalize_embeddings=True, show_progress_bar=False)

    # 3. Build clean item texts & deduplicate unique clusters before inference
    item_texts = build_taxonomic_item_text(df)
    full_sigs = build_full_signature(df)
    df["item_text_clean"] = item_texts
    df["sig_text"] = full_sigs
    inc_df = df.loc[inc_mask].copy()

    unique_groups = (
        inc_df.groupby(["supplier_key", "item_key", "item_text_clean", "sig_text"], as_index=False)
        .agg(
            group_spend=("spend", "sum"),
            row_count=("record_id", "count"),
            sup_sample=("supplier_name", "first"),
        )
        .sort_values("group_spend", ascending=False)
        .reset_index(drop=True)
    )

    lbl_cfg = cfg.get("labels", {})
    use_labels = bool(lbl_cfg.get("use_labels", False)) and (not force_zero_shot) and bool(lbl_cfg.get("path"))

    unique_item_texts = _to_str_list(unique_groups["item_text_clean"])
    unique_full_sigs = _to_str_list(unique_groups["sig_text"])

    item_embeddings = model.encode(unique_item_texts, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    sig_embeddings = (
        model.encode(unique_full_sigs, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
        if use_labels
        else None
    )

    # Zero-shot cosine similarity matrix
    sim_matrix = np.matmul(item_embeddings, tax_embeddings.T)

    # Lexical token overlap boost (weighted toward leaf tokens)
    tax_token_sets = [set(re.findall(r"[a-z0-9]{3,}", p.lower())) for p in tax_embed_texts]
    leaf_token_sets = [set(re.findall(r"[a-z0-9]{3,}", [v for v in t if v][-1].lower())) for t in tax_tuples]

    for i, itm_txt in enumerate(unique_item_texts):
        sig_tokens = set(re.findall(r"[a-z0-9]{3,}", itm_txt.lower()))
        if sig_tokens:
            path_ov = np.array([len(sig_tokens & t_set) / max(len(t_set), 1) for t_set in tax_token_sets])
            leaf_ov = np.array([len(sig_tokens & l_set) / max(len(l_set), 1) for l_set in leaf_token_sets])
            sim_matrix[i] += (0.14 * path_ov) + (0.16 * leaf_ov)

    # 4. Optional Seed / Golden Label k-NN & Exact Lookup
    exact_record_labels: Dict[str, Tuple[str, ...]] = {}
    mat_to_label: Dict[str, Tuple[str, ...]] = {}
    gl_to_label: Dict[str, Tuple[str, ...]] = {}
    sup_to_label: Dict[str, Tuple[str, ...]] = {}
    seed_embeddings = None
    seed_labels_list: List[Tuple[str, ...]] = []

    if use_labels:
        lbl_df = read_csv_safe(str(lbl_cfg["path"]))
        lbl_id_col = "record_id" if "record_id" in lbl_df.columns else "line_id"
        lbl_level_cols = [c for c in ["category_l1", "category_l2", "category_l3", "category_l4"] if c in lbl_df.columns]

        if "include" in lbl_df.columns:
            inc_flags = _to_str_list(lbl_df["include"])
            lbl_df = lbl_df[[flag == "1" for flag in inc_flags]].copy()

        valid_rows: List[Tuple[str, Tuple[str, ...]]] = []
        id_vals = _to_str_list(lbl_df[lbl_id_col])
        lvl_cols_data: List[List[str]] = [_to_str_list(lbl_df[c]) for c in lbl_level_cols]

        for row_idx, rid in enumerate(id_vals):
            tup = tuple(lvl_cols_data[col_i][row_idx] for col_i in range(len(lvl_cols_data)))
            if rid and tup in valid_tax_set:
                exact_record_labels[rid] = tup
                valid_rows.append((rid, tup))

        if valid_rows:
            seed_map_df = pd.DataFrame(valid_rows, columns=["record_id", "label_tup"])
            joined = df[["record_id", "supplier_key", "material_number", "gl_account", "sig_text"]].merge(
                seed_map_df, on="record_id", how="inner"
            )
            if not joined.empty:
                for col, target_dict in [
                    ("material_number", mat_to_label),
                    ("gl_account", gl_to_label),
                    ("supplier_key", sup_to_label),
                ]:
                    col_vals = _to_str_list(joined[col])
                    sub = joined[[len(v) > 0 for v in col_vals]]
                    for key_val, grp in sub.groupby(col):
                        vc = grp["label_tup"].value_counts()
                        if len(grp) >= 2 and (int(vc.iloc[0]) / len(grp)) >= 0.75:
                            target_dict[str(key_val)] = vc.index[0]

                unique_seed = joined[["sig_text", "label_tup"]].drop_duplicates(subset=["sig_text"]).head(8000)
                seed_embeddings = model.encode(
                    _to_str_list(unique_seed["sig_text"]),
                    normalize_embeddings=True,
                    batch_size=256,
                    show_progress_bar=False,
                )
                seed_labels_list = list(unique_seed["label_tup"].tolist())

    # 5. Check optional local Ollama availability for top-spend clusters
    cls_cfg = cfg.get("classification", {})
    ollama_url = str(os.getenv("OLLAMA_URL", cls_cfg.get("ollama_url", "http://localhost:11434/api/generate")))
    ollama_model = str(os.getenv("OLLAMA_MODEL", cls_cfg.get("ollama_model", "qwen2.5:7b")))
    use_ollama = bool(cls_cfg.get("use_ollama_rerank", False)) and check_ollama_available(ollama_url)
    max_ollama_calls = int(cls_cfg.get("max_ollama_clusters", 350))
    ollama_calls_made = 0

    if use_ollama:
        print(f"      Ollama active ({ollama_model}): re-ranking up to {max_ollama_calls} top-spend clusters...")

    group_pred_map: Dict[Tuple[str, str, str], Tuple[Tuple[str, ...], float]] = {}

    knn_sims = None
    if sig_embeddings is not None and seed_embeddings is not None and len(seed_labels_list) > 0:
        knn_sims = np.matmul(sig_embeddings, seed_embeddings.T)

    ug_sup = _to_str_list(unique_groups["supplier_key"])
    ug_itm = _to_str_list(unique_groups["item_key"])
    ug_sig = _to_str_list(unique_groups["sig_text"])
    ug_itxt = _to_str_list(unique_groups["item_text_clean"])
    ug_sup_name = _to_str_list(unique_groups["sup_sample"])
    ug_spend = pd.to_numeric(unique_groups["group_spend"], errors="coerce").fillna(0.0).tolist()

    for i, (sk_u, ik_u, sig_u, itxt_u, sup_n, grp_spend) in enumerate(
        zip(ug_sup, ug_itm, ug_sig, ug_itxt, ug_sup_name, ug_spend)
    ):
        scores = sim_matrix[i]
        top_indices = np.argsort(scores)[::-1][:7]
        best_idx = int(top_indices[0])
        best_score = float(scores[best_idx])
        second_score = float(scores[top_indices[1]]) if len(top_indices) > 1 else 0.0

        chosen_tup = tax_tuples[best_idx]
        margin = max(0.0, best_score - second_score)
        conf = min(0.96, max(0.35, 0.35 + (best_score * 0.65) + (margin * 0.50)))
        used_knn = False

        if knn_sims is not None:
            k_row = knn_sims[i]
            top_k = np.argsort(k_row)[::-1][:5]
            top_k_sim = float(k_row[top_k[0]])
            if top_k_sim >= 0.72:
                vote_scores: Dict[Tuple[str, ...], float] = {}
                for k_idx in top_k:
                    s_val = float(k_row[int(k_idx)])
                    if s_val >= 0.65:
                        lbl = seed_labels_list[int(k_idx)]
                        vote_scores[lbl] = vote_scores.get(lbl, 0.0) + s_val
                if vote_scores:
                    chosen_tup = max(vote_scores.items(), key=lambda x: x[1])[0]
                    conf = min(0.97, max(conf, top_k_sim))
                    used_knn = True

        # Re-rank top-spend zero-shot clusters with local Ollama if available
        if use_ollama and (not used_knn) and ollama_calls_made < max_ollama_calls and float(grp_spend) > 0:
            cand_paths = [tax_display_paths[int(idx)] for idx in top_indices]
            reranked = ollama_classify_top_candidate(itxt_u, sup_n, cand_paths, ollama_url, ollama_model)
            ollama_calls_made += 1
            if reranked is not None:
                r_idx, r_conf = reranked
                chosen_tup = tax_tuples[int(top_indices[r_idx])]
                conf = r_conf

        group_pred_map[(sk_u, ik_u, sig_u)] = (chosen_tup, round(conf, 4))

    # 6. Map predictions back to all included rows
    l_arrays = [df[f"l{k+1}"].to_numpy(dtype=object) for k in range(4)]
    conf_arr = df["confidence"].to_numpy(dtype=np.float64)

    rid_list = _to_str_list(df["record_id"])
    sk_list = _to_str_list(df["supplier_key"])
    ik_list = _to_str_list(df["item_key"])
    mat_list = _to_str_list(df["material_number"])
    gl_list = _to_str_list(df["gl_account"])

    for idx in np.flatnonzero(inc_mask):
        rid = rid_list[idx]
        sk = sk_list[idx]
        ik = ik_list[idx]
        sig = full_sigs[idx]
        mat = mat_list[idx]
        gl = gl_list[idx]

        if use_labels and rid in exact_record_labels:
            tup, c = exact_record_labels[rid], 0.99
        elif use_labels and mat and mat in mat_to_label:
            tup, c = mat_to_label[mat], 0.94
        elif use_labels and gl and gl in gl_to_label:
            tup, c = gl_to_label[gl], 0.90
        elif use_labels and sk in sup_to_label and group_pred_map[(sk, ik, sig)][1] < 0.62:
            tup, c = sup_to_label[sk], 0.85
        else:
            tup, c = group_pred_map[(sk, ik, sig)]

        for k in range(num_levels):
            l_arrays[k][idx] = tup[k]
        conf_arr[idx] = c

    for k in range(4):
        df[f"l{k+1}"] = l_arrays[k]
    df["confidence"] = np.round(conf_arr, 4)
    df.drop(columns=["item_text_clean", "sig_text"], inplace=True, errors="ignore")
    return df