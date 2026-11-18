from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from src.ingest import get_reference_df, read_csv_safe


_ERP_ABBREV_MAP: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bstnedst\b", re.I), "stone dust"),
    (re.compile(r"\banlyz\b", re.I), "analyze sampling"),
    (re.compile(r"\bsmpl\b", re.I), "sampling"),
    (re.compile(r"\bdst\s+suprsn\b", re.I), "dust suppression ventilation"),
    (re.compile(r"\bair\s+xng\b", re.I), "ventilation structure air crossing"),
    (re.compile(r"\bsubsta\b", re.I), "substations cable repairs"),
    (re.compile(r"\bcnvy\b|\bconvyr\b|\bconvey\b", re.I), "conveyors belt"),
    (re.compile(r"\bbarng\b", re.I), "barring stone dust barrier"),
    (re.compile(r"\bbld\s+seal\b|\bseal\s+bld\b", re.I), "seal building underground explosive water seals"),
    (re.compile(r"\bbld\s+wal\s+ug\b", re.I), "underground structure wall building"),
    (re.compile(r"\bln\s+bor\b", re.I), "line boring continuous miner cm"),
    (re.compile(r"\bmini\s+intv\b|\bintv\b", re.I), "intervention overhaul continuous miner cm"),
    (re.compile(r"\botsd\s+scp\b|\bovhl\b", re.I), "full overhaul refurbishment"),
    (re.compile(r"\bf/serv\b|\bfld\s+serv\b", re.I), "field service maintenance"),
    (re.compile(r"\brod\s+anchr\b|\banchr\b|\breb\s+qc\b", re.I), "roof bolts anchor rod"),
    (re.compile(r"\brod\s+dril\b|\baugr\b", re.I), "roof support accessories drill steel rod"),
    (re.compile(r"\bbit\s+rck\b|\brck\s+bit\b", re.I), "roof support drill bits rock bit"),
    (re.compile(r"\bhex\b.*\bmm\b", re.I), "roof support drill steel extension"),
    (re.compile(r"\bthin\s+spry\b", re.I), "roof support services spray lining"),
    (re.compile(r"\bfan\s+elec\b|\bflng\b", re.I), "mining ventilation and hoists electric fan"),
    (re.compile(r"\bpick\b.*\bdrum\s+cutter\b|\bsleeve\s+pick\b", re.I), "picks and sleeves cutter"),
    (re.compile(r"\bseamex\b|\bexplos\b", re.I), "mining explosives gel"),
    (re.compile(r"\bbreakr\b|\bfdbkr\b|\bcrush\b", re.I), "feeder breakers and crushers"),
    (re.compile(r"\bstarter\s+automotive\b|\bengine\s+starter\b", re.I), "flameproof diesel equipment starters"),
    (re.compile(r"\bmntn\b|\bmaint\b", re.I), "maintenance service overhaul"),
    (re.compile(r"\brepr\b", re.I), "repair maintenance spares"),
    (re.compile(r"\bhyd\b", re.I), "hydraulic"),
    (re.compile(r"\bcyl\b|\bcyldr\b", re.I), "cylinder"),
    (re.compile(r"\bele\b|\belec\b", re.I), "electrical"),
]


def _to_str_list(series: pd.Series) -> List[str]:
    return [str(v).strip() for v in series.fillna("").tolist()]


def _clean_supplier_for_text(name: str) -> str:
    s = re.sub(r"\b(pty|ltd|limited|inc|corp|corporation|llc|gmbh|sa|co|plc)\b\.?", "", name, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", s).strip(" ,.-")


def expand_erp_shorthand(text: str) -> str:
    if not text:
        return ""
    # Replace ERP punctuation separators (; , / & :) with spaces while keeping alphanumeric tokens intact
    s = re.sub(r"P/N:[A-Za-z0-9\-_]+", " ", text, flags=re.IGNORECASE)
    s = re.sub(r"[;,/&:()]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    expanded = s
    for pat, repl in _ERP_ABBREV_MAP:
        if pat.search(s):
            expanded = f"{expanded} {repl}"
    return re.sub(r"\s+", " ", expanded).strip()


def build_taxonomic_item_text(df: pd.DataFrame) -> List[str]:
    """Build rich, abbreviation-expanded procurement text for zero-shot and k-NN taxonomy matching."""
    desc_l = _to_str_list(df["description"])
    gl_l = _to_str_list(df["gl_account_description"])
    po_l = _to_str_list(df["po_type"])
    sup_l = [_clean_supplier_for_text(v) for v in _to_str_list(df["supplier_name"])]
    stype_l = _to_str_list(df["supplier_type"])
    spend_type_l = _to_str_list(df["spend_type"]) if "spend_type" in df.columns else [""] * len(df)
    mat_type_l = _to_str_list(df["material_type"]) if "material_type" in df.columns else [""] * len(df)
    tr_l = [
        "freight transport logistics shipping courier" if v.lower() in ("1", "y", "yes", "true") else ""
        for v in _to_str_list(df["transport_flag"])
    ]

    texts: List[str] = []
    for desc, gl, po, sup, stype, spt, mtype, tr in zip(
        desc_l, gl_l, po_l, sup_l, stype_l, spend_type_l, mat_type_l, tr_l
    ):
        parts: List[str] = []
        if desc:
            parts.append(expand_erp_shorthand(desc))
        if gl and gl.lower() != desc.lower():
            parts.append(expand_erp_shorthand(gl))
        if spt:
            parts.append(f"Type: {spt}")
        if mtype and mtype.lower() not in ("none", "nan"):
            parts.append(mtype)
        if tr:
            parts.append(tr)
        if sup and not sup.isdigit():
            parts.append(f"Supplier: {sup}")
        if stype and stype.lower() not in ("standard", "regular", "vendor", "supplier"):
            parts.append(stype)
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


def _expand_taxonomy_leaf_text(clean_vals: List[str], extra_vals: List[str]) -> str:
    """Enrich taxonomy leaf text with domain synonym expansions (e.g. CM -> Continuous Miner, RB -> Roof Bolter)."""
    unique_ordered: List[str] = []
    for v in reversed(clean_vals):
        if v not in unique_ordered:
            unique_ordered.append(v)
    for ex_v in extra_vals:
        if ex_v:
            unique_ordered.append(ex_v)
    base = " | ".join(unique_ordered)
    low = base.lower()
    synonyms: List[str] = []
    if re.search(r"\bcm\b", low):
        synonyms.append("continuous miner mc470")
    if re.search(r"\brb\b", low):
        synonyms.append("roof bolter")
    if re.search(r"\bsc\b", low):
        synonyms.append("shuttle car")
    if "ventilation" in low:
        synonyms.append("fan electric air crossing dust suppression")
    if "roof bolts" in low:
        synonyms.append("anchor rod reb")
    if "drill bits" in low:
        synonyms.append("bit rock")
    if "drill steel" in low:
        synonyms.append("extension rod hex")
    if "stone dust" in low and "sampling" not in low and "barrier" not in low:
        synonyms.append("powder stnedst")
    if "sampling" in low:
        synonyms.append("analyze sample h2o ph chem test")
    return f"{base} | {' '.join(synonyms)}" if synonyms else base


def resolve_ollama_model(url: str, preferred_model: str) -> Optional[str]:
    try:
        base = url.split("/api/")[0]
        r = requests.get(f"{base}/api/tags", timeout=1.5)
        if r.status_code != 200:
            return None
        models = [m.get("name", "") for m in r.json().get("models", []) if m.get("name")]
        if not models:
            return None
        if preferred_model in models:
            return preferred_model
        for candidate in ("llama-3-8b-instruct:latest", "llama3.1:latest", "qwen2.5:7b", "phi3:latest"):
            if candidate in models:
                return candidate
        return models[0]
    except Exception:
        return None


def ollama_classify_batch(
    batch_items: List[Tuple[int, str, List[str]]],
    url: str,
    model: str,
) -> Dict[int, Tuple[int, float]]:
    lines: List[str] = []
    for local_id, text, cands in batch_items:
        opts = " ; ".join([f"[{idx}] {path}" for idx, path in enumerate(cands)])
        lines.append(f"ITEM_{local_id}: {text}\nOPTIONS_{local_id}: {opts}")

    prompt = (
        "You are an expert procurement spend taxonomy classifier. For each ITEM below, select the single best matching "
        "0-based option index from its OPTIONS list.\n\n"
        + "\n\n".join(lines)
        + '\n\nReturn strictly valid JSON mapping each item id string ("0", "1", ...) to {"idx": <int>, "conf": <float 0.60-0.96>}.'
    )
    try:
        resp = requests.post(
            url,
            json={
                "model": model,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 32 * len(batch_items)},
            },
            timeout=45,
        )
        resp.raise_for_status()
        parsed = json.loads(str(resp.json().get("response", "{}")))
        results: Dict[int, Tuple[int, float]] = {}
        for local_id, _, cands in batch_items:
            entry = parsed.get(str(local_id)) or parsed.get(f"ITEM_{local_id}")
            if isinstance(entry, dict) and "idx" in entry:
                chosen_idx = int(entry["idx"])
                conf = float(entry.get("conf", 0.82))
                if 0 <= chosen_idx < len(cands):
                    results[local_id] = (chosen_idx, max(0.55, min(0.96, conf)))
        return results
    except Exception:
        return {}


def classify_taxonomy(
    df: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    cfg: Dict[str, Any],
    force_zero_shot: bool = False,
) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    level_cols = [
        c for c in taxonomy_df.columns
        if str(c).lower().startswith("category_l") or str(c).lower() in ("l1", "l2", "l3", "l4")
    ]
    if not level_cols:
        level_cols = [str(c) for c in taxonomy_df.columns]
    extra_tax_cols = [str(c) for c in taxonomy_df.columns if str(c) not in level_cols]
    num_levels = len(level_cols)

    for lvl in ["l1", "l2", "l3", "l4"]:
        df[lvl] = ""
    df["confidence"] = pd.to_numeric(df["_excl_confidence"], errors="coerce").fillna(0.95)

    inc_mask = (df["include"] == 1).to_numpy(dtype=bool)
    if not np.any(inc_mask):
        return df

    # 1. Build leaf-focused taxonomy representations + L1 parent representations for hierarchy regularization
    tax_embed_texts: List[str] = []
    tax_l1_texts: List[str] = []
    tax_display_paths: List[str] = []
    tax_tuples: List[Tuple[str, ...]] = []

    tax_cols_lists: List[List[str]] = [_to_str_list(taxonomy_df[c]) for c in level_cols]
    extra_cols_lists: List[List[str]] = [_to_str_list(taxonomy_df[c]) for c in extra_tax_cols]

    for row_idx, row_vals in enumerate(zip(*tax_cols_lists)):
        vals = tuple(str(v).strip() for v in row_vals)
        clean_vals = [v for v in vals if v]
        tax_display_paths.append(" > ".join(clean_vals))
        tax_tuples.append(vals)
        tax_l1_texts.append(" | ".join(clean_vals[:2]) if len(clean_vals) >= 2 else clean_vals[0])

        extra_vals = [ex_col[row_idx] for ex_col in extra_cols_lists]
        tax_embed_texts.append(_expand_taxonomy_leaf_text(clean_vals, extra_vals))

    valid_tax_set = set(tax_tuples)

    # 2. Load multilingual sentence-transformer
    model_name = str(cfg.get("classification", {}).get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2"))
    model = SentenceTransformer(model_name)
    tax_embeddings = model.encode(tax_embed_texts, normalize_embeddings=True, show_progress_bar=False)
    tax_l1_embeddings = model.encode(tax_l1_texts, normalize_embeddings=True, show_progress_bar=False)

    # 3. Build rich item texts & deduplicate unique clusters before inference
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

    # Hybrid zero-shot similarity: Leaf Dense (62%) + Parent L1/L2 Dense (16%) + Word/Char TF-IDF (22%)
    dense_leaf_sim = np.matmul(item_embeddings, tax_embeddings.T)
    dense_parent_sim = np.matmul(item_embeddings, tax_l1_embeddings.T)

    try:
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        tfidf_tax = tfidf.fit_transform(tax_embed_texts)
        tfidf_itm = tfidf.transform(unique_item_texts)
        tfidf_sim = np.asarray(linear_kernel(tfidf_itm, tfidf_tax), dtype=np.float64)
    except Exception:
        tfidf_sim = np.zeros_like(dense_leaf_sim)

    sim_matrix = (0.62 * dense_leaf_sim) + (0.16 * dense_parent_sim) + (0.22 * tfidf_sim)

    tax_token_sets = [set(re.findall(r"[a-z0-9]{3,}", p.lower())) for p in tax_embed_texts]
    leaf_token_sets = [set(re.findall(r"[a-z0-9]{3,}", [v for v in t if v][-1].lower())) for t in tax_tuples]

    for i, itm_txt in enumerate(unique_item_texts):
        sig_tokens = set(re.findall(r"[a-z0-9]{3,}", itm_txt.lower()))
        if sig_tokens:
            path_ov = np.array([len(sig_tokens & t_set) / max(len(t_set), 1) for t_set in tax_token_sets])
            leaf_ov = np.array([len(sig_tokens & l_set) / max(len(l_set), 1) for l_set in leaf_token_sets])
            sim_matrix[i] += (0.15 * path_ov) + (0.20 * leaf_ov)

    # 4. Seed / Golden Label k-NN & Master-Data Lookup (supports both current df and reference historical df for A2/B2)
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
            ref_df = get_reference_df(cfg, df)
            if "sig_text" not in ref_df.columns:
                ref_df = ref_df.copy()
                ref_df["sig_text"] = build_full_signature(ref_df)
            if "supplier_key" not in ref_df.columns:
                ref_df["supplier_key"] = _to_str_list(ref_df["supplier_id"])

            joined = ref_df[["record_id", "supplier_key", "material_number", "gl_account", "sig_text"]].merge(
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

    # 5. Initial Embedding + k-NN Prediction per Cluster
    group_pred_map: Dict[Tuple[str, str, str], Tuple[Tuple[str, ...], float]] = {}
    top_cand_indices_per_cluster: List[np.ndarray] = []
    zero_shot_cluster_indices: List[int] = []

    knn_sims = None
    if sig_embeddings is not None and seed_embeddings is not None and len(seed_labels_list) > 0:
        knn_sims = np.matmul(sig_embeddings, seed_embeddings.T)

    ug_sup = _to_str_list(unique_groups["supplier_key"])
    ug_itm = _to_str_list(unique_groups["item_key"])
    ug_sig = _to_str_list(unique_groups["sig_text"])
    ug_itxt = _to_str_list(unique_groups["item_text_clean"])
    ug_spend = pd.to_numeric(unique_groups["group_spend"], errors="coerce").fillna(0.0).tolist()

    for i, (sk_u, ik_u, sig_u, grp_spend) in enumerate(zip(ug_sup, ug_itm, ug_sig, ug_spend)):
        scores = sim_matrix[i]
        top_indices = np.argsort(scores)[::-1][:15]
        top_cand_indices_per_cluster.append(top_indices)

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

        if (not used_knn) and float(grp_spend) > 0:
            zero_shot_cluster_indices.append(i)

        group_pred_map[(sk_u, ik_u, sig_u)] = (chosen_tup, round(conf, 4))

    # 6. Batched Local Ollama Re-Ranking on Top-Spend Zero-Shot Clusters (if Ollama is running and enabled)
    cls_cfg = cfg.get("classification", {})
    ollama_url = str(os.getenv("OLLAMA_URL", cls_cfg.get("ollama_url", "http://localhost:11434/api/generate")))
    preferred_model = str(os.getenv("OLLAMA_MODEL", cls_cfg.get("ollama_model", "llama-3-8b-instruct:latest")))
    active_ollama_model = (
        resolve_ollama_model(ollama_url, preferred_model)
        if bool(cls_cfg.get("use_ollama_rerank", False))
        else None
    )
    max_ollama_clusters = int(cls_cfg.get("max_ollama_clusters", 500))
    batch_size_llm = 12

    if active_ollama_model and zero_shot_cluster_indices:
        target_clusters = zero_shot_cluster_indices[:max_ollama_clusters]
        print(f"      Ollama active ({active_ollama_model}): batch re-ranking {len(target_clusters)} top-spend zero-shot clusters...")
        for b_start in range(0, len(target_clusters), batch_size_llm):
            batch_idxs = target_clusters[b_start : b_start + batch_size_llm]
            batch_payload = [
                (
                    pos,
                    ug_itxt[c_idx],
                    [tax_display_paths[int(t_idx)] for t_idx in top_cand_indices_per_cluster[c_idx]],
                )
                for pos, c_idx in enumerate(batch_idxs)
            ]
            batch_res = ollama_classify_batch(batch_payload, ollama_url, active_ollama_model)
            for pos, (r_idx, r_conf) in batch_res.items():
                c_idx = batch_idxs[pos]
                chosen_tax_idx = int(top_cand_indices_per_cluster[c_idx][r_idx])
                group_pred_map[(ug_sup[c_idx], ug_itm[c_idx], ug_sig[c_idx])] = (
                    tax_tuples[chosen_tax_idx],
                    round(r_conf, 4),
                )

    # 7. Map predictions back to all included rows
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
