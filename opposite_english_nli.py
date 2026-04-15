# opposites_he_nli.py
from __future__ import annotations
import re, json, os
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import hnswlib

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

# =============================== Topic filter (English) ===============================

def make_topic_mask_english(
    meta: pd.DataFrame,
    topic_query: str | Iterable[str],
    require_all: bool = True
) -> np.ndarray:
    """
    Build a boolean mask over meta['message'] (Hebrew dataset) for rows containing the keywords.
    - topic_query: string with words OR an iterable of keywords.
    - require_all: True => all keywords must appear (AND), False => any keyword (OR).
    """
    titles = meta["comment_text"].astype(str)

    if isinstance(topic_query, str):
        kws = [t for t in re.split(r"[\W_]+", topic_query) if t]
    else:
        kws = [str(t) for t in topic_query if str(t).strip()]

    if not kws:
        return np.ones(len(meta), dtype=bool)

    if require_all:
        mask = np.ones(len(meta), dtype=bool)
        for kw in kws:
            mask &= titles.str.contains(re.escape(kw), case=False, na=False).values
    else:
        mask = np.zeros(len(meta), dtype=bool)
        for kw in kws:
            mask |= titles.str.contains(re.escape(kw), case=False, na=False).values
    return mask


# =============================== Exact cosine top-k over subset ===============================

def topk_min_cosine_over_indices(
    q: np.ndarray,
    embs: np.ndarray,
    idxs: np.ndarray,
    k: int = 3,
    chunk: int = 50_000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (best_idxs, best_sims) where best_idxs are GLOBAL indices into embs/meta,
    and best_sims are the corresponding cosine similarities (ascending).
    Assumes embs rows and q are L2-normalized (cosine = dot).
    """
    k = min(k, len(idxs))
    if k <= 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32)

    best_idxs = np.empty(0, dtype=int)
    best_sims = np.empty(0, dtype=np.float32)

    for s in range(0, len(idxs), chunk):
        block = idxs[s:s + chunk]
        sims = embs[block] @ q  # exact cosine

        # take k smallest from this block
        k_block = min(k, sims.size)
        part = np.argpartition(sims, k_block - 1)[:k_block]
        cand_idxs = block[part]
        cand_sims = sims[part]

        # merge with global top-k so far
        if best_idxs.size == 0:
            best_idxs = cand_idxs
            best_sims = cand_sims
        else:
            best_idxs = np.concatenate([best_idxs, cand_idxs])
            best_sims = np.concatenate([best_sims, cand_sims])
            if best_sims.size > k:
                keep = np.argpartition(best_sims, k - 1)[:k]
                best_idxs = best_idxs[keep]
                best_sims = best_sims[keep]

    order = np.argsort(best_sims)
    return best_idxs[order], best_sims[order]


# =============================== Cosine "most opposite" (English) ===============================

def most_opposite_in_topic_english(
    query_text: str,
    topic_query: str | Iterable[str],
    *,
    meta: pd.DataFrame,                 # must include at least 'message' (used for topic mask) and your comment columns
    embs: np.ndarray,                   # (N, d) float32, L2-normalized rows
    encoder: SentenceTransformer,       # same model used to build embs
    index: Optional["hnswlib.Index"] = None,  # HNSW over embs (optional)
    require_all_keywords: bool = False, # ANY by default
    bf_threshold: int = 500_000,        # if topic slice ≤ this → do exact only
    ann_k0: int = 10_000,
    ann_step: int = 10_000,
    ann_max_k: int = 200_000,
    top_k: int = 50
) -> List[Dict]:
    """
    Returns up to `top_k` rows (dicts) with the **lowest cosine(q, x)** inside the topic slice.
    Correctness guaranteed by exact cosine re-scoring. (Larger top_k → better pool for NLI re-rank)
    """
    # 0) encode & normalize query
    with torch.inference_mode():
        q = encoder.encode(query_text, convert_to_tensor=False, normalize_embeddings=True).astype("float32")

    # 1) topic filter → indices
    topic_mask = make_topic_mask_english(meta, topic_query, require_all=require_all_keywords)
    topic_indices = np.flatnonzero(topic_mask)
    if topic_indices.size == 0:
        raise ValueError(f"No rows matched topic filter: {topic_query!r}")

    method = "exact"

    # 2) exact if no ANN or slice small
    if (index is None) or (topic_indices.size <= bf_threshold):
        top_idxs, top_sims = topk_min_cosine_over_indices(q, embs, topic_indices, k=top_k)

    else:
        # 3) ANN near -q → filter by topic → exact re-score
        q_neg = -q
        N = embs.shape[0]
        k_req = min(ann_k0, N)
        filtered: List[int] = []

        while True:
            labels, _ = index.knn_query(q_neg, k=k_req)
            ids = labels[0].astype(int)
            filtered = [i for i in ids if topic_mask[i]]
            if filtered or k_req >= min(ann_max_k, N):
                break
            k_req = min(k_req + ann_step, N)

        if filtered:
            filtered = np.array(filtered, dtype=int)
            sims = embs[filtered] @ q
            k_eff = min(top_k, sims.size)
            sel = np.argpartition(sims, k_eff - 1)[:k_eff]
            order = np.argsort(sims[sel])  # ascending → most opposite
            top_idxs = filtered[sel][order]
            top_sims = sims[sel][order]
            method = "ann+exact"
        else:
            top_idxs, top_sims = topk_min_cosine_over_indices(q, embs, topic_indices, k=top_k)

    # package results
    out: List[Dict] = []
    for gi, sim in zip(top_idxs.tolist(), top_sims.tolist()):
        row = meta.iloc[int(gi)].to_dict()
        row["row_index"] = int(gi)
        row["similarity"] = float(sim)   # cosine; lower = more opposite
        row["topic_match_count"] = int(topic_indices.size)
        row["method"] = method
        out.append(row)
    return out


# =============================== Multilingual NLI (XNLI) ===============================

from transformers import AutoTokenizer, AutoModelForSequenceClassification

@torch.inference_mode()
def load_nli(model_name: str = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"):
    """
    Multilingual NLI (supports Hebrew). Alternatives:
        "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
      - "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" (good speed/quality)
    """
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    return tok, mdl

@torch.inference_mode()
def nli_contradiction_probs_batch(tok, mdl, premises: List[str], hypotheses: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Batched P(contradiction) for pairs (premise, hypothesis).
    Returns np.array of shape [len(premises)], values in [0,1].
    """
    assert len(premises) == len(hypotheses)
    all_probs = []
    # find contradiction index robustly from model config
    id2label = getattr(mdl.config, "id2label", None)
    if id2label and isinstance(id2label, dict):
        labels = [id2label[i].lower() for i in range(len(id2label))]
        c_idx = labels.index("contradiction") if "contradiction" in labels else 2
    else:
        c_idx = 2

    for s in range(0, len(premises), batch_size):
        p = premises[s:s+batch_size]
        h = hypotheses[s:s+batch_size]
        batch = tok(p, h, return_tensors="pt", truncation=True, padding=True)
        logits = mdl(**batch).logits  # [B,3]
        probs = torch.softmax(logits, dim=-1)[:, c_idx].cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0) if all_probs else np.array([], dtype=np.float32)


def rerank_with_nli_english(
    user_text: str,
    candidates: List[Dict],
    *,
    tok=None,
    mdl=None,
    cosine_weight: float = 0.35,
    nli_weight: float = 0.65,
    min_chars: int = 20,
) -> List[Dict]:
    """
    Re-rank candidates by a weighted sum:
      combined = w_cosine * normalized_inverse_cosine + w_nli * P(contradiction)
    - Filters out very short candidates (< min_chars) before NLI.
    Returns sorted list (descending combined_score).
    """
    if tok is None or mdl is None:
        tok, mdl = load_nli("MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli")

    # Filter/prepare
    kept = []
    for c in candidates:
        txt = str(c.get("message") or c.get("comment_text") or "")
        if len(txt) >= min_chars:
            kept.append(c)
    if not kept:
        kept = candidates[:]  # fallback if everything was too short

    sims = np.array([float(c["similarity"]) for c in kept], dtype=np.float32)
    s_min, s_max = float(sims.min()), float(sims.max())
    # invert cosine (lower is better → closer to 1)
    if s_max > s_min:
        cos_norm = (s_max - sims) / (s_max - s_min)
    else:
        cos_norm = np.ones_like(sims)

    premises  = [user_text] * len(kept)
    hypotheses = [str(c.get("message") or c.get("comment_text") or "") for c in kept]
    p_contra = nli_contradiction_probs_batch(tok, mdl, premises, hypotheses)

    out = []
    for i, c in enumerate(kept):
        combined = cosine_weight * float(cos_norm[i]) + nli_weight * float(p_contra[i])
        c2 = dict(c)
        c2["nli_contradiction"] = float(p_contra[i])
        c2["combined_score"] = float(combined)
        out.append(c2)

    out.sort(key=lambda x: x["combined_score"], reverse=True)
    return out


# =============================== High-level: cosine + NLI ===============================

def most_opposite_in_topic_english_with_nli(
    query_text: str,
    topic_query: str | Iterable[str],
    *,
    meta: pd.DataFrame,
    embs: np.ndarray,
    encoder: SentenceTransformer,
    index: Optional["hnswlib.Index"] = None,
    require_all_keywords: bool = False,
    bf_threshold: int = 500_000,
    ann_k0: int = 10_000,
    ann_step: int = 10_000,
    ann_max_k: int = 200_000,
    top_k_candidates: int = 200,   # get a good pool for NLI
    top_k_final: int = 3,          # final results to return
    nli_model_name: str = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
    cosine_weight: float = 0.35,
    nli_weight: float = 0.65,
) -> List[Dict]:
    """
    1) Topic mask → "most opposite" by cosine to get a candidate pool (ANN+exact).
    2) Re-rank the pool by multilingual NLI contradiction + cosine signal.
    3) Return top_k_final.
    """
    # Step 1: cosine-based candidate pool
    pool = most_opposite_in_topic_english(
        query_text,
        topic_query,
        meta=meta, embs=embs, encoder=encoder, index=index,
        require_all_keywords=require_all_keywords,
        bf_threshold=bf_threshold, ann_k0=ann_k0, ann_step=ann_step, ann_max_k=ann_max_k,
        top_k=top_k_candidates,
    )

    # Step 2: NLI re-rank
    tok, mdl = load_nli(nli_model_name)
    ranked = rerank_with_nli_english(
        user_text=query_text,
        candidates=pool,
        tok=tok, mdl=mdl,
        cosine_weight=cosine_weight,
        nli_weight=nli_weight,
    )

    return ranked[:top_k_final]


# =============================== Artifacts loader (English) ===============================

# def load_english(art_dir: Path | str = "./english"):
#     """
#     Loads English artifacts: config.json, meta.parquet, embeddings.npy, optional hnsw_cosine.bin
#     meta must include a 'comment_text' column (the comment text).
#     """
#     ART_DIR = Path(art_dir)
#     CFG_PATH = ART_DIR / "config.json"
#     META_PATH = ART_DIR / "meta.parquet"
#     EMB_PATH  = ART_DIR / "embeddings.npy"
#     HNSW_PATH = ART_DIR / "hnsw_cosine.bin"  # optional

#     if not CFG_PATH.exists() or not META_PATH.exists() or not EMB_PATH.exists():
#         raise FileNotFoundError(
#             f"Missing artifacts in {ART_DIR}. Need: config.json, meta.parquet, embeddings.npy (and optionally hnsw_cosine.bin)."
#         )

#     cfg  = json.load(open(CFG_PATH, "r", encoding="utf-8"))
#     meta = pd.read_parquet(META_PATH)
#     embs = np.load(EMB_PATH, mmap_mode="r")  # float32, L2-normalized

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     encoder = SentenceTransformer(cfg["model"], device=device)

#     index: Optional["hnswlib.Index"] = None
#     if (hnswlib is not None) and HNSW_PATH.exists():
#         index = hnswlib.Index(space="cosine", dim=int(cfg["dim"]))
#         index.load_index(str(HNSW_PATH), max_elements=int(cfg["n"]))
#         index.set_ef(int(cfg.get("efQuery", 200)))

#     return meta, embs, index, encoder

def _ensure_artifacts_from_hf(
    local_dir: str | Path,
    repo_id: Optional[str] = None,
    repo_type: str = "dataset",
    token_env: str = "HF_TOKEN",
) -> None:
    """
    If repo_id is provided and required files are missing locally,
    download a snapshot from Hugging Face Hub into local_dir.
    """
    if not repo_id:
        return  # HF not requested

    target = Path(local_dir)
    need = [
        target / "config.json",
        target / "meta.parquet",
        target / "embeddings.npy",
        # optional:
        target / "hnsw_cosine.bin",
    ]
    if all(p.exists() for p in need[:-1]):  # all required exist; skip download
        return

    token = os.getenv(token_env, None)  # required if private repo
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=token,
    )

def load_english(
    art_dir: str | Path = "./english",
    *,
    # pass these ONLY if you want automatic HF download when local files are missing
    repo_id: Optional[str] = None,       # e.g., "yourname/hebrew-opposite-artifacts"
    repo_type: str = "dataset",
    hf_token_env: str = "HF_TOKEN",
    # override model used for encoding (defaults to value in config.json)
    override_model: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray, Optional["hnswlib.Index"], SentenceTransformer]:
    """
    Load English artifacts: config.json, meta.parquet, embeddings.npy, optional hnsw_cosine.bin.
    - If repo_id is provided and files are missing, pulls them from Hugging Face Hub first.
    - Validates that meta contains a 'message' column (comment text).
    Returns: (meta, embs, index, encoder)
    """
    ART_DIR = Path(art_dir)

    # Try to bring files from HF if missing
    _ensure_artifacts_from_hf(ART_DIR, repo_id=repo_id, repo_type=repo_type, token_env=hf_token_env)

    #CFG_PATH = ART_DIR / "config.json"
    META_PATH = ART_DIR / "meta.parquet"
    #EMB_PATH  = ART_DIR / "embeddings.npy"
    #HNSW_PATH = ART_DIR / "hnsw_cosine.bin"  # optional

    # missing = [p.name for p in (CFG_PATH, META_PATH, EMB_PATH) if not p.exists()]
    # if missing:
    #     raise FileNotFoundError(
    #         f"Missing artifacts in {ART_DIR} → {', '.join(missing)}. "
    #         f"If you host them on Hugging Face, pass repo_id='user/repo' (and set HF_TOKEN if private)."
    #     )

    # Load config/meta/embeddings
    # cfg  = json.load(open(CFG_PATH, "r", encoding="utf-8"))
    meta = pd.read_parquet(META_PATH)
    # embs = np.load(EMB_PATH, mmap_mode="r")  # float32, L2-normalized

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # encoder = SentenceTransformer(cfg["model"], device=device)

    # index: Optional["hnswlib.Index"] = None
    # if (hnswlib is not None) and HNSW_PATH.exists():
    #     index = hnswlib.Index(space="cosine", dim=int(cfg["dim"]))
    #     index.load_index(str(HNSW_PATH), max_elements=int(cfg["n"]))
    #     index.set_ef(int(cfg.get("efQuery", 200)))

    return meta#, embs, index, encoder


def other_comments_same_author_same_topic(
    meta: pd.DataFrame,
    topic_query: str | Iterable[str],
    *,
    base_row: dict,                 # from your opposite finder; must contain 'commenter_id' (and optionally 'row_index')
    require_all_keywords: bool = False,
    text_col: str = "message",
    id_col: str = "commenter_id",
    limit: Optional[int] = None,
) -> List[str]:
    """
    Return ONLY the message strings of other comments by the same commenter_id
    that also match the same topic keywords.
    - Skips if commenter_id == '-1' or empty.
    - Excludes the base row (by row_index if present; else by exact text match).
    - If 'date' exists in meta, sorts by date desc (newest first).
    """
    if id_col not in meta.columns:
        raise ValueError(f"meta missing '{id_col}'")
    if text_col not in meta.columns:
        raise ValueError(f"meta missing '{text_col}'")
    
    cid = str(base_row.get(id_col, "")).strip()
    print(cid)
    if cid in ("", "-1"):
        return []
    
    author_mask = meta[id_col].astype(str).str.strip().eq(cid).values
    topic_mask  = make_topic_mask_english(meta, topic_query, require_all=require_all_keywords)

    mask = author_mask & topic_mask

    # exclude the base row
    base_idx = base_row.get("row_index", None)
    if isinstance(base_idx, (int, np.integer)) and 0 <= int(base_idx) < len(meta):
        mask[int(base_idx)] = False
    else:
        base_text = str(base_row.get(text_col, ""))
        if base_text:
            mask &= ~(meta[text_col].astype(str) == base_text).values

    idxs = np.flatnonzero(mask)
    if idxs.size == 0:
        return []

    if limit is not None:
        idxs = idxs[:int(limit)]

    # return only message strings
    return meta.iloc[idxs][text_col].astype(str).tolist()



if __name__ == "__main__":

    meta, embs, index, encoder = load_english("./english")

    #user_text = "ביבי אינטרסנט שדואג אך ורק לעצמו"
    # user_text = "ביבי הרס את המדינה"
    # user_text = "ביבי האפס"
    #user_text="ביבי צריך להספיק להיות ראש ממשלה"
    user_text="ביבי הרס לנו את המדינה"
    topic_keywords = ["בינימין נתניהו", "ביבי"]

    results = most_opposite_in_topic_english_with_nli(
        query_text=user_text,
        topic_query=topic_keywords,
        meta=meta, embs=embs, encoder=encoder, index=index,
        require_all_keywords=False,
        top_k_candidates=200,   # pool size for NLI
        top_k_final=3,          # return 3 best
        cosine_weight = 0.2,
        nli_weight= 0.8,
    )

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] cos={r['similarity']:.4f}  nli={r['nli_contradiction']:.3f}  combined={r['combined_score']:.3f}")
        print(f"message: {str(r.get('message',''))}")
        print(f"commenter: {str(r.get('commenter_id', ''))}")

        other_comments = ""
        others_msgs = other_comments_same_author_same_topic(
        meta,
        topic_query=topic_keywords,
        base_row=r,            # from most_opposite_in_topic_hebrew(_with_nli)
        require_all_keywords=False
        )
        for m in others_msgs:
            print("-", m)
