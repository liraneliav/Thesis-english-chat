# app_opposite_streamlit.py
# -----------------------------------------------------------
# Streamlit app: "Most Opposite" comment finder (Hebrew-ready)
# Cascade: Topic mask → ANN/cosine pool → NLI re-rank → (optional) GPT re-rank
# Uses Azure OpenAI for the GPT stage.
# -----------------------------------------------------------

from __future__ import annotations
import os, json, re, time
from typing import Iterable, List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Torch / NLI
import torch
#from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# Azure OpenAI (official openai package >= 1.0)
from openai import AzureOpenAI
from dotenv import load_dotenv

#from opposite_english_nli import load_english, other_comments_same_author_same_topic  


# ===============================
# Azure OpenAI client
# ===============================
load_dotenv()
endpoint = os.getenv("ENDPOINT_URL", "https://YOUR-ENDPOINT.openai.azure.com/")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2025-01-01-preview",
)

# ===============================
# GPT re-ranker (Azure OpenAI)
# ===============================
SYSTEM_RERANK = """\
You are a careful evaluator. Score how strongly each candidate comment CONTRADICTS the user's opinion.
Return strict JSON with an array 'scores', where each item is:
{"id": <string>, "contradiction": <float 0..1>, "rationale": <short string>}

Rules:
- 1.0 means strong, direct contradiction; 0.0 means agreement or same stance; ~0.5 is neutral/irrelevant.
- Consider stance, claims, and implications — not just wording overlap.
- Be concise and deterministic.
"""

def gpt_rerank_contradiction(
    user_opinion: str,
    candidates: List[Dict],                 # each: {'id', 'text', 'topic'(opt), 'cosine'(opt)}
    model: str = "gpt-5-mini",
    batch_size: int = 40
) -> List[Dict]:
    """
    Returns the same list with 'gpt_contra' and 'gpt_rationale' added, sorted by gpt_contra desc.
    """
    out_scores: Dict[str, tuple[float, str]] = {}

    for s in range(0, len(candidates), batch_size):
        batch = candidates[s:s+batch_size]
        compact = [{"id": c["id"], "text": c["text"], "topic": c.get("topic","")} for c in batch]
        msg_user = (
            "User opinion (premise):\n"
            + user_opinion.strip()
            + "\n\nCandidates (hypotheses):\n"
            + json.dumps(compact, ensure_ascii=False)
            + "\n\nReturn JSON: {\"scores\": [{\"id\": \"...\", \"contradiction\": 0.0-1.0, \"rationale\": \"...\"}, ...]}"
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_RERANK},
                {"role": "user", "content": msg_user},
            ],
        )
        txt = resp.choices[0].message.content
        try:
            data = json.loads(txt)
            for item in data.get("scores", []):
                cid = str(item.get("id",""))
                score = float(item.get("contradiction", 0.0))
                rat = str(item.get("rationale",""))
                out_scores[cid] = (score, rat)
        except Exception:
            # fallback: neutral if parsing fails
            for c in batch:
                out_scores[c["id"]] = (0.5, "Parse error; defaulted to neutral")

    enriched = []
    for c in candidates:
        sc, ra = out_scores.get(c["id"], (0.5, "missing"))
        c2 = dict(c)
        c2["gpt_contra"] = float(sc)
        c2["gpt_rationale"] = ra
        enriched.append(c2)

    enriched.sort(key=lambda x: x["gpt_contra"], reverse=True)
    return enriched


# ===============================
# NLI (multilingual) – fast proxy
# ===============================
# @st.cache_resource(show_spinner=False)
# def load_nli(model_name: str = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     mdl = SentenceTransformer(model_name, device=device)
#     return mdl

# @torch.inference_mode()
# def nli_contradiction_proxy(
#     premise: str,
#     hypotheses: List[str],
#     nli_model: SentenceTransformer,
#     batch_size: int = 64  #48# Increased from 48 for faster processing
# ) -> np.ndarray:
#     """
#     Fast proxy using a SBERT-style NLI checkpoint:
#     encode [premise, hypothesis] pairs and map to a 0..1 'contradiction-ish' score.
#     (If you have a proper XNLI cross-encoder with logits, swap this for true P(contradiction).)
#     """
#     pairs = [[premise, h] for h in hypotheses]
#     embs = nli_model.encode(
#         pairs, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=False
#     )
#     if embs.dim() == 2:
#         # crude mapping to [0..1]
#         scores = torch.tanh(-embs.norm(dim=1))
#         probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
#     else:
#         probs = torch.full((len(hypotheses),), 0.5)

#     return probs.detach().cpu().numpy().astype("float32")

# def rerank_with_nli_only(
#     user_text: str,
#     pool_rows: List[Dict],
#     *,
#     nli_model: SentenceTransformer
# ) -> List[Dict]:
#     if not pool_rows:
#         return []

#     hyps = [str(r.get("comment_text", "")) for r in pool_rows]
#     nli_p = nli_contradiction_proxy(user_text, hyps, nli_model=nli_model, batch_size=48)

#     order = np.argsort(-nli_p)  # descending

#     ranked = []
#     for i in order.tolist():
#         r = dict(pool_rows[i])
#         r["nli_contradiction"] = float(nli_p[i])
#         r["combined_score"] = float(nli_p[i])  # final blend happens after GPT
#         ranked.append(r)
#     return ranked

@st.cache_resource(show_spinner=False)
def load_nli(model_name: str = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    return tok, mdl

@torch.inference_mode()
def nli_contradiction_probs_batch(
    tok,
    mdl,
    premises: List[str],
    hypotheses: List[str],
    batch_size: int = 16,
) -> np.ndarray:
    """
    Batched P(contradiction) for (premise, hypothesis) pairs.
    Returns np.array of shape [len(premises)], values in [0,1].
    """
    assert len(premises) == len(hypotheses)

    all_probs = []

    id2label = getattr(mdl.config, "id2label", None)
    if id2label and isinstance(id2label, dict):
        labels = [id2label[i].lower() for i in range(len(id2label))]
        c_idx = labels.index("contradiction") if "contradiction" in labels else 2
    else:
        c_idx = 2

    for s in range(0, len(premises), batch_size):
        p = premises[s:s + batch_size]
        h = hypotheses[s:s + batch_size]

        batch = tok(
            p,
            h,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        logits = mdl(**batch).logits
        probs_tensor = torch.softmax(logits, dim=-1)[:, c_idx].detach().cpu()
        probs = np.asarray(probs_tensor.tolist(), dtype=np.float32)
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0) if all_probs else np.array([], dtype=np.float32)

def rerank_with_nli_only(
    user_text: str,
    pool_rows: List[Dict],
    *,
    nli_model,
) -> List[Dict]:
    if not pool_rows:
        return []

    tok, mdl = nli_model

    hypotheses = [str(r.get("comment_text", "")) for r in pool_rows]
    premises = [user_text] * len(pool_rows)

    nli_p = nli_contradiction_probs_batch(
        tok,
        mdl,
        premises,
        hypotheses,
        batch_size=16,
    )

    order = np.argsort(-nli_p)  # descending

    ranked = []
    for i in order.tolist():
        r = dict(pool_rows[i])
        r["nli_contradiction"] = float(nli_p[i])
        r["combined_score"] = float(nli_p[i])  # final blend happens after GPT
        ranked.append(r)

    return ranked


def build_pool_topic_only(
    *,
    meta: pd.DataFrame,
    topic_kws: List[str],
    K: int,
    text_col: str = "comment_text",
    require_all: bool = False,
    random_seed: int = 42,
) -> List[Dict]:
    """Candidate pool by keyword-topic filter ONLY (no embeddings, no ANN). Returns up to K rows."""
    titles = meta[text_col].astype(str)
    kws = [k for k in (topic_kws or []) if k]

    if not kws:
        idxs = np.arange(len(meta))
    else:
        if require_all:
            mask = np.ones(len(meta), dtype=bool)
            for kw in kws:
                mask &= titles.str.contains(re.escape(kw), case=False, na=False).values
        else:
            mask = np.zeros(len(meta), dtype=bool)
            for kw in kws:
                mask |= titles.str.contains(re.escape(kw), case=False, na=False).values
        idxs = np.flatnonzero(mask)

    if idxs.size == 0:
        return []

    rng = np.random.default_rng(random_seed)
    if idxs.size > K:
        idxs = rng.choice(idxs, size=K, replace=False)

    pool: List[Dict] = []
    for i in idxs.tolist():
        row = meta.iloc[i].to_dict()
        row["row_index"] = i
        pool.append(row)
    return pool


# def _load_artifacts(path: str):
#     meta, embs, index, encoder = load_english(path)  # expects config-trained encoder, l2-normalized embs
#     return meta, embs, index, encoder

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

def other_comments_same_author_same_topic(
    meta: pd.DataFrame,
    topic_query: str | Iterable[str],
    *,
    base_row: dict,                 # from your opposite finder; must contain 'commenter_id' (and optionally 'row_index')
    require_all_keywords: bool = False,
    text_col: str = "comment_text",
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



def run_opposite_pipeline_and_render(
    *,
    user_opinion: str,
    topic_keywords: str | list[str],
    meta,
    #embs, index, encoder,
    # retrieval / ranking knobs
    pool_size: int = 100,  # Reduced from 200 for faster processing
    k2_short: int = 5,    # Reduced from 15 for faster GPT calls
    nli_model_name: str = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
    use_gpt: bool = True,
    gpt_model: str = "gpt-5-mini",
    # blend weights
    beta: float  = 0.3,            # NLI weight
    gamma: float = 0.5,            # GPT weight
    # display
    top_k_show: int = 3,
    # dependencies (pass your functions/clients)
    load_nli_fn    = load_nli,         # e.g., load_nli
    nli_rerank_fn  = rerank_with_nli_only,         # e.g., rerank_with_nli
    gpt_rerank_fn  = gpt_rerank_contradiction,         # e.g., gpt_rerank_contradiction
    gpt_client     = client,         # AzureOpenAI or OpenAI client
    include_author_threads: bool = False,#True,              # NEW
    other_comments_fn = other_comments_same_author_same_topic,  # NEW
    author_limit: int = 15,                            # NEW
    author_text_col: str = "comment_text",                 # NEW
    author_id_col: str = "author",              # NEW
    require_all_keywords_for_author: bool = False,    # NEW
    on_progress=None,
):
    """
    End-to-end: validates input → NLI re-rank → optional GPT re-rank →
    final blend → RENDERS results in Streamlit → returns (enriched_top, timings_dict).

    Return shape (NEW):
      enriched_top = [
        { "row": <top item dict>, "other_by_author": [str, ...] },
        ...
      ]
    """
    if meta is None or not isinstance(meta, pd.DataFrame) or meta.empty:
        print("meta not loaded or empty")

    if "comment_text" not in meta.columns:
        print(f"meta missing 'comment_text' column, meta colums {list(meta.columns)}")
    # --- 0) validation ---
    if not user_opinion or not user_opinion.strip():
        #st.warning("Please type your opinion first.")
        return [], {"error": "empty_opinion"}

    # normalize topic keywords to list[str]
    if isinstance(topic_keywords, str):
        kws = [k.strip() for k in topic_keywords.split(",") if k.strip()]
    else:
        kws = [str(k).strip() for k in topic_keywords if str(k).strip()]

    timings: Dict[str, int] = {}
    t0 = time.time()

    def _prog(p: int, m: str):
        if on_progress is not None:
            try:
                on_progress(int(p), str(m))
            except Exception:
                pass

    _prog(5, "Starting...")

    # --- 1) Topic pool (NO cosine) ---
    _prog(20, "Elevate the conversation environment")
    pool_rows = build_pool_topic_only(meta=meta, topic_kws=kws, K=pool_size, text_col="comment_text")
    t1 = time.time()
    timings["pool_ms"] = int((t1 - t0) * 1000)

    if not pool_rows:
        #st.info("No candidates found. Try broader topic keywords.")
        return [], timings

    # --- 2) NLI re-rank ---
    _prog(50, "Collecting relevant information...")
    nli_model = load_nli_fn(nli_model_name)
    nli_ranked = nli_rerank_fn(
        user_text=user_opinion,
        pool_rows=pool_rows,
        nli_model=nli_model,
    )
    t2 = time.time()
    timings["nli_ms"] = int((t2 - t1) * 1000)

    # shortlist for GPT
    shortlist = nli_ranked[:min(k2_short, len(nli_ranked))]

    # --- 3) GPT final judge (optional) ---
    if use_gpt:
        _prog(75, "Updating the system")
        # pack compact candidates
        candidates = []
        for r in shortlist:
            rid = str(r.get("row_index", len(candidates)))
            txt = str(r.get("comment_text", ""))
            candidates.append({"id": rid, "text": txt, "topic": "", "row": r})

        gpt_ranked = gpt_rerank_fn(user_opinion, candidates, model=gpt_model, batch_size=8)

        # Final blend: β * NLI + γ * GPT
        out = []
        for item in gpt_ranked:
            r = dict(item["row"])
            nli_p   = r.get("nli_contradiction", 0.0)
            gpt_p   = float(item["gpt_contra"])
            final   = beta * nli_p + gamma * gpt_p
            r["gpt_contradiction"] = gpt_p
            r["gpt_rationale"] = item.get("gpt_rationale", "")
            r["combined_score"] = float(final)
            out.append(r)

        out.sort(key=lambda z: z["combined_score"], reverse=True)
        ranked = out
        t3 = time.time()
        timings["gpt_ms"] = int((t3 - t2) * 1000)
        timings["total_ms"] = int((t3 - t0) * 1000)
    else:
        _prog(75, "Just a moment, almost there...")
        out = []
        for r in shortlist:
            inv_cos = r.get("inv_cosine_norm", 0.0)
            nli_p   = r.get("nli_contradiction", 0.0)
            final   = beta * nli_p
            rr = dict(r)
            rr["combined_score"] = float(final)
            out.append(rr)
        out.sort(key=lambda z: z["combined_score"], reverse=True)
        ranked = out
        t3 = time.time()
        timings["gpt_ms"] = 0
        timings["total_ms"] = int((t3 - t0) * 1000)

    # ===== NEW: enrich the top-k with same-author/same-topic history =====
    # --- 4) Enrich top-k with same-author/same-topic history ---
    _prog(90, "Just a moment, almost there...")
    enriched_top: List[Dict] = []
    for r in ranked[:top_k_show]:
        others: List[str] = []
        if include_author_threads and other_comments_fn is not None:
            try:
                others = other_comments_fn(
                    meta,
                    topic_query=kws,
                    base_row=r,
                    require_all_keywords=require_all_keywords_for_author,
                    text_col=author_text_col,
                    id_col=author_id_col,
                    limit=author_limit,
                )
            except Exception:
                others = []
        enriched_top.append({"row": r, "other_by_author": others})

    _prog(100, "Done ✅")
    return enriched_top, timings




if __name__ == "__main__":  
    #meta, embs, index, encoder = _load_artifacts("./english")
    ART_DIR = Path("./english")
    META_PATH = ART_DIR / "meta.parquet"

    if not META_PATH.exists():
        raise FileNotFoundError(
            f"Missing artifacts in {ART_DIR}. Need: config.json, meta.parquet, embeddings.npy (and optionally hnsw_cosine.bin)."
        )

    meta = pd.read_parquet(META_PATH)
    print("hi")
    # ranked, timings = run_opposite_pipeline_and_render(user_opinion="ביבי הרס לנו את המדינה", topic_keywords=["בנימין נתניהו","ביבי"], meta=meta, embs=embs, index=index, encoder=encoder)
    # print(ranked)
    # print(timings)

    ranked, timings = run_opposite_pipeline_and_render(user_opinion="just making a huge mess", topic_keywords=["black lives matter"], meta=meta, pool_size=100,)
    all_comments = []
    for i, item in enumerate(ranked, 1):
        row = item["row"]
        print(f"\n### TOP {i} ###")
        print(row.get("comment_text", ""))  # main opposite comment
        all_comments.append(row.get("comment_text", ""))
        for j, t in enumerate(item.get("other_by_author", []), 1):
            print(f"[other {j}] {t}")
            all_comments.append(t)

    print(all_comments)
    print("Timings (ms):", timings)
