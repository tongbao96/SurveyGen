# -*- coding: utf-8 -*-
import re
import json
import time
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI

TOPIC_CSV   = "./example/topic.csv"
PAPERS_JSON = "final_ref_set.json"
OUTPUT_FILE = "paper_after_reranking.csv"

# Metric weights
W_CITED, W_AUTH, W_JOUR, W_REL, W_AVGCS = 0.165, 0.067, 0.10, 0.33, 0.33

# LLM call
LLM_MODEL = "gpt-4o"
MAX_RETRIES_PER_PAPER = 50
BASE_SLEEP = 1.5

client = OpenAI(
    base_url="your api address",
    api_key="your api key"
)

# ===================== 1) Read Data =====================
topic_df = pd.read_csv(TOPIC_CSV)
topic_title = str(topic_df.loc[0, "title"]).strip()
topic_keywords = str(topic_df.loc[0, "keywords"]) if "keywords" in topic_df.columns else ""
raw_max_refs = topic_df.loc[0, "max_refs"] if "max_refs" in topic_df.columns else 50
topic_max_refs = int(str(raw_max_refs).strip()) if not pd.isna(raw_max_refs) else 50
topic_text = f"{topic_title}. {topic_keywords.replace(';', ', ')}".strip()

with open(PAPERS_JSON, "r", encoding="utf-8") as f:
    papers = json.load(f)

df = pd.DataFrame(papers)

# Add missing columns if they don't exist (avoid KeyError; skipped papers will remain empty)
for col in ["cited_by_count", "first_author_hindex", "last_author_hindex",
            "journal_hindex", "abstract", "title", "doi"]:
    if col not in df.columns:
        df[col] = np.nan

# Perform all subsequent operations only on papers with abstracts
has_abs = df["abstract"].apply(lambda x: isinstance(x, str) and x.strip() != "")
work = df[has_abs].copy()
total_work = len(work)

def percentile_bin_q1_best(series, n_bins=4, labels=("Q1","Q2","Q3","Q4")):
    """Q1 = highest, Q4 = lowest; only compute for the given subset (make sure to filter for papers with abstracts first)"""
    ranks = series.rank(method="average", pct=True, ascending=False)
    return pd.cut(ranks, bins=np.linspace(0,1,n_bins+1), labels=labels,
                  include_lowest=True, duplicates="drop")

def author_hindex_avg_row(row):
    vals = [v for v in [row.get("first_author_hindex"), row.get("last_author_hindex")] if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan

def map_bin_q1_best(s):
    return s.map({"Q1":4,"Q2":3,"Q3":2,"Q4":1})

def minmax01(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if s.dropna().nunique() <= 1:
        return pd.Series([0.5]*len(s), index=s.index)
    return (s - s.min()) / (s.max() - s.min())

def to_none(df_in: pd.DataFrame) -> pd.DataFrame:
    return df_in.where(pd.notna(df_in), None)

def safe_text(x):
    return x if isinstance(x, str) else ""

work["author_hindex_avg"] = work.apply(author_hindex_avg_row, axis=1)
work["cited_by_bin"]  = percentile_bin_q1_best(work["cited_by_count"])
work["author_h_bin"]  = percentile_bin_q1_best(work["author_hindex_avg"])
work["journal_h_bin"] = percentile_bin_q1_best(work["journal_hindex"])

# Write stratification results back to the main table, keep others (without abstracts) as NaN
for col in ["author_hindex_avg","cited_by_bin","author_h_bin","journal_h_bin"]:
    df.loc[work.index, col] = work[col]

# ===================== 4) Average Cosine Similarity (subset only; BGE-Large) =====================
print("loading BGE-Large ...")
model = SentenceTransformer("BAAI/bge-large-en")

doc_texts = [f"{safe_text(r.get('title',''))}. {safe_text(r.get('abstract',''))}".strip() for _, r in work.iterrows()]
if len(doc_texts) >= 1:
    doc_embs = model.encode(doc_texts, convert_to_tensor=True, normalize_embeddings=True).cpu().numpy()
    if len(doc_embs) > 1:
        pairwise = cosine_similarity(doc_embs)
        avg_sims = [float(pairwise[i, np.arange(len(work)) != i].mean()) for i in range(len(work))]
    else:
        avg_sims = [np.nan]
    df.loc[work.index, "avg_cos_sim_to_others"] = avg_sims

def parse_score(text: str):
    if not isinstance(text, str):
        return None
    m = re.search(r"[1-5]", text)
    return int(m.group(0)) if m else None

def relevance_by_gpt_until(topic, title, abstract, max_retries=MAX_RETRIES_PER_PAPER, base_sleep=BASE_SLEEP, log_prefix=""):
    prompt = f"""
You are an academic expert helping to write a survey on the topic: "{topic}".
You will be provided with the title and abstract of a research paper. Your task is to rating the paper's relevance to the survey topic based on the following criteria:

Score-1 (poor): The paper is unrelated to the topic or only mentions it in passing, with no meaningful contribution.
Score-2 (low): The paper is loosely connected or provides general background, but not focused on the topic.
Score-3 (moderate): The paper discusses a specific sub-aspect of the topic; somewhat useful, but not central.
Score-4 (high): The paper substantially addresses key elements of the topic; would likely be cited in the survey.
Score-5 (very high): The paper is entirely focused on the topic, offering essential insights; likely foundational to the survey.

Your output should only include a single score from 1 to 5. Do not provide any explanation or additional text.

Title: {title}
Abstract: {abstract}

Topical relevance score:
""".strip()

    attempt, sleep_s = 0, base_sleep
    while True:
        attempt += 1
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            raw = (resp.choices[0].message.content or "").strip()
            score = parse_score(raw)
            if score is not None:
                print(f"{log_prefix} Response on attempt {attempt} -> {raw!r}, parsed={score}")
                return score, attempt
            else:
                print(f"{log_prefix} Response parsing failed on attempt {attempt} -> {raw!r}")
        except Exception as e:
            print(f"{log_prefix} Exception on attempt {attempt}: {e}")

        if (MAX_RETRIES_PER_PAPER is not None) and (attempt >= MAX_RETRIES_PER_PAPER):
            print(f"{log_prefix} ❌ Exceeded maximum retries ({MAX_RETRIES_PER_PAPER}), returning None")
            return None, attempt

        time.sleep(sleep_s)
        sleep_s = min(sleep_s * 1.7, 30)

if total_work > 0:
    print(f"Starting GPT relevance scoring (only for {total_work} papers with abstracts)...")
    scores = []
    t0_all = time.time()
    for pos, (idx, row) in enumerate(work.iterrows(), start=1):
        t0 = time.time()
        s, att = relevance_by_gpt_until(topic_title, row.get("title",""), row.get("abstract",""),
                                        log_prefix=f"[{pos}/{total_work}]")
        scores.append((idx, s))
        t1 = time.time()
        avg_t = (t1 - t0_all) / pos
        remain = avg_t * (total_work - pos)
        print(
            f"[{pos}/{total_work}] Score={s} | Attempts={att} | This round:{t1 - t0:.1f}s | Avg:{avg_t:.1f}s/paper | Est. remaining:{remain / 60:.1f}min")

        # Write back to main table
        for idx, s in scores:
            df.at[idx, "relevance_llm_1to5"] = float(s) if s is not None else np.nan

        # Stratify (subset only)

df.loc[work.index, "relevance_bin"] = percentile_bin_q1_best(df.loc[work.index, "relevance_llm_1to5"])

# ===================== 6) Composite Score (subset only, dynamic weights) =====================
if total_work > 0:
    # First, construct each component for the subset (all oriented as "higher = better")
    cited_score = minmax01(map_bin_q1_best(df.loc[work.index, "cited_by_bin"]))
    auth_score  = minmax01(map_bin_q1_best(df.loc[work.index, "author_h_bin"]))
    jour_score  = minmax01(map_bin_q1_best(df.loc[work.index, "journal_h_bin"]))
    rel_score   = minmax01(df.loc[work.index, "relevance_llm_1to5"].astype(float))
    uniq_score  = 1 - minmax01(df.loc[work.index, "avg_cos_sim_to_others"].astype(float))  # Lower is better -> higher score

    parts = pd.DataFrame({
        "cited_score": cited_score,
        "auth_score":  auth_score,
        "jour_score":  jour_score,
        "rel_score":   rel_score,
        "uniq_score":  uniq_score
    }, index=work.index)

    W = np.array([W_CITED, W_AUTH, W_JOUR, W_REL, W_AVGCS], dtype=float)

    def row_weighted_score(row_vals):
        vals = row_vals.values.astype(float)
        mask = ~np.isnan(vals)
        if not mask.any():
            return np.nan
        wsum = W[mask].sum()
        if wsum <= 0:
            return np.nan
        return float((vals[mask] * W[mask]).sum() / wsum)

    df.loc[work.index, "composite_score"] = parts.apply(row_weighted_score, axis=1)

# ===================== 7) Label and Export =====================
# Select top max_refs only from the subset with abstracts and composite scores

df["tag"] = None
rank_pool = df.loc[work.index].dropna(subset=["composite_score"]).sort_values("composite_score", ascending=False)
top_k_idx = rank_pool.head(topic_max_refs).index
df.loc[top_k_idx, "tag"] = "selected"

cols = [
    "title","doi", "abstract",
    "cited_by_count","cited_by_bin",
    "first_author_hindex","last_author_hindex","author_hindex_avg","author_h_bin",
    "journal_hindex","journal_h_bin",
    "relevance_llm_1to5","relevance_bin",
    "avg_cos_sim_to_others",
    "composite_score",
    "tag"
]

# Final export sorted by composite score (null values at the end)
df_final = df.sort_values("composite_score", ascending=False)[cols]
to_none(df_final).to_csv(OUTPUT_FILE, index=False)
print(f"✅ Exported: {OUTPUT_FILE} (only papers with abstracts were calculated and selected)")
