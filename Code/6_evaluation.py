#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import json
import math
import difflib
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Any

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

# --------- Configuration: File Paths ----------
GOLD_JSON_PATH          = "./example/Gold_Standard_Deep Learning Applications in Computer Vision.json"
RANKED_PAPERS_CSV       = "paper_after_reranking.csv"
SURVEY_OUTLINE_JSON     = "survey_outline.json"
SURVEY_CONTENT_JSON     = "survey_content.json"
SURVEY_SECTIONS_JSON    = "survey_sections.json"

# --------- Configuration: Thresholds / Models ----------
TITLE_TEXTUAL_THR       = 0.95
SECTION_SEM_THR         = 0.80
EMBED_MODEL_NAME        = "BAAI/bge-large-en"


USE_LLM_STRUCTURE_SCORE = True
LLM_BASE_URL            = "your api address"
LLM_API_KEY             = "your api key"
LLM_MODEL_NAME          = "gpt-4o"
LLM_MAX_RETRIES         = 8
LLM_TEMPERATURE         = 0

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  #

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,:;!?\"'()[]{}")
    return s

def textual_similarity(a: str, b: str) -> float:
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    return difflib.SequenceMatcher(None, a_n, b_n).ratio()

def clean_json_string(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def load_gold_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_gold_reference_titles(gold: Dict[str, Any]) -> List[str]:
    """Extract all citations[].title from the gold standard (remove empty / deduplicate)"""
    titles = []
    for sec in gold.get("sections", []):
        for para in sec.get("paragraphs", []):
            for cit in para.get("citations", []):
                t = cit.get("title")
                if t and isinstance(t, str) and t.strip():
                    titles.append(t.strip())
    seen = set()
    uniq = []
    for t in titles:
        key = normalize_text(t)
        if key and key not in seen:
            seen.add(key)
            uniq.append(t)
    return uniq

# remove section like（funding / acknowledgements / author contributions / competing interests / supplementary material 等）
NON_CONTENT_SEC_PATTERNS = [
    r"\bfunding\b",
    r"\backnowledg(e)?ments?\b",
    r"\bauthor contributions?\b",
    r"\bcompeting interests?\b",
    r"\bconflicts? of interest\b",
    r"\bsupplementary( materials?| information)?\b",
    r"\bappendix\b",
    r"\bdata availability\b",
    r"\bethics\b",
]

def is_non_content_section(title: str) -> bool:
    t = normalize_text(title)
    for pat in NON_CONTENT_SEC_PATTERNS:
        if re.search(pat, t):
            return True
    return False

def extract_gold_section_titles_filtered(gold: Dict[str, Any]) -> List[str]:
    """Extract section titles from the gold standard and filter out non-content sections"""
    titles = []
    for sec in gold.get("sections", []):
        t = sec.get("title")
        if t and isinstance(t, str) and t.strip():
            if not is_non_content_section(t):
                titles.append(t.strip())
    return titles

def extract_gold_fulltext(gold: Dict[str, Any]) -> str:
    """Concatenate all paragraph texts from the gold standard as reference content"""
    texts = []
    for sec in gold.get("sections", []):
        for para in sec.get("paragraphs", []):
            tx = para.get("text")
            if tx and isinstance(tx, str):
                texts.append(tx.strip())
    return "\n".join(texts).strip()

# ------------------------- Read/Extract: Our Results -------------------------
def load_selected_titles(csv_path: str) -> List[str]:
    """读取 ranked_papers.csv 中 tag == 'selected' 的 title"""
    df = pd.read_csv(csv_path)
    if "tag" in df.columns:
        df = df[df["tag"] == "selected"]
    titles = []
    for t in df.get("title", []):
        if isinstance(t, str) and t.strip():
            titles.append(t.strip())
    return titles

def load_outline_section_titles(path: str) -> List[str]:
    """Extract the major and sub-section titles from our outline"""
    with open(path, "r", encoding="utf-8") as f:
        outline = json.loads(clean_json_string(f.read()))
    titles = []
    for sec in outline.get("outline", []):
        t = sec.get("section_title")
        if t and isinstance(t, str) and t.strip():
            titles.append(t.strip())
        subs = sec.get("subsections", {}) or {}
        if isinstance(subs, dict):
            for sub_t in subs.keys():
                if sub_t and isinstance(sub_t, str) and sub_t.strip():
                    titles.append(sub_t.strip())
        elif isinstance(subs, list):
            for sub_t in subs:
                if sub_t and isinstance(sub_t, str) and sub_t.strip():
                    titles.append(sub_t.strip())
    return titles

def stringify_outline_for_prompt_from_titles(title_list: List[str]) -> str:
    if not title_list:
        return "(empty outline)"
    return "\n".join([f"- {t}" for t in title_list])

def load_generated_fulltext(path_primary: str, path_fallback: str) -> str:
    path = path_primary if Path(path_primary).exists() else path_fallback
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    contents = []
    if isinstance(data, dict):
        if "subsections" in data and isinstance(data["subsections"], list):
            for it in data["subsections"]:
                c = it.get("content")
                if isinstance(c, str) and c.strip():
                    contents.append(c.strip())
        elif "sections" in data and isinstance(data["sections"], list):
            for it in data["sections"]:
                c = it.get("content")
                if isinstance(c, str) and c.strip():
                    contents.append(c.strip())
        else:
            def collect_strings(obj):
                if isinstance(obj, dict):
                    for v in obj.values():
                        collect_strings(v)
                elif isinstance(obj, list):
                    for v in obj:
                        collect_strings(v)
                elif isinstance(obj, str):
                    if obj.strip():
                        contents.append(obj.strip())
            collect_strings(data)
    return "\n".join(contents).strip()

# ------------------------- Metric Calculation -------------------------
def compute_reference_prf(gold_titles: List[str], selected_titles: List[str], thr: float) -> Dict[str, Any]:
    """PRF based on title lexical similarity (≥ thr counts as a match)"""
    gold_norm = [normalize_text(t) for t in gold_titles]
    sel_norm  = [normalize_text(t) for t in selected_titles]

    matched_sel_idx = set()
    matched_gold_idx = set()

    for i, s in enumerate(sel_norm):
        best_j, best_sim = -1, -1.0
        for j, g in enumerate(gold_norm):
            sim = difflib.SequenceMatcher(None, s, g).ratio()
            if sim > best_sim:
                best_sim, best_j = sim, j
        if best_sim >= thr:
            matched_sel_idx.add(i)
            matched_gold_idx.add(best_j)

    tp = len(matched_sel_idx)
    prec = tp / len(sel_norm) if sel_norm else 0.0
    rec  = tp / len(gold_norm) if gold_norm else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0

    return {
        "selected_count": len(sel_norm),
        "gold_ref_count": len(gold_norm),
        "matched_count": tp,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

def embed_texts(model: SentenceTransformer, texts: List[str]):
    if not texts:
        return None
    embs = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True).cpu().numpy()
    return embs

def greedy_semantic_matching(A: List[str], B: List[str], model: SentenceTransformer, thr: float) -> Tuple[int, Any]:
    """Semantic matching"""
    if not A or not B:
        return 0, None
    A_emb = embed_texts(model, A)
    B_emb = embed_texts(model, B)
    S = cosine_similarity(A_emb, B_emb)

    matched_a = set()
    matched_b = set()
    pairs = []

    flat = []
    for i in range(len(A)):
        for j in range(len(B)):
            flat.append((S[i, j], i, j))
    flat.sort(reverse=True, key=lambda x: x[0])

    for sim, i, j in flat:
        if sim < thr:
            break
        if i in matched_a or j in matched_b:
            continue
        matched_a.add(i)
        matched_b.add(j)
        pairs.append((i, j, float(sim)))

    return len(pairs), pairs

def structural_consistency(gen_sections: List[str], gold_sections: List[str], model: SentenceTransformer, thr: float) -> Dict[str, Any]:
    match_count, pairs = greedy_semantic_matching(gen_sections, gold_sections, model, thr)
    denom = len(gen_sections) + len(gold_sections)
    score = match_count / denom if denom > 0 else 0.0
    return {
        "generated_sections": len(gen_sections),
        "gold_sections": len(gold_sections),
        "matched_sections": match_count,
        "consistency": score,
        "pairs": [
            {
                "gen_idx": gi,
                "gold_idx": gj,
                "gen_title": gen_sections[gi],
                "gold_title": gold_sections[gj],
                "similarity": sim
            } for gi, gj, sim in (pairs or [])
        ]
    }

def compute_rouge_and_semantic(gen_text: str, gold_text: str, model: SentenceTransformer) -> Dict[str, Any]:
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeLsum'], use_stemmer=True)
    r = scorer.score(gold_text, gen_text)
    rouge = {
        "rouge1": {"p": r['rouge1'].precision, "r": r['rouge1'].recall, "f": r['rouge1'].fmeasure},
        "rouge2": {"p": r['rouge2'].precision, "r": r['rouge2'].recall, "f": r['rouge2'].fmeasure},
        "rougeLsum": {"p": r['rougeLsum'].precision, "r": r['rougeLsum'].recall, "f": r['rougeLsum'].fmeasure},
    }


    embs = embed_texts(model, [gen_text, gold_text])
    if embs is None or len(embs) < 2:
        sem_sim = 0.0
    else:
        sem_sim = float(cosine_similarity([embs[0]], [embs[1]])[0, 0])

    return {"rouge": rouge, "semantic_similarity": sem_sim}

# ------------------------- LLM Structural Consistency Scoring -------------------------
def build_llm_structure_prompt(topic: str, gen_outline_titles: List[str], gold_outline_titles: List[str]) -> str:
    gen_outline_str  = stringify_outline_for_prompt_from_titles(gen_outline_titles)
    gold_outline_str = stringify_outline_for_prompt_from_titles(gold_outline_titles)
    return f"""
You are an academic expert tasked with evaluating a draft outline for a literature survey on the topic of "{topic}".

You will be provided with two outlines:
1.A draft outline  written as a preliminary version.
--{gen_outline_str}

2.A gold-standard outline created  by domain experts.
--{gold_outline_str}

Your task is to rate the structural consistency between the draft outline and the gold-standard outline, following the criteria below:
Score 1 (Very Poor): The draft outline is largely inconsistent with the gold-standard; major sections are missing or irrelevant.
Score 2 (Poor): The draft outline only partially overlaps with the gold-standard; some key themes are omitted or misplaced.
Score 3 (Moderate): The draft includes some of the core topics but lacks structural alignment or completeness.
Score 4 (Good): The draft mostly aligns with the gold-standard; minor deviations in structure or scope.
Score 5 (Excellent): The draft closely mirrors the gold-standard in both structure and content coverage.

Your output should only include a single score from 1 to 5. Do not provide any explanation or additional text.
""".strip()

def call_llm_for_score(prompt: str, max_retries: int = LLM_MAX_RETRIES) -> int:
    if not USE_LLM_STRUCTURE_SCORE:
        return None
    if OpenAI is None:
        raise RuntimeError("pip install openai）")

    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_TEMPERATURE
            )
            text = (resp.choices[0].message.content or "").strip()
            text = clean_json_string(text)
            m = re.search(r"[1-5]", text)
            if m:
                return int(m.group(0))
            else:
                last_err = f"Failed to parse score: {text!r}"
        except Exception as e:
            last_err = e
        time.sleep(1.2 * attempt)
    raise RuntimeError(f"LLM structural scoring failed: {last_err}")


def main():
    # Read the gold standard
    gold = load_gold_json(GOLD_JSON_PATH)
    gold_ref_titles = extract_gold_reference_titles(gold)
    gold_sec_titles_all = extract_gold_section_titles_filtered(gold)
    gold_fulltext   = extract_gold_fulltext(gold)

    # Read generated survey
    selected_titles = load_selected_titles(RANKED_PAPERS_CSV)
    gen_sec_titles  = load_outline_section_titles(SURVEY_OUTLINE_JSON)
    gen_fulltext    = load_generated_fulltext(SURVEY_CONTENT_JSON, SURVEY_SECTIONS_JSON)

    # Reference PRF
    prf = compute_reference_prf(gold_ref_titles, selected_titles, TITLE_TEXTUAL_THR)

    # Structure
    model = SentenceTransformer(EMBED_MODEL_NAME)
    struct = structural_consistency(gen_sec_titles, gold_sec_titles_all, model, SECTION_SEM_THR)

    # Content
    quality = compute_rouge_and_semantic(gen_fulltext, gold_fulltext, model)

    # LLM score
    topic_csv = Path("topic.csv")
    if topic_csv.exists():
        topic_df = pd.read_csv(topic_csv)
        topic = str(topic_df.loc[0, "title"]).strip()
    else:
        topic = ""

    llm_score = None
    if USE_LLM_STRUCTURE_SCORE:
        prompt = build_llm_structure_prompt(topic, gen_sec_titles, gold_sec_titles_all)
        try:
            llm_score = call_llm_for_score(prompt)
        except Exception as e:
            print(f"⚠️ Problem occurred in LLM structural scoring: {e}")


    print("\n================== Reference PRF (Title textual) ==================")
    print(f"Gold references: {prf['gold_ref_count']} | Selected: {prf['selected_count']} | Matched: {prf['matched_count']}")
    print(f"Precision: {prf['precision']:.4f} | Recall: {prf['recall']:.4f} | F1: {prf['f1']:.4f}")

    print("\n================== Structural Consistency (Semantic) ===============")
    print(f"Gold sections (filtered): {struct['gold_sections']} | Generated sections: {struct['generated_sections']} | Matched: {struct['matched_sections']}")
    print(f"Consistency = matched / (gold + generated) = {struct['matched_sections']} / ({struct['gold_sections']} + {struct['generated_sections']}) = {struct['consistency']:.4f}")
    if struct.get("pairs"):
        print("Matched pairs (top):")
        for p in struct["pairs"]:
            print(f"  - GEN[{p['gen_idx']}] '{p['gen_title']}'  <->  GOLD[{p['gold_idx']}] '{p['gold_title']}' | sim={p['similarity']:.3f}")

    if llm_score is not None:
        print("\n================== Structural Consistency (LLM Score) =============")
        print(f"LLM structural consistency score (1-5): {llm_score}")

    print("\n================== Content Quality ================================")
    r = quality["rouge"]
    print(f"ROUGE-1  P/R/F: {r['rouge1']['p']:.4f} / {r['rouge1']['r']:.4f} / {r['rouge1']['f']:.4f}")
    print(f"ROUGE-2  P/R/F: {r['rouge2']['p']:.4f} / {r['rouge2']['r']:.4f} / {r['rouge2']['f']:.4f}")
    print(f"ROUGE-Ls P/R/F: {r['rougeLsum']['p']:.4f} / {r['rougeLsum']['r']:.4f} / {r['rougeLsum']['f']:.4f}")
    print(f"Semantic Similarity (full text): {quality['semantic_similarity']:.4f}")

    #Save JSON
    out = {
        "reference_prf": prf,
        "structural_consistency_semantic": struct,
        "structural_consistency_llm_score": llm_score,
        "content_quality": quality
    }
    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n✅ Evaluation completed, details saved to evaluation_report.json")


if __name__ == "__main__":
    main()
