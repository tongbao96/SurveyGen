#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import json
import time
import pandas as pd
from pathlib import Path
from openai import OpenAI
from typing import Dict, Any, List

TOPIC_CSV       = "./example/topic.csv"
RANKED_FILE     = "paper_after_reranking.csv"
OUTLINE_JSON    = "survey_outline.json"
OUTPUT_DIR      = Path("sections")
OUTPUT_BUNDLE   = "survey_content.json"

client = OpenAI(
    base_url="your api address",
    api_key="your api key"
)

LLM_MODEL = "gpt-4o"
TEMPERATURE = 0
MAX_RETRIES_JSON = 8

def clean_json_string(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def ensure_valid_json(text: str) -> Dict[str, Any]:
    cleaned = clean_json_string(text)
    obj = json.loads(cleaned)
    if not isinstance(obj, dict):
        raise ValueError("JSON root must be an object.")
    if "content" not in obj or "references" not in obj:
        raise ValueError("JSON must contain 'content' and 'references'.")
    if not isinstance(obj["references"], list):
        raise ValueError("'references' must be a list.")
    return obj

def stringify_outline_for_prompt(outline_obj: Dict[str, Any]) -> str:
    title = outline_obj.get("title", "").strip()
    lines = [f"Survey Title: {title}", "Sections:"]
    for sec in outline_obj.get("outline", []):
        s_title = sec.get("section_title", "").strip()
        s_desc  = sec.get("description", "").strip()
        lines.append(f"- {s_title}: {s_desc}")
        subs = sec.get("subsections", {}) or {}
        if isinstance(subs, dict):
            for sub_title, sub_desc in subs.items():
                lines.append(f"    • {sub_title}: {str(sub_desc).strip()}")
    return "\n".join(lines)

def build_numbered_references(selected_df: pd.DataFrame, max_refs: int = None) -> List[Dict[str, str]]:
    refs = []
    for _, row in selected_df.iterrows():
        title = str(row.get("title", "")).strip()
        abstract = str(row.get("abstract", "")).strip()
        if not title or not abstract or abstract.lower() == "nan":
            continue
        refs.append({"title": title, "abstract": abstract})
        if max_refs is not None and len(refs) >= max_refs:
            break
    for i, r in enumerate(refs, start=1):
        r["refNo"] = i
    return refs

def format_references_for_prompt(refs: List[Dict[str, str]]) -> str:
    return "\n".join([f"- ref [{r['refNo']}]: Title: {r['title']}\n  Abstract: {r['abstract']}" for r in refs])

def build_subsection_prompt(topic: str, outline_text: str, subsection_title: str, refs_prompt: str) -> str:
    return f"""
You are an academic expert in the field of "{topic}" with deep expertise in survey writing. Your task is to write a subsection of a survey, based on the following information:  

The overall structure of the survey is as follows:
  - {outline_text}

You are now asked to write the following subsection:
- Subsection: {subsection_title}  

The following highly relevant papers, including their titles and abstracts, are provided for reference:
- {refs_prompt}

Instructions for generating subsection content:
    1. Carefully analyze the provided references and identify those highly relevant to the subsection topic as the basis for your generation.
    2. Ensure the content directly addresses the specific topic of the subsection, and the generated content should be a minimum of 300 words.
    3. Ensure that all claims are fully supported by relevant academic literature. Cite each reference using in-text citations in the format ref [1], ref [2], etc. If a source is cited multiple times, use only the reference number assigned to its first occurrence. 
    4. Maintain alignment with the parent section and overall survey topic, ensuring thematic and conceptual consistency.
    5. Use a formal academic tone, with logically structured arguments and scholarly language.
    6. Your output should strictly as a JSON object with the following two fields.
    {{
      "content": "the content of the subsection",
      "references": [
        {{
          "refNo": "reference number",
          "title": "title of the reference"
        }}
      ]
    }}

Now, please generate the content for the given subsection: {subsection_title}. Please ensure the output is formatted according to the requirements mentioned above. Do not include any explanation, commentary, or preamble in your response.
""".strip()

def call_llm_for_json(prompt: str):
    last_err = None
    for attempt in range(1, MAX_RETRIES_JSON + 1):
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE
        )
        text = resp.choices[0].message.content.strip()
        usage = resp.usage.model_dump() if resp.usage else {}
        try:
            obj = ensure_valid_json(text)
            return obj, usage
        except Exception as e:
            last_err = e
            print(f"⚠️ Attempt {attempt}/{MAX_RETRIES_JSON}: JSON parsing failed -> {e}")
    print(f"⚠️ Attempt {attempt}/{MAX_RETRIES_JSON}: JSON parsing failed -> {e}")


def main():
    start_time = time.time()

    topic = str(pd.read_csv(TOPIC_CSV).loc[0, "title"]).strip()
    selected = pd.read_csv(RANKED_FILE)
    selected = selected[selected["tag"] == "selected"].copy()

    with open(OUTLINE_JSON, "r", encoding="utf-8") as f:
        outline_obj = json.loads(clean_json_string(f.read()))
    outline_text = stringify_outline_for_prompt(outline_obj)

    refs = build_numbered_references(selected)
    refs_prompt = format_references_for_prompt(refs)

    outline_list = outline_obj.get("outline", []) or []
    total_big = len(outline_list)
    all_subs = []
    for big_idx, sec in enumerate(outline_list, start=1):
        subs = list((sec.get("subsections") or {}).keys())
        for sub_idx, sub_t in enumerate(subs, start=1):
            all_subs.append((big_idx, total_big, sec.get("section_title",""), sub_idx, len(subs), sub_t))

    OUTPUT_DIR.mkdir(exist_ok=True)
    bundle=[]
    global_tokens = {"prompt":0,"completion":0,"total":0}

    for global_idx,(big_idx,total_big,big_title,sub_idx,total_subs,sub_t) in enumerate(all_subs, start=1):
        print(
            f"\n===== Generating Section {big_idx}/{total_big}: \"{big_title}\" - Subsection {sub_idx}/{total_subs}: \"{sub_t}\" =====")
        print(f"📖 Overall progress: {global_idx}/{len(all_subs)} subsections")

        prompt = build_subsection_prompt(topic, outline_text, sub_t, refs_prompt)
        obj, usage = call_llm_for_json(prompt)

        safe_name=re.sub(r"[^a-zA-Z0-9\-_.]+","_",sub_t)[:120]
        out_path=OUTPUT_DIR/f"{big_idx:02d}_{sub_idx:02d}_{safe_name}.json"
        with open(out_path,"w",encoding="utf-8") as f:
            json.dump({
                "section_title":big_title,
                "subsection_title":sub_t,
                "content":obj.get("content",""),
                "references":obj.get("references",[])
            },f,indent=2,ensure_ascii=False)
        print(f"✅ Save: {out_path}")

        if usage:
            global_tokens["prompt"]+=usage.get("prompt_tokens",0)
            global_tokens["completion"]+=usage.get("completion_tokens",0)
            global_tokens["total"]+=usage.get("total_tokens",0)
            print(f"📊 Token : prompt={usage.get('prompt_tokens')} | completion={usage.get('completion_tokens')} | total={usage.get('total_tokens')}")
            print(f"📈 Token : prompt={global_tokens['prompt']} | completion={global_tokens['completion']} | total={global_tokens['total']}")

        bundle.append({
            "section_title":big_title,
            "subsection_title":sub_t,
            "content":obj.get("content",""),
            "references":obj.get("references",[])
        })

    with open(OUTPUT_BUNDLE,"w",encoding="utf-8") as f:
        json.dump({"title":outline_obj.get("title",topic),"subsections":bundle},f,indent=2,ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"\n🎉 All {len(all_subs)} subsections generated, bundled and saved to {OUTPUT_BUNDLE}")
    print(f"⏱ Total time: {elapsed / 60:.1f} minutes ({elapsed:.1f} seconds)")


if __name__=="__main__":
    main()
