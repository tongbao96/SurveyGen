#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from openai import OpenAI
import json
import re

TOPIC_CSV = "./examlpe/topic.csv"
RANKED_FILE = "paper_after_reranking.csv"
OUTPUT_JSON = "survey_outline.json"

# OpenAI API
client = OpenAI(
    base_url="your api address",
    api_key="your api key"
)

# ========== 1. get topic ==========
topic_df = pd.read_csv(TOPIC_CSV)
topic = str(topic_df.loc[0, "title"]).strip()

# ========== 2. Read Selected Papers ==========
df = pd.read_csv(RANKED_FILE)
selected = df[df["tag"] == "selected"]

references_list = ""
for _, row in selected.iterrows():
    title = str(row["title"]).strip()
    abstract = str(row["abstract"]).strip()
    if not abstract or abstract.lower() == "nan":
        continue
    references_list += f"\n- Title: {title}\n  Abstract: {abstract}\n"

# ========== 3. prompt ==========
prompt = f"""
You are an academic expert in the field of "{topic}" with deep expertise in survey writing. Your task is to generate a well-structured outline for a survey on this topic. 

The following highly relevant papers, including their titles and abstracts, are provided for reference:
{references_list}

Please follow the instructions below:
Step 1: Based on the given topic, identify 3 to 7 major thematic sections that define the overall scope and objectives of the survey. For each section, provide a academically styled title, along with a brief description summarizing its focus and relevance. 

Step 2: For the major thematic sections, list several subsections representing more specific research areas, concepts, or points to be covered. Subsections should be conceptually related to their parent section and serve to further structure the survey.

Step 3: Your output should be in JSON format and must include the survey title, a structured outline with section titles, descriptions, and subsections. Ensure the structure is logically coherent, well-aligned with the topic, and suitable for developing a full-length academic survey. 

The output example should follow the format below:
{{
  "title": "TITLE OF THE SURVEY",
  "outline": [
    {{
      "section_title": "SECTION TITLE",
      "description": "A brief description summarizing the focus and scope of this section.",
      "subsections": {{
            "subsection title1": "content",
            "...": "...",
            "subsection title n": "content"
      }}
    }},
    {{
      "section_title": "SECTION TITLE 2",
      "description": "Another major theme description.",
      "subsections": {{
            "subsection title 1": "content",
            "...": "...",
            "subsection title n": "content"
      }}
    }}
  ]
}}

Now, based on the given topic {topic}, please generate the outline by following the steps above. Do not include any additional commentary, explanation, or formatting instructions, only return the structured JSON output as specified.
"""

# ========== 4. Call LLM ==========
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

outline_json = response.choices[0].message.content.strip()

# ========== 4.1 Clean GPT Output ==========
def clean_json_string(s: str) -> str:
    s = re.sub(r"^```(?:json)?", "", s.strip(), flags=re.IGNORECASE)
    s = re.sub(r"```$", "", s.strip())
    return s.strip()

cleaned = clean_json_string(outline_json)

# Verify whether it is valid JSON
try:
    parsed = json.loads(cleaned)
except Exception as e:
    print("❌ GPT output is not valid JSON:", e)
    parsed = None

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    if parsed is not None:
        json.dump(parsed, f, indent=2, ensure_ascii=False)
    else:
        f.write(cleaned)

print(f"✅ Outline generated and saved to {OUTPUT_JSON}")
