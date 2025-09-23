import requests
import time
import json
import os
import re
import csv
import random
from collections import Counter

INPUT_FILE = "./example/topic.csv"
OUTPUT_DIR = "./"

LIMIT = 100  # Semantic Scholar max
MAX_RESULTS_PER_QUERY = 300
WAIT_SECONDS = 5

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "corpusId,externalIds,title,abstract,citationCount,referenceCount,influentialCitationCount,publicationDate,journal,authors,references"

def safe_get(d, *keys, default=None):
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d

def is_valid_paper(paper):
    doi = safe_get(paper, "externalIds", "DOI", default="")
    abstract = paper.get("abstract", "")
    return bool(doi and abstract)

def extract_data(paper):
    # authors = paper.get("authors") or []
    # if authors:
    #     authors = [authors[0]] if len(authors) == 1 else [authors[0], authors[-1]]
    #
    # references = paper.get("references") or []
    # #references = [{"paperId": ref.get("paperId")} for ref in references if ref.get("paperId")]

    return {
        "corpusId": paper.get("corpusId", ""),
        "doi": safe_get(paper, "externalIds", "DOI", default=""),
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract", ""),
        # "citationCount": paper.get("citationCount", 0),
        # "referenceCount": paper.get("referenceCount", 0),
        # "influentialCitationCount": paper.get("influentialCitationCount", 0),
        # "publicationDate": paper.get("publicationDate", ""),
        "journalName": safe_get(paper, "journal", "name", default=""),
        # "authors": authors,
        # "references": references
    }

def fetch_papers_by_keyword_query(query, year_end, max_results=300):
    collected = []
    offset = 0
    while len(collected) < max_results:
        params = {
            "query": query,
            "fields": FIELDS,
            "limit": LIMIT,
            "offset": offset,
            "year": f"2000-{year_end}",
        }
        print(f"🔍 Searching: {query} | Offset: {offset} | Year ≤ {year_end}")
        response = requests.get(BASE_URL, params=params)

        if response.status_code == 429:
            print("⏳ Hit rate limit (429). Waiting before retrying...")
            time.sleep(15 + random.randint(5, 10))
            continue
        if response.status_code == 400:
            print(f"❌ Error 400: Likely query too complex or offset too large.")
            break
        if response.status_code != 200:
            print(f"⚠Error {response.status_code} for query: {query}")
            break
        papers = response.json().get("data", [])
        if not papers:
            print("🛑 No more papers returned by API.")
            break

        valid_count_before = len(collected)
        for paper in papers:
            if is_valid_paper(paper):
                collected.append(paper)
                if len(collected) >= max_results:
                    break

        added_valid = len(collected) - valid_count_before
        print(f"✅ Added {added_valid} valid papers (total: {len(collected)})")

        if len(papers) < LIMIT:
            print("🚫 Reached last page of results (less than LIMIT returned).")
            break

        offset += LIMIT
        time.sleep(2 + random.random() * 2)

    return collected[:max_results]

def sanitize_filename(text: str) -> str:
    text = re.sub(r"[\\/*?\"<>|:\n\r\t]", "_", text)
    return text.strip()[:100] or "untitled"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(INPUT_FILE, "r", encoding="utf-8") as infile:
        reader = list(csv.DictReader(infile))  #
        total = len(reader)

        for i, row in enumerate(reader):
            if i >= 1:
                break

            print(f"\n📘 Processing article {i + 1} of {total}...")
            try:
                title = row["title"].strip()
                year = row["year"].strip()
                keywords_raw = row["keywords"].strip()
                if not (title and year and keywords_raw):
                    print("⚠Skipping due to missing title/year/keywords.")
                    continue

                year = int(year)
                keywords = [kw.strip() for kw in keywords_raw.split(";") if kw.strip()]
                keyword_query = ", ".join(keywords)

                related_papers = fetch_papers_by_keyword_query(keyword_query, year, MAX_RESULTS_PER_QUERY)
                result_list = []

                for paper in related_papers:
                    if not is_valid_paper(paper):  # ✅ Require DOI and abstract to exist
                        continue
                    data = extract_data(paper)
                    # data["authors"] = json.dumps(data["authors"], ensure_ascii=False)
                    # data["references"] = json.dumps(data["references"], ensure_ascii=False)
                    result_list.append(data)

                if result_list:
                    safe_title = sanitize_filename(title)
                    filename = f"{safe_title}.json"
                    output_path = os.path.join(OUTPUT_DIR, filename)
                    with open(output_path, "w", encoding="utf-8") as out_file:
                        json.dump(result_list, out_file, ensure_ascii=False, indent=2)
                    print(f"📁 Saved to: {output_path}")
                else:
                    print("⚠No valid results found.")

                time.sleep(WAIT_SECONDS)

            except Exception as e:
                print(f"❌ Error processing row {i + 1}: {e}")
                continue
    print(f"\n✅ All done. Results saved in '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()

