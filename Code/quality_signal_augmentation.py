import json
import requests
import time
from collections import Counter

# 配置参数
BATCH_SIZE = 10  # OpenAlex allows up to 50 OR conditions per batch
API_MAILTO = "your email address"  # Replace with your email
RATE_LIMIT = 1  # Seconds per request (complies with free API limits)

def normalize_doi(doi):
    """Normalize DOI"""
    if doi:
        return doi.lower().replace("https://doi.org/", "").strip()
    return ""
def invert_abstract(inverted_index):
    """get abstract"""
    if not inverted_index:
        return ""
    word_positions = {
        pos: word
        for word, positions in inverted_index.items()
        for pos in positions
    }
    return ' '.join([word_positions.get(i, '') for i in range(max(word_positions.keys(), default=-1) + 1)])
# -------------------- 批处理 works --------------------
def process_doi_batch(batch_dois, original_data):
    """Query OpenAlex works in batches via DOI"""
    base_url = f"https://api.openalex.org/works?mailto={API_MAILTO}&filter=doi:"
    query_filter = "|".join(["https://doi.org/" + normalize_doi(doi) for doi in batch_dois])
    url = base_url + query_filter

    try:
        response = requests.get(url)
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 5))
            print(f"⚠️ Rate limit triggered, waiting {retry_after} seconds...")
            time.sleep(retry_after)
            return process_doi_batch(batch_dois, original_data)

        if response.status_code != 200:
            print(f"❌ Request failed, status code: {response.status_code}")
            return original_data

        results = response.json().get('results', [])
        return enhance_batch_data(original_data, results)

    except Exception as e:
        print(f"error: {str(e)}")
        return original_data

def safe_get(d, key, default=None):
    if isinstance(d, dict):
        return d.get(key, default)
    return default

def merge_data(original, openalex_data):
    if not isinstance(openalex_data, dict):
        return original

    # --- Extract first-author and last-author IDs ---
    first_author_id, last_author_id = None, None
    authorships = safe_get(openalex_data, 'authorships', []) or []
    for auth in authorships:
        if not isinstance(auth, dict):
            continue
        if auth.get("author_position") == "first":
            first_author_id = safe_get(auth.get("author"), "id")
        elif auth.get("author_position") == "last":
            last_author_id = safe_get(auth.get("author"), "id")

    # --- Extract journal ID ---
    primary_location = safe_get(openalex_data, "primary_location", {})
    source = safe_get(primary_location, "source", {})
    journal_id = safe_get(source, "id")

    # --- Extract citation count ---
    cited_by_count = safe_get(openalex_data, "cited_by_count", 0) or 0

    # --- update ---
    original.update({
        'abstract': invert_abstract(safe_get(openalex_data, 'abstract_inverted_index', {})) or original.get('abstract', ''),
        'topics': [c.get('display_name', '') for c in (safe_get(openalex_data, 'topics', []) or []) if isinstance(c, dict)],
        'keywords': [kw.get('display_name', '') for kw in (safe_get(openalex_data, 'keywords', []) or []) if isinstance(kw, dict)],
        'referenced_works': safe_get(openalex_data, 'referenced_works', []) or [],
        'referenced_works_count': len(safe_get(openalex_data, 'referenced_works', []) or []),
        'first_author_id': first_author_id,
        'last_author_id': last_author_id,
        'journal_id': journal_id,
        'cited_by_count': cited_by_count
    })
    return original

def enhance_batch_data(original_batch, api_results):
    """Enhance batch data"""
    doi_map = {}
    for item in api_results:
        if item is None:
            print("⚠️ Warning: OpenAlex API return None.")
            with open("debug_none.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"bad_item": item}, ensure_ascii=False) + "\n")
            continue
        if not isinstance(item, dict):
            print(f"⚠️ Warning: Unexpected type {type(item)} in api_results: {item}")
            with open("debug_badtype.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"bad_item": str(item)}, ensure_ascii=False) + "\n")
            continue
        doi_val = normalize_doi(item.get("doi"))
        if doi_val:
            doi_map[doi_val] = item

    enhanced_data = []
    for orig_item in original_batch:
        orig_doi = normalize_doi(orig_item.get('doi'))
        match = doi_map.get(orig_doi)

        if match:
            try:
                enhanced_data.append(merge_data(orig_item, match))
            except Exception as e:
                print(f"❌ merge_data error，doi={orig_doi}, error={e}")
                with open("debug_merge_fail.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "doi": orig_doi,
                        "match": match
                    }, ensure_ascii=False) + "\n")
                raise
        else:
            orig_item.setdefault('topics', [])
            orig_item.setdefault('keywords', [])
            orig_item.setdefault('referenced_works', [])
            orig_item.setdefault('referenced_works_count', 0)
            orig_item.setdefault('first_author_id', None)
            orig_item.setdefault('last_author_id', None)
            orig_item.setdefault('journal_id', None)
            orig_item.setdefault('cited_by_count', 0)
            enhanced_data.append(orig_item)
    return enhanced_data

# -------------------- Batch Processing Author & Journal h-index --------------------

def fetch_author_hindex_batch(ids):
    """Batch query author h-index (<=50 per batch)"""
    if not ids:
        return {}
    url = f"https://api.openalex.org/authors?mailto={API_MAILTO}&filter=ids.openalex:" + "|".join(ids)
    resp = requests.get(url)
    print(url)
    if resp.status_code != 200:
        print(f"❌ Author error: {resp.status_code}")
        return {}
    results = resp.json().get("results", [])
    return {r["id"]: r.get("summary_stats", {}).get("h_index", 0) for r in results}


def fetch_journal_hindex_batch(ids):
    """Batch query journal h-index (<=50 per batch)"""
    if not ids:
        return {}
    url = f"https://api.openalex.org/sources?mailto={API_MAILTO}&filter=ids.openalex:" + "|".join(ids)
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"❌ Journal error: {resp.status_code}")
        return {}
    results = resp.json().get("results", [])
    return {r["id"]: r.get("summary_stats", {}).get("h_index", 0) for r in results}

if __name__ == "__main__":
    input_file = "./Gold_Standard_Deep Learning Applications in Computer Vision.json"
    output_file = "./final_ref_set.json"

    with open(input_file, "r", encoding="utf-8") as fin:
        data = json.load(fin)

    total_processed = 0
    batch = []
    enhanced_results = []

    # Step1:  works
    for paper in data:
        doi = paper.get('doi')
        if not doi:
            paper.setdefault('topics', [])
            paper.setdefault('keywords', [])
            paper.setdefault('referenced_works', [])
            paper.setdefault('referenced_works_count', 0)
            paper.setdefault('first_author_id', None)
            paper.setdefault('last_author_id', None)
            paper.setdefault('journal_id', None)
            paper.setdefault('cited_by_count', 0)
            enhanced_results.append(paper)
            total_processed += 1
            continue

        batch.append(paper)
        if len(batch) >= BATCH_SIZE:
            enhanced = process_doi_batch([p['doi'] for p in batch], batch)
            enhanced_results.extend(enhanced)
            total_processed += len(enhanced)
            batch = []
            print(f"Processed {total_processed} papers")
            time.sleep(RATE_LIMIT)

    if batch:
        enhanced = process_doi_batch([p['doi'] for p in batch], batch)
        enhanced_results.extend(enhanced)
        total_processed += len(enhanced)

    # Step2: Collect authors / journals to query (from original papers)
    author_ids = set()
    journal_ids = set()
    for paper in enhanced_results:
        if paper.get("first_author_id"):
            author_ids.add(paper["first_author_id"])
        if paper.get("last_author_id"):
            author_ids.add(paper["last_author_id"])
        if paper.get("journal_id"):
            journal_ids.add(paper["journal_id"])

    # Step3: Batch query author h-index
    author_hindex_map = {}
    author_ids = list(author_ids)
    for i in range(0, len(author_ids), BATCH_SIZE):
        batch_ids = author_ids[i:i+BATCH_SIZE]
        author_hindex_map.update(fetch_author_hindex_batch(batch_ids))
        time.sleep(RATE_LIMIT)

    # Step4: Batch query journal h-index
    journal_hindex_map = {}
    journal_ids = list(journal_ids)
    for i in range(0, len(journal_ids), BATCH_SIZE):
        batch_ids = journal_ids[i:i+BATCH_SIZE]
        journal_hindex_map.update(fetch_journal_hindex_batch(batch_ids))
        time.sleep(RATE_LIMIT)

    # Step5: Merge back into papers
    for paper in enhanced_results:
        paper["first_author_hindex"] = author_hindex_map.get(paper.get("first_author_id"))
        paper["last_author_hindex"] = author_hindex_map.get(paper.get("last_author_id"))
        paper["journal_hindex"] = journal_hindex_map.get(paper.get("journal_id"))

    # Step6: Co-citation statistics (only count original papers)
    ref_counter = Counter()
    for paper in enhanced_results:
        for ref in paper.get("referenced_works", []):
            ref_counter[ref] += 1
    top_refs = ref_counter.most_common(50)
    print("Top 50 most co-occurring references:", len(top_refs))

    # Step7: Batch request these works
    co_cited_results = []
    for i in range(0, len(top_refs), BATCH_SIZE):
        batch_ids = [rid for rid, _ in top_refs[i:i + BATCH_SIZE]]
        url = f"https://api.openalex.org/works?mailto={API_MAILTO}&filter=ids.openalex:" + "|".join(batch_ids)
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f"❌ co-cited error: {resp.status_code}")
            continue
        results = resp.json().get("results", [])
        co_cited_results.extend(results)
        time.sleep(RATE_LIMIT)

    # Step8: Convert to co_cited_x format and add co_cited_count + author/journal info
    final_co_cited = []
    id_to_count = dict(top_refs)
    for idx, item in enumerate(co_cited_results, start=1):
        if not isinstance(item, dict):
            continue
        work_id = item.get("id")

        # Extract first author / last author / journal
        first_author_id, last_author_id = None, None
        for auth in item.get("authorships", []):
            if auth.get("author_position") == "first":
                first_author_id = (auth.get("author") or {}).get("id")
            elif auth.get("author_position") == "last":
                last_author_id = (auth.get("author") or {}).get("id")
        journal_id = ((item.get("primary_location") or {}).get("source") or {}).get("id")

        record = {
            f"co_cited_{idx}": work_id,
            "doi": item.get("doi"),
            "title": item.get("title"),
            "abstract": invert_abstract(item.get("abstract_inverted_index", {})),
            "topics": [t.get("display_name", "") for t in (item.get("topics") or []) if isinstance(t, dict)],
            "keywords": [kw.get("display_name", "") for kw in (item.get("keywords") or []) if isinstance(kw, dict)],
            "referenced_works": item.get("referenced_works", []),
            "referenced_works_count": len(item.get("referenced_works", []) or []),
            "cited_by_count": item.get("cited_by_count", 0),
            "co_cited_count": id_to_count.get(work_id, 0),
            "first_author_id": first_author_id,
            "last_author_id": last_author_id,
            "journal_id": journal_id
        }
        final_co_cited.append(record)

    # Step9: Query co-cited authors / journals h-index again
    co_author_ids = set()
    co_journal_ids = set()
    for paper in final_co_cited:
        if paper.get("first_author_id"):
            co_author_ids.add(paper["first_author_id"])
        if paper.get("last_author_id"):
            co_author_ids.add(paper["last_author_id"])
        if paper.get("journal_id"):
            co_journal_ids.add(paper["journal_id"])

    for i in range(0, len(co_author_ids), BATCH_SIZE):
        batch_ids = list(co_author_ids)[i:i+BATCH_SIZE]
        author_hindex_map.update(fetch_author_hindex_batch(batch_ids))
        time.sleep(RATE_LIMIT)

    for i in range(0, len(co_journal_ids), BATCH_SIZE):
        batch_ids = list(co_journal_ids)[i:i+BATCH_SIZE]
        journal_hindex_map.update(fetch_journal_hindex_batch(batch_ids))
        time.sleep(RATE_LIMIT)

    for paper in final_co_cited:
        paper["first_author_hindex"] = author_hindex_map.get(paper.get("first_author_id"))
        paper["last_author_hindex"] = author_hindex_map.get(paper.get("last_author_id"))
        paper["journal_hindex"] = journal_hindex_map.get(paper.get("journal_id"))

    # Step10: enhanced_results
    enhanced_results.extend(final_co_cited)

    # Step11: write to file
    with open(output_file, "w", encoding="utf-8") as fout:
        json.dump(enhanced_results, fout, ensure_ascii=False, indent=2)
    print(
        f"Processing completed! Processed {total_processed} papers + {len(final_co_cited)} co-cited papers, results saved to {output_file}")
