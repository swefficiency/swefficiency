
from swefficiency.perf_filter.attributes.constants import VERBATIM_KEYWORDS, BASE_PERF_KEYWORDS
import re
from collections import defaultdict
from datasets import load_dataset

KEYWORDS = {k.lower() for k in (set(VERBATIM_KEYWORDS) | set(BASE_PERF_KEYWORDS))}

SPLITS = ["test"]

def find_keyword_hits(text: str):
    text_low = text.lower()
    hits = []
    for kw in KEYWORDS:
        if kw in text_low:
            hits.append(kw)
    return hits

def scan_split(split_name: str):
    ds = load_dataset("princeton-nlp/SWE-bench", split=split_name)
    violations = []
    for idx, row in enumerate(ds):
        prob = row.get("problem_statement") or ""
        hints = row.get("hints_text") or ""
        fields = {"problem_statement": prob, "hints_text": hints}
        row_hits = {}
        for field_name, content in fields.items():
            hits = find_keyword_hits(content)
            if hits:
                row_hits[field_name] = hits
        if row_hits:
            violations.append({
                "idx": idx,
                "instance_id": row.get("instance_id"),
                "hits": row_hits
            })
    return violations

def main():
    overall = defaultdict(int)
    any_violations = False
    instance_ids = set()
    for split in SPLITS:
        try:
            violations = scan_split(split)
        except Exception as e:
            print(f"Failed to load or scan split {split}: {e}")
            continue
        if violations:
            any_violations = True
            print(f"\nSplit: {split} -> {len(violations)} items with keyword hits")
            for v in violations:
                print(f"- instance_id={v['instance_id']} idx={v['idx']} hits={v['hits']}")
                for field, kws in v["hits"].items():
                    for kw in kws:
                        overall[kw] += 1

                instance_ids.add(v['instance_id'])

    if not any_violations:
        print("No keywords found in any split.")
    else:
        print("\nSummary keyword frequencies:")
        for kw, count in sorted(overall.items(), key=lambda x: -x[1]):
            print(f"{kw}: {count}")
            
    # Print instance ids.
    print(" ".join(list(instance_ids)))
    
    

if __name__ == "__main__":
    main()