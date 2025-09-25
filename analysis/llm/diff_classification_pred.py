SYSTEM_PROMPT = """\
You are an excellent performance engineer. Given a code diff and an affected performance workload that shows a speedup as a result of this diff, output a single high-level performance bucket, the concrete signals from the diff that justify that bucket, a mechanism-level explanation of **why the specific code edits** improve performance, and a confidence score. Prefer software-side mechanisms; ignore hardware/microarchitecture unless explicitly cited.

## Inputs

* Commit message and code diff(s).
* Optional: filenames, symbols, and language contexts.
* **Do not** consider runtime/benchmark numbers; classify based on the code changes themselves.

## Buckets (choose **exactly one** `classification`)

1. **Algorithmic / Data Structure Improvements** — Better asymptotic complexity or more suitable data structures; removes redundant passes.
2. **Memory Efficiency & Management** — Fewer allocations/copies; pooling/reuse; layout/locality changes; alignment; `reserve()`; SoA/AoS.
3. **Concurrency & Parallelism** — Threading/async; work partitioning; lock scope/structure; atomics; SIMD/vectorization.
4. **I/O and Storage Efficiency** — Fewer syscalls; buffering/batching; async I/O; (de)serialization changes; payload trimming.
5. **Code Simplification / Dead-Code Elimination** — Early exits; pruning logging/asserts on hot paths; removing unnecessary work/branches.
6. **Compiler / Build / Low-level Tuning** — Flags (LTO/PGO), inlining hints, intrinsics, UB fixes enabling optimization, branch hints.
7. **Configuration / Parameter Tuning** — Constants/thresholds/buffer sizes, thread-pool/GC settings, feature flags altering performance behavior.
8. **Caching & Reuse** — Memoization, caches, reuse of precomputed artifacts/results, avoiding repeated expensive calls.
9. **Unknown / Not Enough Information** — Claimed speedup but mechanism not inferable from available changes.

## Secondary Tags (optional)

* Other relevant buckets or keywords, if any (e.g., "Memory Efficiency & Management" and "Caching & Reuse" could both apply).
* These can be more specific, e.g., "memoization", "lock-free", "buffered I/O", but try to use standard terms where possible.

### Disambiguation rules

* **Algorithmic vs Caching**: If an algorithm was fundamentally changed and a cache was added as a helper, choose **Algorithmic**; add `memoization` in `mechanism_signals`.
* **Concurrency vs Algorithmic**: If parallelism is added without changing the algorithm, choose **Concurrency & Parallelism**.
* **I/O vs Memory**: If copies were removed primarily to cut syscalls or shrink payloads, choose **I/O**; if focused on allocation/locality/pressure, choose **Memory**.
* **Compiler/Build**: Choose **Compiler / Build** when source logic is the same but flags/hints/toolchain changed.
* **Benchmark-only**: Choose **Workload/Benchmark-Only** when only harness/warmup/affinity/timers changed.

## What to extract as “signals”

Short, concrete phrases tied to the diff, e.g.:

* containers/algos: “`vector`→`unordered_map`”, “removed nested loop”, “added binary search”, “streaming parse”
* memory: “added `reserve()`”, “object pool”, “moved to stack allocation”, “SoA layout”
* concurrency: “introduced thread pool”, “reduced lock scope”, “lock-free queue”, “SIMD intrinsics”
* I/O: “batched writes”, “buffered reader”, “protobuf→flat serialization”, “compression level tuned”, “fewer syscalls”
* simplification: “early return before parse”, “pruned logging on hot path”, “deleted dead branch”
* compiler/build: “enabled LTO/PGO”, “added `inline`/`cold`/`likely`”, “UB fix unblocking vectorization”
* config: “increased read buffer to 1MB”, “thread pool size = cores”
* caching: “added LRU cache”, “memoized function result”
* meta: “only bench harness changed”

## Output Requirements (STRICT)

* Output **only** the JSON object — no prose, no Markdown, no code fences.
* Keep `explanation` ≤ 6 sentences and tie it to specific lines/files/patterns from the diff.
* If evidence is weak or ambiguous, use `classification: "Unknown / Not Enough Information"` and lower `confidence`.

### JSON Schema

```json
{
  "classification": "<one of the 10 buckets>",
  "secondary_tags": ["<optional: other relevant buckets or keywords, if any>"],
  "mechanism_signals": ["<short phrases pulled from the diff that justify the classification>"],
  "affected_components": ["<files/modules/functions inferred from paths/symbols>"],
  "explanation": "<mechanism-level rationale grounded in the diff: what changed, how it reduces work/contention/latency/allocs/syscalls, and why that maps to the chosen bucket>",
  "confidence": "<high|medium|low>"
}
```

## Final sanity check (do this before emitting JSON)

1. Have I picked **exactly one** bucket that best explains the performance mechanism?
2. Do my `mechanism_signals` cite concrete code changes from the diff that motivate that bucket?
3. Is the explanation mechanism-centric and grounded in the edits (not benchmarks)?

---

### Mini-examples

**A. Algorithmic / Data Structure Improvements**

```json
{
  "classification": "Algorithmic / Data Structure Improvements",
  "secondary_tags": ["asymptotic complexity"],
  "mechanism_signals": ["removed nested O(n^2) scan", "introduced unordered_set", "added reserve() to avoid rehash"],
  "affected_components": ["src/import/dedupe.cpp", "Importer::dedupeRecords"],
  "explanation": "The patch replaces a quadratic duplicate search with hash-based membership checks and preallocates the table to avoid rehash, removing repeated comparisons across the hot loop.",
  "confidence": "high"
}
```

**B. Concurrency & Parallelism**

```json
{
  "classification": "Concurrency & Parallelism",
  "secondary_tags": ["parallelism", "lock contention"],
  "mechanism_signals": ["introduced thread pool", "tile-based work partitioning", "narrowed mutex scope"],
  "affected_components": ["decoder/pipeline.cc", "decoder/tiler.cc"],
  "explanation": "Work is partitioned by tile across a shared pool and critical sections are reduced to short regions, lowering contention and enabling parallel execution of the same algorithm.",
  "confidence": "high"
}
```

**C. Unknown**

```json
{
  "classification": "Unknown / Not Enough Information",
  "secondary_tags": [],
  "mechanism_signals": ["broad refactor", "no evident hot-path edits"],
  "affected_components": ["loader/*"],
  "explanation": "Large refactor touches many files without showing hot-path changes or recognizable performance mechanisms, so the cause of any improvement cannot be determined from the diff alone.",
  "confidence": "low"
}
```
"""

USER_PROMPT = """\
### Commit Diff

```diff
{diff}
```

### Affected Workload

```
{workload}
```
"""

import multiprocessing
import time

import datasets
from litellm import completion

ds = datasets.load_dataset("swefficiency/swefficiency", split="test")
predictions_file = "predictions/converted/oh_claude37sonnet.jsonl"

predictions = {}
for line in open(predictions_file):
    import json

    obj = json.loads(line)
    predictions[obj["instance_id"]] = obj


def worker(instance):
    if instance["instance_id"] not in predictions:
        return {
            "classification": "Unknown / Not Enough Information",
            "confidence": "low",
            "instance_id": instance["instance_id"],
            "repo": instance["repo"],
        }

    diff = instance["patch"]
    workload = instance["workload"]

    prompt = USER_PROMPT.format(diff=diff, workload=workload)

    while True:
        try:
            response = completion(
                model="gemini/gemini-2.5-flash",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": USER_PROMPT.format(diff=diff, workload=workload),
                    },
                    {
                        "role": "user",
                        "content": "Think step-by-step about the code changes and their performance implications, then output the JSON object as specified.",
                    },
                ],
                temperature=0.0,
            )
            text = response.choices[0].message.content

            # Extract out the JSON object from the response
            import json
            import re

            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
            else:
                result = {
                    "classification": "Unknown / Not Enough Information",
                    "confidence": "low",
                }

            # Add instance ID for traceability
            result["instance_id"] = instance["instance_id"]
            result["repo"] = instance["repo"]

            return result
        except Exception as e:
            print(f"Error processing instance {instance['instance_id']}: {e}")
            time.sleep(5)
            continue


import tqdm

with multiprocessing.Pool(processes=4) as pool:
    results = []
    for r in tqdm.tqdm(pool.imap(worker, ds), total=len(ds), desc="Classifying diffs"):
        results.append(r)

# Save results
import json
from pathlib import Path

output_dir = Path("analysis/llm/outputs")

with open(output_dir / "diff_classification_results_pred.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
