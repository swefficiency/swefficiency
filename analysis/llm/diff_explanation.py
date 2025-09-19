

SYSTEM_PROMPT = """\
You are an expert software engineer specializing in performance analysis and optimization. Given the following inputs:

You will receive:
1) A unified git patch (diff) that changes a codebase.
2) A Python workload script that shows improved performance when run against the patched codebase (relative to the baseline).

Task: Explain, in concrete technical terms, how the code edit makes THIS workload faster. Base your reasoning ONLY on evidence you can extract from the patch and workload script (plus any provided notes). Do static analysis; do not assume you can run code.

--- INPUT CONTRACT ---
- GIT_PATCH: a unified diff (may span multiple files/languages).
- WORKLOAD_SCRIPT: a Python script that imports/executes parts of the codebase.
- METRICS (optional): free-form notes with benchmark deltas, environment hints, or config changes.

--- WHAT TO DO (internal method; do NOT output these steps) ---
1) Map workload → code paths
   • Parse WORKLOAD_SCRIPT to identify imported modules, symbols, entry points, CLI invocations, and data paths that it triggers.
   • From GIT_PATCH, locate touched files/functions/classes/config that are reachable by the workload’s call graph or binaries it invokes.

2) Identify optimization mechanisms in the patch
   Consider common perf levers (use exact terms if present in the diff):
   • Algorithm/complexity: e.g., O(n^2) → O(n log n), early-exit, pruning, memoization.
   • Data structures/layout: contiguous arrays, hash maps, tries, arenas, SoA vs AoS, smaller key types.
   • Concurrency/parallelism: thread pools, async I/O, vectorization/SIMD, GPU/accelerator paths.
   • Memory & allocation: fewer allocations, reuse buffers, pooling, alignment, prefetch, cache locality.
   • I/O & serialization: batching, streaming, compression, zero-copy, mmap, fewer syscalls, less logging.
   • Compiler/build/config: -O3/LTO/PGO, flags/macros, Cython/Numba/JIT on hot paths, disabling assertions, release builds.
   • Numerical/precision tradeoffs: fp64→fp32/bf16, tolerance tweaks.
   • API/protocol changes: fewer round-trips, larger batch size, vectorized NumPy/PyTorch ops, avoiding Python loops.
   • Locking/coordination: finer-grained locks, lock-free structures, reduced contention, async queues.
   • Caching: result caches, memoization, disk cache, warm-up strategies, reuse of plans (FFT/GEMM).
   • Dead work removal: eliminate redundant parsing/validation/copying, hoist invariants out of loops.
   • Configuration defaults: env vars, thread counts, block sizes, jemalloc/tcmalloc switches.

3) Connect cause → effect for THIS workload
   • Show how the workload triggers the changed code paths.
   • Explain why the new behavior reduces CPU cycles, memory traffic, I/O, synchronization, or interpreter overhead in this context.
   • If METRICS are provided, tie the mechanism to the reported deltas (e.g., “fewer allocations” → lower GC time).
   • Note any tradeoffs or changed semantics that might impact correctness or generality.

4) Extract “key signals”
   • Include succinct keywords/phrases that were decisive in your reasoning: function/class names, file paths, flags/macros, algorithm names, API calls, library ops, env vars, config keys, and performance concepts observed in the patch and workload.

5) Assess confidence (high/medium/low)
   Use this rubric:
   • HIGH: The workload clearly exercises the changed code; the patch shows a canonical perf pattern; names/paths match; (optional) metrics corroborate.
   • MEDIUM: Reasonable mapping but some gaps or multiple plausible mechanisms.
   • LOW: Cannot map changes to workload paths; changes look unrelated (e.g., tests/docs only) or evidence is weak/ambiguous.

Please output a detailed explanation of your reasoning process, citing specific lines/files/patterns from the diff and workload script.
"""

USER_PROMPT = """\
--- INPUTS START ---
GIT_PATCH:
```diff
{diff}
```

WORKLOAD_SCRIPT:
```python
{workload}
```
--- INPUTS END ---"""

import pandas as pd
import json

def extract_json_objects(s: str):
    dec = json.JSONDecoder()
    i, n = 0, len(s)
    out = []
    while i < n:
        try:
            i = s.index('{', i)           # next candidate
            obj, end = dec.raw_decode(s, i)
            if isinstance(obj, dict):     # only keep objects (not arrays, numbers, etc.)
                out.append(s[i:end])
            i = end
        except ValueError:
            break                          # no more '{'
        except json.JSONDecodeError:
            i += 1                         # not valid here; move one char and keep scanning
    return out



import multiprocessing
import time
import datasets
from litellm import completion

ds = datasets.load_dataset("swefficiency/swefficiency", split="test")

sonnet37_openhands_predictions = pd.read_json("predictions/converted/oh_claude37sonnet.jsonl", lines=True)
gpt5_mini_openhands_predictions = pd.read_json("predictions/converted/oh_gpt5mini.jsonl", lines=True)
gemini25_flash_openhands_predictions = pd.read_json("predictions/converted/oh_gemini25flash.jsonl", lines=True)

MAP_PATCH_TYPE = {
    "gold": {d["instance_id"]: d["patch"] for d in ds},
    "sonnet37_openhands": {d["instance_id"]: d["model_patch"] for d in sonnet37_openhands_predictions.to_dict(orient="records")},
    "gpt5mini_openhands": {d["instance_id"]: d["model_patch"] for d in gpt5_mini_openhands_predictions.to_dict(orient="records")},
    "gemini25_openhands": {d["instance_id"]: d["model_patch"] for d in gemini25_flash_openhands_predictions.to_dict(orient="records")},
}

TYPE = "gold"  # or "sonnet37_openhands"
TYPE = "sonnet37_openhands"
TYPE = "gpt5mini_openhands"
TYPE = "gemini25_openhands"

import regex as re

PATTERN = re.compile(
    r'\{(?=\s*")'                                 # first non-space after { is a quote (JSON key)
    r'(?:[^{}"]+|"[^"\\]*(?:\\.[^"\\]*)*"|(?R))*' # text, strings, or nested {...}
    r'\}',
    re.DOTALL
)


def worker(instance):
    diff = MAP_PATCH_TYPE[TYPE].get(instance["instance_id"], None)    
    workload = instance["workload"]
    
    prompt = USER_PROMPT.format(diff=diff, workload=workload)

    for _ in range(5):
        try:
            response = completion(
                model="gemini/gemini-2.5-flash", 
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT.format(
                        diff=diff,
                        workload=workload
                    )},
                ],
                temperature=0.0,
            )
            text = response.choices[0].message.content
            
            # Extract out the JSON object from the response
            result = {}
            result["explanation"] = text

            # Add instance ID for traceability
            result["instance_id"] = instance["instance_id"]
            result["repo"] = instance["repo"]

            return result
        except Exception as e:
            print(f"Error processing instance {instance['instance_id']}: {e}")
            time.sleep(5)
            continue
        
    return {
        "explanation": None,
        "instance_id": instance["instance_id"],
        "repo": instance["repo"]
    }
         
import tqdm   
 
with multiprocessing.Pool(processes=8) as pool:
    results = []
    for r in tqdm.tqdm(pool.imap(worker, ds), total=len(ds), desc="Explaining diffs"):
        results.append(r)  
    
# Save results
import json

from pathlib import Path

output_dir = Path("analysis/llm/outputs")

with open(output_dir / f"diff_explanation_{TYPE}.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

