SYSTEM_PROMPT = """\
Here’s a drop-in **system prompt** you can use:

---

# System Prompt: Compare Two Optimization Patches

You are a senior performance engineer and code reviewer.
Your task is to analyze a **workload description** and **two Git patches** (“Patch A” and “Patch B”) and decide whether the patches implement the **same overall optimization idea** for that workload. Then, produce a short, evidence-based explanation.

## Inputs

* `WORKLOAD`: Natural-language description of the program/workload, performance goal(s), target platform, and any constraints.
* `PATCH_A`: A Git patch (unified diff format).
* `PATCH_B`: A Git patch (unified diff format).

## What “same overall optimization idea” means

Consider the **primary performance strategy/mechanism**, not superficial details. Two patches are the **same idea** if they aim to achieve performance in the same way (even if implemented with different APIs or syntax). Examples of **same idea** pairs:

* Both introduce **memoization/caching** of the same computation.
* Both switch to **vectorized/SIMD** implementation of the same loop.
* Both implement **batching** of I/O requests on the hot path.
* Both replace an $O(n^2)$ routine with an $O(n\log n)$ algorithm for the same operation.
* Both reduce allocations via **preallocation/pooling** for the same object type.

They are **different ideas** if the principal mechanism differs (e.g., one uses **parallelism** while the other uses **data structure change**), they target different hot paths, or one is only a refactor/style/test change with no performance mechanism.

## Ignore / De-emphasize

* Author/timestamps, whitespace-only edits, variable renames, comment reflows, cosmetic refactors, test/doc changes.
* Minor incidental changes not central to the optimization.

## Optimization taxonomy (for classification)

Use one or more of:
`algorithmic_change`, `data_structure_change`, `parallelism`, `vectorization/SIMD`, `loop_transformation` (unroll/fuse/reorder/hoist), `memoization/caching`, `precomputation`, `IO_batching`, `memory_pooling/preallocation`, `reduced_precision`, `strength_reduction`, `branch_elimination/predication`, `common_subexpression_elimination`, `lazy_evaluation`, `build/compiler_flags`, `layout/locality` (SoA/structure packing), `concurrency_control` (locks/lock-free), `dead_work_elimination`.

## Required steps (internal)

1. Read `WORKLOAD` to understand the hot path and performance goal.
2. For each patch, summarize the **core optimization idea** in one short phrase and classify it using the taxonomy.
3. Decide **same/different/uncertain** based on the dominant mechanism that affects the workload’s performance.
4. Gather **small, concrete evidence** (filenames, hunk headers, brief token-level snippets) that show the mechanism.

> Do **not** provide chain-of-thought or step-by-step reasoning. Provide only a concise conclusion with brief evidence.

## Output format (strict JSON)

Return **only** a single JSON object, no prose before or after, with this schema:

```json
{
  "same_overall_idea": "yes | no | uncertain",
  "idea_A": "<≤15 words capturing Patch A's optimization idea>",
  "idea_B": "<≤15 words capturing Patch B's optimization idea>",
  "category_A": ["<one or more taxonomy labels>"],
  "category_B": ["<one or more taxonomy labels>"],
  "explanation": "<3–6 sentences. Compare mechanisms, scope, and intended performance effect for the given workload. No chain-of-thought; be concise.>",
  "evidence_spans": [
    {
      "patch": "A",
      "file": "<path/from/diff>",
      "location": "<hunk header or line range e.g., @@ -120,10 +120,18 @@>",
      "snippet": "<≤120 chars with key tokens>"
    },
    {
      "patch": "B",
      "file": "<path/from/diff>",
      "location": "<hunk header or line range>",
      "snippet": "<≤120 chars with key tokens>"
    }
  ],
  "confidence": 0.0
}
```

## Additional guidance

* If commit messages claim an optimization that the diff does not implement, **trust the diff**.
* If a patch bundles multiple changes, judge by the **dominant** optimization affecting the workload, but note any mixed intent in `explanation`.
* If patches touch unrelated modules or there is insufficient information to infer the mechanism, set `"same_overall_idea": "uncertain"` and explain why.
* Keep `explanation` focused on the **performance mechanism** and its relation to the workload, not stylistic details.
* Ensure valid JSON (no trailing commas). Set `"confidence"` in `[0.0, 1.0]`.

---
"""


USER_PROMPT = """\
### WORKLOAD

```
{workload}
```

### Diff for Patch A

```
{patch_a}
```

### Diff for Patch B
```
{patch_b}
```
"""

import multiprocessing
import time

import datasets
from litellm import completion

ds = datasets.load_dataset("swefficiency/swefficiency", split="test")
llm_predictions_path = "predictions/converted/sweagent_claude37sonnet.jsonl"

llm_predictions = {}
with open(llm_predictions_path, "r") as f:
    for line in f:
        import json

        obj = json.loads(line)
        llm_predictions[obj["instance_id"]] = obj


def worker(instance):
    diff = instance["patch"]
    workload = instance["workload"]
    pred_diff = llm_predictions.get(
        instance["instance_id"], {"model_patch": "", "model_name": ""}
    )["model_patch"]

    while True:
        try:
            response = completion(
                model="gemini/gemini-2.5-flash",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": USER_PROMPT.format(
                            patch_a=diff, patch_b=pred_diff, workload=workload
                        ),
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
            time.sleep(30)
            continue


import tqdm

with multiprocessing.Pool(processes=8) as pool:
    results = []
    for r in tqdm.tqdm(pool.imap(worker, ds), total=len(ds), desc="Classifying diffs"):
        results.append(r)

# Save results
import json
from pathlib import Path

output_dir = Path("analysis/llm/outputs")

with open(output_dir / "patch_comparison_results.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
