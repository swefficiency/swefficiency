import json
import os
import random
import re
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from litellm import completion
from tqdm import tqdm

from swefficiency.harness.constants import SWEfficiencyInstance
from swefficiency.harness.utils import load_swefficiency_dataset

WORKLOAD_GENERATION_DIR = Path("logs/workload_generation")

SYSTEM_MSG = """You are a performance testing expert. You will be provided a code edit as a git diff and the pre-edit source files. You need to generate a **self-contained Python performance workload script** that measures perfomance of code paths or APIs changed in the diff.

Guidelines for the workload script contents.

- Use a `setup()` function to prepare any realistic, non-trivial data or environment needed for the test.
  - Data must be representative of real-world usage (avoid trivial arrays or easily optimizable patterns).
  - Prefer real datasets or realistic synthetic data with reproducibility (set a random seed).
  - All expensive or one-time setup (e.g., file download, preprocessing) must be in `setup()`, not in the workload.

- Use a `workload()` function to run the actual operation(s) being timed.
  - The workload should reflect a **representative and challenging real-world use case** of the API or library under test.
  - Avoid corner cases that could be trivially optimized.
  - Inputs should be varied enough to prevent caching or constant-folding from affecting results.

- Run the benchmark using `timeit.repeat(workload, number=..., repeat=..., setup=setup)`.
  - `number` should match a realistic single-run execution count (do not batch multiple runs for cumulative timing).
  - `repeat` should be high enough to gather stable statistics.

- Print the mean and standard deviation of the last set of runtimes using `statistics.mean()` and `statistics.stdev()`.
  - Output should be clear and ready for performance comparison.

- The output must be a **complete Python script** containing only:
  1. import statements
  2. `setup()` function
  3. `workload()` function
  4. the `timeit.repeat()` call
  5. mean/stddev printing

The script should only print two lines at the end: the mean of measured runtimes and the standard deviation of runtimes.

Example workload to follow (please strictly follow this format of imports, setup function, workload function, timeit call, and print statements). In particular, make sure the mean and standard deviation print statements are exactly as shown below.

```python
import timeit
import statistics
import numpy as np

def setup():
    global arr
    np.random.seed(42)
    arr = np.random.rand(5000, 5000)

def workload():
    global arr
    _ = arr @ arr.T

runtimes = timeit.repeat(workload, number=1, repeat=10, setup=setup)

print("Mean:", statistics.mean(runtimes))
print("Std Dev:", statistics.stdev(runtimes))
```
"""


CONTEXT_MSG = """Here's a commit and it's information that does some optimization in the {repo_name} repository that might be relevant to writing the test:
## Commit Diff:
```
{commit_diff}
```

## Pre-edit source files:
{pre_edit_code}
"""


def extract_code_block(text):
    if text is None:
        return None
    match = re.search(r"```(?:[^\n]*)\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def worker_function(
    datum: SWEfficiencyInstance,
    run_id: str,
):
    output_file = WORKLOAD_GENERATION_DIR / run_id / f"{datum['instance_id']}.py"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Get relevant files from the patch.
    patch = datum["patch"]
    diff_pattern = r"diff --git a/.* b/(.*)"
    directives = re.findall(diff_pattern, patch)
    directives = [d for d in directives]

    owner, repo = datum["repo"].split("/")
    commit_hash = datum["base_commit"]

    file_contents = []

    for file_path in directives:
        max_retries = 3
        for attempt in range(max_retries):
            url = f"https://raw.githubusercontent.com/{owner}/{repo}/{commit_hash}/{file_path}"
            response = requests.get(url)

            if response.status_code == 200:
                file_contents.append(f"File: {file_path}")
                file_contents.append(f"```\n{response.text}\n```\n")
                break
            else:
                time.sleep(1)  # Wait before retrying
                print(f"Failed to fetch {file_path} from {url}, retrying...")
                continue

    # Combine all file contents into a single string
    commit_diff = patch.strip()
    all_preedit_file_contents = "\n".join(file_contents)

    while True:
        try:
            response = completion(
                model="gemini/gemini-2.5-flash",
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {
                        "role": "user",
                        "content": CONTEXT_MSG.format(
                            repo_name=repo,
                            commit_diff=commit_diff,
                            pre_edit_code=all_preedit_file_contents,
                        ),
                    },
                    {
                        "role": "user",
                        "content": "Can you write a workload in same style as the example?",
                    },
                ],
                temperature=0.0,
            )
            break
        except Exception as e:
            print(f"Error during completion: {e}")
            time.sleep(5)

    result = response.choices[0].message.content
    code_block_content = extract_code_block(result)

    with open(output_file, "w") as f:
        if code_block_content:
            f.write(code_block_content)

    return {
        "instance_id": datum["instance_id"],
        "run_id": run_id,
        "workload": code_block_content if code_block_content else result,
    }


def main(
    dataset_name: str,
    split: str,
    instance_ids: list[str],
    max_workers: int,
    run_id: str,
):
    dataset = load_swefficiency_dataset(dataset_name, split)
    random.shuffle(dataset)  # Shuffle dataset for randomness

    WORKLOAD_GENERATION_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = WORKLOAD_GENERATION_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "workload_generation.json"

    # Filter dataset by instance_ids if provided
    if instance_ids:
        dataset = [d for d in dataset if d["instance_id"] in instance_ids]

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(worker_function, datum, run_id): datum["instance_id"]
            for datum in dataset
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Generating workloads"
        ):
            result = future.result()
            results.append(result)

    # Save results to JSON lines.
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        default="swefficiency/swefficiency",
        type=str,
        help="Name of dataset or path to JSON file.",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Split of the dataset"
    )
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        help="Instance IDs to run (space separated)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="Maximum number of workers (should be <= 75%% of CPU cores)",
    )
    parser.add_argument(
        "--run_id", type=str, required=True, help="Run ID - identifies the run"
    )
    args = parser.parse_args()

    main(**vars(args))
