import codecs
import functools
import json
import os
from pathlib import Path

import gspread
import pandas as pd
import tqdm
from datasets import Dataset, DatasetDict, load_dataset
from gspread_dataframe import get_as_dataframe
from huggingface_hub import HfApi, HfFolder, Repository

from swefficiency.harness.constants import MAP_REPO_VERSION_TO_SPECS
from swefficiency.harness.log_parsers import MAP_REPO_TO_PARSER
from swefficiency.harness.test_spec import make_test_spec

gc = gspread.service_account()

instance_path = Path("artifacts/2_versioning")
data_path = Path("data/")
run_evaluation_path = Path("logs/run_evaluation")


@functools.cache
def get_instance_file(repo_name):
    """Get the instance file for a given repo name, reading all fields as raw strings."""
    instance_file = (
        instance_path / f"{repo_name}-task-instances_attribute_versions.non-empty.jsonl"
    )
    with open(instance_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Convert every value in every row to string exactly as-is
    for row in data:
        for k, v in row.items():
            row[k] = (
                json.dumps(v) if not isinstance(v, str) else v
            )  # preserve formatting

    df = pd.DataFrame(data)
    return df


def get_sweperf_data(annotate_spreadsheet_name="global_sweperf_all_data_annotate"):
    annotate_spreadsheet = gc.open(annotate_spreadsheet_name)
    annotate_worksheet = annotate_spreadsheet.get_worksheet(0)

    df = get_as_dataframe(
        annotate_worksheet,
        header=0,
        dtype=str,
    )

    # Ignore where "status" != "APPROVED".
    df = df[df["status"] == "APPROVED"]

    entries = []
    for i, annotate_row in tqdm.tqdm(
        df.iterrows(), total=len(df), desc="Processing SWE-Perf data"
    ):
        instance_id = annotate_row["instance_id"]

        owner = instance_id.split("__")[0]
        repo_name, remainder = instance_id.split("__")[1].rsplit("-", 1)
        pull_number = int(remainder)

        repo_df = get_instance_file(repo_name)
        instance_row = repo_df[repo_df["instance_id"] == instance_id].iloc[0]

        workload = annotate_row["workload"]
        instance_data_dir = data_path / instance_id
        covering_tests = (
            (instance_data_dir / "covering_tests.txt").read_text().splitlines()
            if (data_path / instance_id / "covering_tests.txt").exists()
            else []
        )

        # We probably need to use the perf runs.
        perf_data_run_correctness_path = (
            run_evaluation_path
            / f"perf_profiling_20250612_{repo_name}"
            / "gold"
            / instance_id
            / "raw_correctness_output"
        )
        # if perf_data_run_correctness_path.exists():
        #     perf_data_run_correctness = {}
        #     for raw_correctness_output in perf_data_run_correctness_path.glob("*.txt"):
        #         raw_correctness_output_text = raw_correctness_output.read_text().strip()

        #         test_status = MAP_REPO_TO_PARSER[instance_row['repo']](raw_correctness_output_text)
        #         perf_data_run_correctness.update(test_status)

        #     speedup = 0.0
        #     pass_to_pass = [
        #         test for test, status in perf_data_run_correctness.items() if status == "PASSED"
        #     ]
        # else:
        #     speedup = 0.0

        # This will get filled in later.
        pass_to_pass = []
        speedup = 0.0

        version = instance_row.get("version", None)
        repo = instance_row["repo"]

        specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
        test_cmd = specs["test_cmd"]
        rebuild_cmd = specs["install"]

        entry = {
            "repo": str(instance_row["repo"]),
            "instance_id": str(instance_row["instance_id"]),
            "base_commit": str(instance_row["base_commit"]),
            "patch": str(instance_row["patch"]),
            "test_patch": str(instance_row["test_patch"]),
            "problem_statement": str(instance_row["problem_statement"]),
            "hints_text": str(instance_row["hints_text"]),
            "created_at": str(instance_row["created_at"]),
            "version": str(instance_row["version"]),
            "PASS_TO_PASS": pass_to_pass,
            "FAIL_TO_PASS": [],
            "environment_setup_commit": instance_row.get(
                "environment_setup_commit", instance_row["base_commit"]
            ),  # Fallback to base_commit if not present
            # SWE-Perf specific fields
            "workload": workload,
            "speedup": speedup,
            "covering_tests": covering_tests,
            # Metadata fields
            "notes": str(annotate_row["notes"]),
            "test_cmd": str(test_cmd),
            "rebuild_cmd": codecs.decode(
                str(rebuild_cmd), "unicode_escape"
            ),  # Decode unicode escape sequences
            "image_name": f"ghcr.io/swefficiency/swefficiency:{str(instance_row['instance_id'])}",
        }
        entries.append(entry)

    # Deduplicate entries by instance_id and keep first occurrence
    unique_entries = {entry["instance_id"]: entry for entry in entries}
    entries = list(unique_entries.values())

    data = {
        "repo": [],
        "instance_id": [],
        "base_commit": [],
        "patch": [],
        "test_patch": [],
        "problem_statement": [],
        "hints_text": [],
        "created_at": [],
        "version": [],
        "PASS_TO_PASS": [],
        "FAIL_TO_PASS": [],
        "environment_setup_commit": [],
        "workload": [],
        "speedup": [],
        "covering_tests": [],
        "notes": [],
        "test_cmd": [],
        "rebuild_cmd": [],
        "image_name": [],
    }

    for entry in entries:
        for key in data.keys():
            data[key].append(entry[key])

    for key, value in data.items():
        print(
            f"{key}: {set([type(v) for v in value])}"
        )  # Print types of values for each key

    return data


print("Fetching SWE-Perf data...")
data = get_sweperf_data()
print("Data fetched successfully.")


# Create a Hugging Face Dataset
dataset = Dataset.from_dict(data)
dataset_dict = DatasetDict({"test": dataset})

# # Save locally
# dataset_dict.save_to_disk("my_dataset")

# Define repo info
repo_id = (
    "swefficiency-anon/swefficiency"  # Change to your username and desired dataset name
)

# Push to hub
dataset_dict.push_to_hub(repo_id)
print(f"Dataset uploaded to https://huggingface.co/datasets/{repo_id}")
