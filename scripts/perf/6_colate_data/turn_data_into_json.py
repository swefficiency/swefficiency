import argparse
import json
from pathlib import Path
import pandas as pd

from swefficiency.harness.grading import get_eval_tests_report

parser = argparse.ArgumentParser(description="Process some data.")
parser.add_argument("--repos", type=str, required=True, help="Repo to process")

args = parser.parse_args()

repos = args.repos.split(",")

base_correctness_dir = Path("logs/run_evaluation/perf_coverage_correctness")
correctness_empty_dir = base_correctness_dir / "empty"
correctness_gold_dir = base_correctness_dir / "gold"

versioning_dir = Path("artifacts/2_versioning")

labelled_data_dir = Path("data")

for repo in repos:
    version_instance = (
        versioning_dir / f"{repo}-task-instances_attribute_versions.non-empty.jsonl"
    )

    repo_instances = pd.read_json(version_instance, lines=True)

    # List all folders in gold dir and mak
    instances_ids = [
        x.name for x in correctness_gold_dir.iterdir() if x.is_dir() and repo in x.name
    ]

    selected_repo_instances = repo_instances[
        repo_instances["instance_id"].isin(instances_ids)
    ]

    def add_correctness_columns(row):
        instance_id = row["instance_id"]

        pre_edit_correctness = (
            correctness_empty_dir / instance_id / "subtest_status.json"
        )
        post_edit_correctness = (
            correctness_gold_dir / instance_id / "subtest_status.json"
        )

        pre_edit_results = json.loads(pre_edit_correctness.read_text())
        post_edit_results = json.loads(post_edit_correctness.read_text())

        per_test_report = {}

        print(pre_edit_correctness)
        print(post_edit_correctness)

        for test_file in pre_edit_results:
            # Compute PASS_TO_PASS, PASS_TO_FAIL, FAIL_TO_PASS, and FAIL_TO_FAIL.

            p2p, p2f, f2p, f2f = [], [], [], []

            for subtest in pre_edit_results[test_file]:
                # Assume subtest is in both.
                pre_edit_subtest_result = pre_edit_results[test_file][subtest]

                if (
                    subtest not in post_edit_results[test_file]
                    and "SKIP" in pre_edit_subtest_result
                ):
                    continue

                post_edit_subtest_result = post_edit_results[test_file][subtest]

                if (
                    "PASS" in pre_edit_subtest_result
                    and "PASS" in post_edit_subtest_result
                ):
                    p2p.append(subtest)
                elif (
                    "PASS" in pre_edit_subtest_result
                    and "FAIL" in post_edit_subtest_result
                ):
                    p2f.append(subtest)
                elif (
                    "FAIL" in pre_edit_subtest_result
                    and "PASS" in post_edit_subtest_result
                ):
                    f2p.append(subtest)
                elif (
                    "FAIL" in pre_edit_subtest_result
                    and "FAIL" in post_edit_subtest_result
                ):
                    f2f.append(subtest)

            per_test_report[test_file] = {
                "PASS_TO_PASS": p2p,
                "PASS_TO_FAIL": p2f,
                "FAIL_TO_PASS": f2p,
                "FAIL_TO_FAIL": f2f,
            }

        row["test_report"] = per_test_report
        return row

    def add_workload(row):
        instance_id = row["instance_id"]

        workload_file = labelled_data_dir / instance_id / "workload.py"
        row["workload"] = workload_file.read_text()

        return row

    selected_repo_instances = selected_repo_instances.apply(
        add_correctness_columns, axis=1
    )
    selected_repo_instances = selected_repo_instances.apply(add_workload, axis=1)

output_dir = Path("harness_data")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "initial_data2.jsonl"

selected_repo_instances.to_json(output_file, orient="records", lines=True)
