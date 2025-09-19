import collections
import json
import multiprocessing
from pathlib import Path

import datasets
import tqdm

from swefficiency.harness.log_parsers import MAP_REPO_TO_PARSER

# Take old dataset and check: some errors:
# dataset = datasets.load_dataset("swefficiency/swefficiency", split='test', revision="718e5821f73f86414fa72bf8b7f716651a3a835a")
# ground_truth_data = Path("logs/run_evaluation/perf_eval_ground_truth/gold")


# dataset = datasets.load_dataset("swefficiency/swefficiency", split='test')

new_dataset = []

# Try 2: verify any flaky tests?
dataset = datasets.load_dataset(
    "swefficiency/swefficiency",
    split="test",
    revision="cf49f3526b31f6b54ada55d9ab3d3d08ddab67d4",
)

# latest? 39f0c020f75114c7e46d439bcc8a1d315f9b81ed

log_dir = Path("logs/run_evaluation/")
ground_truth_data_dirs = [
    log_dir / "jeff_perf_ground_truth/gold",
    log_dir / "jeff_perf_ground_truth2/gold",
    log_dir / "jeff_perf_ground_truth3/gold",
    log_dir / "jeff_perf_ground_truth4/gold",
    log_dir / "jeff_perf_ground_truth5/gold",
]


def parse_perf_summary(perf_summary):
    perf_lines = perf_summary.strip().splitlines()

    before_mean = float(perf_lines[0].split(":")[1].strip())
    before_std = float(perf_lines[1].split(":")[1].strip())
    after_mean = float(perf_lines[2].split(":")[1].strip())
    after_std = float(perf_lines[3].split(":")[1].strip())
    improvement = (after_mean - before_mean) / before_mean * 100

    return {
        "before_mean": before_mean,
        "after_mean": after_mean,
        "before_std": before_std,
        "after_std": after_std,
        "improvement": improvement,
    }


flaky_tests_across_all = set()


def worker(d):
    instance_id = d["instance_id"]

    owner = instance_id.split("__")[0]
    repo_name, remainder = instance_id.split("__")[1].rsplit("-", 1)

    d_dict = dict(d)

    correctness_report = {}

    good_perf = True

    flaky_tests = collections.defaultdict(list)

    print("==========")
    print(instance_id)

    if any(not (run_dir / instance_id).exists() for run_dir in ground_truth_data_dirs):
        return None

    for ground_truth_data in ground_truth_data_dirs:
        covering_test_status_json = (
            ground_truth_data / instance_id / "covering_test_status.json"
        )
        perf_summary = ground_truth_data / instance_id / "perf_summary.txt"

        instance_correctness_report_raw = dict()
        reparsed_covering_test_status_file_dir = (
            ground_truth_data / instance_id / "raw_correctness_output"
        )
        for file in reparsed_covering_test_status_file_dir.glob("*.txt"):
            instance_correctness_report_raw.update(
                MAP_REPO_TO_PARSER[f"{owner}/{repo_name}"](file.read_text())
            )
        instance_correctness_report = instance_correctness_report_raw

        # instance_correctness_report = json.loads(covering_test_status_json.read_text()) if covering_test_status_json.exists() else None

        if instance_correctness_report is not None:
            for k, v in instance_correctness_report.items():
                flaky_tests[k].append(v)

        instance_perf_report = (
            perf_summary.read_text() if perf_summary.exists() else None
        )
        if instance_perf_report is not None:
            perf_output = parse_perf_summary(instance_perf_report)

            delta = perf_output["after_mean"] - perf_output["before_mean"]
            max_std = max(perf_output["before_std"], perf_output["after_std"])

            if delta >= 0 or abs(delta) < max_std:
                good_perf = False
                break

    # Keep correctness report only if all in lists are the same
    real_flaky_tests = {k: v for k, v in flaky_tests.items() if len(set(v)) > 1}
    non_flaky_tests = {k: v for k, v in flaky_tests.items() if len(set(v)) == 1}

    for k, v in real_flaky_tests.items():
        print(k, flaky_tests[k])

    if not good_perf:
        return None

    d_dict["PASS_TO_PASS"] = [
        test for test, status in non_flaky_tests.items() if status == "PASSED"
    ]
    d_dict["image_name"] = f"ghcr.io/swefficiency/swefficiency-images:{instance_id}"

    d_dict["single_thread_tests"] = list(
        set(
            [
                "/testbed/" + k.split("::")[0]
                for k, v in real_flaky_tests.items()
                if any("PASS" in status for status in v)
            ]
        )
    )

    return d_dict


with multiprocessing.Pool() as pool:
    results = list(tqdm.tqdm(pool.imap(worker, dataset), total=len(dataset)))

new_dataset = [result for result in results if result is not None]

print(f"Total instances with good performance: {len(new_dataset)}")
print(
    f"Instances with flaky tests:",
    " ".join(
        d["instance_id"]
        for d in new_dataset
        if "single_thread_tests" in d and d["single_thread_tests"]
    ),
)

new_dataset = datasets.Dataset.from_list(new_dataset)

# Upload the new dataset
new_dataset.push_to_hub("swefficiency/swefficiency", split="test")
