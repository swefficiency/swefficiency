import argparse
import json
import multiprocessing
from functools import partial
from pathlib import Path

import datasets
import pandas as pd
from tqdm import tqdm

from swefficiency.harness.log_parsers import MAP_REPO_TO_PARSER


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


def get_number_of_patch_modified_lines(git_patch_text: str) -> int:
    """Count the number of modified lines in a git patch text."""
    num_lines = 0
    for line in git_patch_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            num_lines += 1

        if line.startswith("-") and not line.startswith("---"):
            num_lines += 1
    return num_lines


def evaluate_instance(
    instance: dict, gold_run: Path, pred_run: Path, use_correctnes_files=True
) -> dict:
    instance_id = instance["instance_id"]
    pass_to_pass = instance["PASS_TO_PASS"]

    gold_run_entry = gold_run / instance_id / "perf_summary.txt"
    pred_run_entry = pred_run / instance_id / "perf_summary.txt"

    # Compute prediction speedup ratio.
    if pred_run_entry.exists():
        pred_perf_info = parse_perf_summary(pred_run_entry.read_text())
        pred_speedup_ratio = (
            pred_perf_info["before_mean"] / pred_perf_info["after_mean"]
        )
    else:
        pred_speedup_ratio = 1.0  # No speedup if no prediction run exists

    # Compute gold speedup ratio.
    if gold_run_entry.exists():
        gold_perf_info = parse_perf_summary(gold_run_entry.read_text())
        gold_speedup_ratio = (
            gold_perf_info["before_mean"] / gold_perf_info["after_mean"]
        )
    else:
        gold_speedup_ratio = 1.0

    # Check that pass to pass tests are still passing.
    pred_statuses = {}
    correctness_dir = pred_run / instance_id / "raw_correctness_output"

    num_modified_lines = get_number_of_patch_modified_lines(instance["patch"])

    if not correctness_dir.exists():
        # print(
        #     f"Instance {instance_id} has no correctness output directory.", flush=True
        # )
        pred_speedup_ratio = 1.0
        return {
            "instance_id": instance_id,
            "pred_speedup_ratio": 1.0,
            "gold_speedup_ratio": gold_speedup_ratio,
            "human_speedup_ratio": (
                1.0 / gold_speedup_ratio if gold_speedup_ratio != 0 else 0
            ),
            "correctness": 0.0,
            "correctness_pct": 0.0,
            "pre_edit_runtime": gold_perf_info["before_mean"],
            "patch_length": num_modified_lines,
        }

    if not use_correctnes_files:
        for test_output in correctness_dir.glob("*.txt"):
            test_ouput_text = test_output.read_text()
            pred_statuses.update(MAP_REPO_TO_PARSER[instance["repo"]](test_ouput_text))
    else:
        pred_statuses = json.loads(
            (pred_run / instance_id / "covering_test_status.json").read_text()
        )

    passed_tests = []
    failed_tests = []
    for test in pass_to_pass:
        if "PASS" in pred_statuses.get(test, ""):
            passed_tests.append(test)
        else:
            failed_tests.append(test)

    passed_tests = set(passed_tests)

    correctness_pct = len(passed_tests) / len(pass_to_pass) if pass_to_pass else 1.0
    if correctness_pct < 1.0:
        # print(
        #     f"Instance {instance_id} failed {len(pass_to_pass) - len(passed_tests)} out of {len(pass_to_pass)} pass-to-pass tests.",
        #     flush=True,
        # )
        # for test in pass_to_pass:
        #     if test not in passed_tests:
        #         print(f"  Failed test: {test}")
        # print("==========================", flush=True)
        pass

    adjusted_pred_speedup_ratio = 1.0 if correctness_pct != 1.0 else pred_speedup_ratio

    return {
        "instance_id": instance_id,
        "raw_pred_speedup_ratio": pred_speedup_ratio,
        "pred_speedup_ratio": adjusted_pred_speedup_ratio,
        "gold_speedup_ratio": gold_speedup_ratio,
        "human_speedup_ratio": (
            adjusted_pred_speedup_ratio / gold_speedup_ratio
            if gold_speedup_ratio != 0
            else 0
        ),
        "correctness": 0.0 if correctness_pct != 1.0 else 1.0,
        "correctness_pct": correctness_pct,
        "pre_edit_runtime": gold_perf_info["before_mean"],
        "patch_length": len(instance["patch"].splitlines()),
    }


def main(gold_run, pred_run, num_workers, output_dir):
    ds = datasets.load_dataset("swefficiency-anon/swefficiency", split="test")

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"eval_report_{pred_run.name}"

    worker = partial(evaluate_instance, gold_run=gold_run, pred_run=pred_run)
    with multiprocessing.Pool(num_workers) as pool:
        # Use imap for streaming (ordered) progress updates; switch to imap_unordered for speed if order doesn't matter
        results = list(
            tqdm(
                pool.imap(worker, ds, chunksize=1),
                total=len(ds),
                desc="Evaluating instances",
            )
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / f"{run_name}.csv", index=False)
    print(f"Evaluation report saved to {output_dir / f'{run_name}.csv'}")

    # Take harmonic mean of speedup ratios.
    harmonic_mean_human_speedup = (
        len(results_df) / (1 / results_df["human_speedup_ratio"]).sum()
    )
    base_speedup = len(results_df) / (1 / results_df["pred_speedup_ratio"]).sum()

    # Print correctness ratios.

    # Print human speedup ratios with instace ids.
    # print("Instance Human Speedup Ratios:")
    # counter = 0
    # for _, row in results_df.iterrows():
    #     if row["correctness"] < 1.0:
    #         print(f"  {row['instance_id']}: {row['human_speedup_ratio']}x (Correctness: {row['correctness']}, Correctness %: {row['correctness_pct']*100}%)")
    #         counter += 1

    # print(f"Total incorrect instances: {counter} out of {len(results_df)}")

    print(f"Average Human Speedup Ratio: {harmonic_mean_human_speedup}x")
    print(f"Correctness Percentage: {results_df['correctness'].mean() * 100}%")

    # # Print all instances ids that did not achieve correctness in one line space delimited format.
    # incorrect_instances = results_df[results_df["correctness"] < 1.0]
    # if not incorrect_instances.empty:
    #     print("Incorrect Instances:", " ".join(incorrect_instances["instance_id"].tolist()))
    # else:
    #     print("All instances passed correctness.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold_run", type=Path, required=True, help="Path to the ground truth run dir"
    )
    parser.add_argument(
        "--pred_run", type=Path, required=True, help="Path to the predicted run dir"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for parallel processing",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./eval_reports"),
        help="Output directory for the report",
    )

    args = parser.parse_args()
    main(**vars(args))
