#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from datasets import load_dataset

from swefficiency.harness.run_validation import parse_perf_summary

# Hardcoded path
GOLD_JSONL_PATH = "analysis/llm/outputs/diff_classification_results.jsonl"
JSONL_PATH = "analysis/llm/outputs/diff_classification_results_pred.jsonl"

GOLD_RESULTS_DIR = Path("logs/run_evaluation/ground_truth5/gold")
PRED_RESULTS_DIR = Path(
    "logs/run_evaluation/ground_truth5/us.anthropic.claude-3-7-sonnet-20250219-v1_0_maxiter_100_N_v0.51.1-no-hint-run_1"
)

GOLD_SPEEDUP_RESULTS = "eval_reports2/eval_report_gold.csv"
PRED_SPEEDUP_RESULTS = "eval_reports2/eval_report_us.anthropic.claude-3-7-sonnet-20250219-v1_0_maxiter_100_N_v0.51.1-no-hint-run_1.csv"


def read_jsonl(path):
    """Yield parsed JSON objects from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def disagreeement(gold, pred):
    if gold["classification"] != pred["classification"]:
        return True

    return False


import pandas as pd

gold_results = list(read_jsonl(GOLD_JSONL_PATH))
pred_results = list(read_jsonl(JSONL_PATH))
print(f"Gold: {len(gold_results)}, Pred: {len(pred_results)}")

gold_speedup_results = pd.read_csv(GOLD_SPEEDUP_RESULTS)
pred_speedup_results = pd.read_csv(PRED_SPEEDUP_RESULTS)

# Get easy map from instance ID to speedup
gold_id_to_speedup = {
    row["instance_id"]: row for _, row in gold_speedup_results.iterrows()
}
pred_id_to_speedup = {
    row["instance_id"]: row for _, row in pred_speedup_results.iterrows()
}

ds = load_dataset("swefficiency/swefficiency", split="test")
id_to_instance = {inst["instance_id"]: inst for inst in ds}

counter = []
correct_counter = []
print(f"Examples of disagreements:")
for g, p in zip(gold_results, pred_results):
    pred_speedup = pred_id_to_speedup.get(p["instance_id"], None)

    if pred_speedup["correctness"] != 1.0:
        continue

    correct_counter.append((g["classification"], p["classification"]))

    if disagreeement(g, p):
        print(f"Instance ID: {g['instance_id']}")
        print(f"Gold: {g['classification']}, Pred: {p['classification']}")
        gold_instance = id_to_instance[g["instance_id"]]

        gold_patch = gold_instance["patch"]
        gold_speedup = gold_id_to_speedup.get(g["instance_id"], None)

        pred_patch = ""
        if (PRED_RESULTS_DIR / g["instance_id"] / "patch.diff").exists():
            pred_patch = (
                PRED_RESULTS_DIR / g["instance_id"] / "patch.diff"
            ).read_text()

        print(
            f"Gold Instance: {len(gold_patch.splitlines())}",
            gold_speedup["gold_speedup_ratio"] if gold_speedup is not None else "N/A",
        )
        print(
            f"Pred Patch: {len(pred_patch.splitlines())}",
            pred_speedup["pred_speedup_ratio"] if pred_speedup is not None else "N/A",
        )
        print(
            f"Correctness: {pred_speedup['correctness'] if pred_speedup is not None else 'N/A'}"
        )

        print("-----")

        counter.append((g["instance_id"], g["classification"], p["classification"]))

print(f"Total disagreements: {len(counter)}")
print(f"Total correct: {len(correct_counter)}")


def main():
    field = "classification"  # the field to count
    counter = Counter()

    for obj in read_jsonl(JSONL_PATH):
        if field in obj and obj[field] is not None:
            counter[str(obj[field])] += 1

    if not counter:
        print("No entries found to count.")
        return

    # Sort by count (descending)
    items = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
    labels, counts = zip(*items)

    # Print counts to stdout
    print("\nCounts:")
    width = max(len(s) for s in labels)
    for lab, c in items:
        print(f"{lab:<{width}}  {c}")

    # Plot horizontal bar chart
    plt.figure(figsize=(10, max(4, 0.4 * len(labels))))
    positions = range(len(labels))
    bars = plt.barh(positions, counts, color="purple", alpha=0.7)
    plt.yticks(positions, labels, fontsize=14)
    plt.xlabel("# of Instances", fontsize=14)
    # plt.title(f"Counts of '{field}'")
    plt.gca().invert_yaxis()  # largest at the top

    # Extend x-axis so labels fit
    plt.xlim(0, max(counts) * 1.15)  # add 15% margin

    # Add numeric labels at the end of bars (floating)
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_width() + (0.01 * max(counts)),  # slight offset past the bar
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            ha="left",
            fontsize=14,
            fontweight="bold",
        )

    plt.tight_layout()

    # Save the plot
    plt.savefig("assets/figures/diff_classification_counts_pred.png")


if __name__ == "__main__":
    main()
