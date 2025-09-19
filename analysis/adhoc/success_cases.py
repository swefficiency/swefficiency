#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset

from swefficiency.harness.run_validation import parse_perf_summary

# Hardcoded path
GOLD_JSONL_PATH = "nalysis/llm/outputs/diff_explanation.jsonl"
JSONL_PATH = "analysis/llm/outputs/diff_explanation_sonnet37_openhands.jsonl"

GOLD_RESULTS_DIR = Path("logs/run_evaluation/ground_truth5/gold")
PRED_RESULTS_DIR = Path(
    "logs/run_evaluation/ground_truth5/us.anthropic.claude-3-7-sonnet-20250219-v1_0_maxiter_100_N_v0.51.1-no-hint-run_1"
)

GOLD_SPEEDUP_RESULTS = "eval_reports2/eval_report_gold.csv"
PRED_SPEEDUP_RESULTS = "eval_reports2/eval_report_us.anthropic.claude-3-7-sonnet-20250219-v1_0_maxiter_100_N_v0.51.1-no-hint-run_1.csv"

gold_speedup_results = pd.read_csv(GOLD_SPEEDUP_RESULTS)
pred_speedup_results = pd.read_csv(PRED_SPEEDUP_RESULTS)


def read_jsonl(path):
    """Yield parsed JSON objects from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# Print instance ids with the largest human speedup ratio.
pred_speedup_results = pred_speedup_results.sort_values(
    by="human_speedup_ratio", ascending=False
)
print("Top 40 instances by human speedup ratio:")
print(pred_speedup_results[["instance_id", "human_speedup_ratio"]].head(40))

gold_results = list(read_jsonl(GOLD_JSONL_PATH))
gold_id_to_classification = {r["instance_id"]: r for r in gold_results}
pred_classification_results = list(read_jsonl(JSONL_PATH))
pred_id_to_classification = {r["instance_id"]: r for r in pred_classification_results}

for _, pred_speedup_results in list(pred_speedup_results.iterrows())[:20]:
    patch_file = PRED_RESULTS_DIR / pred_speedup_results["instance_id"] / "patch.diff"
    # print(pred_speedup_results["human_speedup_ratio"], str(patch_file), pred_id_to_classification[pred_speedup_results["instance_id"]]["classification"])

    print("================================")
    print(
        f"Instance ID: {pred_speedup_results['instance_id']}, LM Speedup Ratio (normalized to expert performance): {pred_speedup_results['human_speedup_ratio']}"
    )
    print("--------------------------------")
    print("Explanation of LM Generated Diff:")
    print(pred_id_to_classification[pred_speedup_results["instance_id"]]["explanation"])
    print("--------------------------------")
    print("Explanation of Expert Diff:")
    print(gold_id_to_classification[pred_speedup_results["instance_id"]]["explanation"])
    print("================================")
