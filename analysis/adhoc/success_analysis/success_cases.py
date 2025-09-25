from pathlib import Path

import pandas as pd

PREDICTION_DIR = Path("predictions/converted")

# EVAL_REPORT = "eval_reports/eval_report_gpt-5-mini_maxiter_100_N_v0.51.1-no-hint-run_1.csv"
# PRED_FILE = "oh_gpt5mini.jsonl"

# EVAL_REPORT = "eval_reports/eval_report_us.anthropic.claude-3-7-sonnet-20250219-v1_0_maxiter_100_N_v0.51.1-no-hint-run_1.csv"
# PRED_FILE = "oh_claude37sonnet.jsonl"

# EVAL_REPORT = "eval_reports/eval_report_gold.csv"
# PRED_FILE = None

EVAL_REPORT = (
    "eval_reports/eval_report_gemini-2.5-flash_maxiter_100_N_v0.51.1-no-hint-run_1.csv"
)
PRED_FILE = "oh_gemini25flash.jsonl"

# import pandas

df = pd.read_csv(EVAL_REPORT)

# Sort by human_speedup_ratio.
df = df.sort_values(by="human_speedup_ratio", ascending=False)

# Filter out instances where correctness is not 1.
df = df[df["correctness"] == 1.0]

# Keep human_speedup_ratio and instance_id columns.
df = df[
    ["instance_id", "human_speedup_ratio", "pred_speedup_ratio", "gold_speedup_ratio"]
]
predictions = {}
gold_predictions = {}
if PRED_FILE is not None:
    import json

    with open(PREDICTION_DIR / PRED_FILE, "r") as f:
        for line in f:
            pred = json.loads(line)
            predictions[pred["instance_id"]] = pred["model_patch"]

    # Add a column for the prediction.
    df["patch"] = df["instance_id"].apply(lambda x: predictions.get(x, ""))

import datasets

ds = datasets.load_dataset("swefficiency/swefficiency", split="test")

for item in ds:
    gold_predictions[item["instance_id"]] = item["patch"]


for row in df.itertuples():

    num_gold_lines = len(gold_predictions.get(row.instance_id, "").splitlines())
    num_pred_lines = len(row.patch.splitlines())

    if num_gold_lines > 40 or num_pred_lines > 40:
        continue

    print(f"Instance ID: {row.instance_id}")
    print(
        f"Speedup: {row.human_speedup_ratio} ({row.pred_speedup_ratio:.4f}/{row.gold_speedup_ratio:.4f}s)"
    )
    print("Patch:", len(row.patch.splitlines()), "lines")
    print(row.patch)
    print("-" * 80)
    print(
        "Gold Patch:",
        len(gold_predictions.get(row.instance_id, "").splitlines()),
        "lines",
    )
    print(gold_predictions.get(row.instance_id, "N/A"))
    print("=" * 80)
    print()
