# Evaluation Runner

This directory contains scripts to evaluate model predictions.

## Prerequisites
- Bash, Python deps already installed (if any evaluation scripts require them).
- Your prediction files placed under: `predictions/`
    - Organize as you prefer (e.g., `predictions/<run_id>/predictions.jsonl`), or follow the format required by the scorer.
    - Ensure file names and structure match what the evaluation script expects.

## Steps

1. Generate model predictions.
2. Save them into the `predictions/` directory.
3. Choose a run identifier:
     - `run_id`: arbitrary label (no spaces). Used for logging/output separation.
4. Decide parallelism:
     - `num_worker`: integer > 0 controlling concurrency.

## Run

From repository root:
```
scripts/eval/run_eval.sh <run_id> <num_worker> <path_to_model_predictions>
```

Example:
```
scripts/eval/run_eval.sh gpt4_run 12 gpt4_predictions.jsonl
```

## Outputs
- Metrics and logs will be written under an evaluation output directory (`logs/run_evaluation/`).
- Inspect logs if failures occur (missing prediction file, bad format, etc.).

## Tips
- Dry run with a tiny prediction file first.
- Keep `run_id` unique per experiment to avoid overwriting.
- If resource constrained, lower `num_worker`.

## Troubleshooting
- Permission error: `chmod +x scripts/eval/run_eval.sh`
- Wrong Python env: activate the intended virtual environment before running.
- Format mismatch: open the scoring script to confirm required prediction schema.

# Generating an Eval Report
After evaluation jobs finish you can generate a consolidated report.

## Script
Use:
```
scripts/eval/run_multiple_eval_reports.sh
```
This script will run over a `run_id` in `logs/run_evaluation/` and builds an aggregate summary in `eval_reports/`.

### Typical Usage
1. Run evaluations for each model/run_id.
2. Invoke the report script:
    ```
    scripts/eval/run_multiple_eval_reports.sh
    ```
3. View the produced report artifacts (e.g., markdown, JSON, or CSV) in its output directory (commonly a `reports/` or similar folder created by the script).
