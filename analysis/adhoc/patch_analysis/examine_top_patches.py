run_metadata = [
    {
        "name": "DeepSeek V3.1 (OH)",
        "eval_report": "eval_reports/eval_report_deepseek-reasoner_maxiter_100_N_v0.51.1-no-hint-run_1.csv",
        "log_dir": "logs/run_evaluation/ground_truth5/deepseek-reasoner_maxiter_100_N_v0.51.1-no-hint-run_1",
    },
    {
        "name": "Gemini 2.5 Flash (OH)",
        "eval_report": "eval_reports/eval_report_gemini-2.5-flash_maxiter_100_N_v0.51.1-no-hint-run_1.csv",
        "log_dir": "logs/run_evaluation/ground_truth5/gemini-2.5-flash_maxiter_100_N_v0.51.1-no-hint-run_1",
    },
    {
        "name": "GPT-5 Mini (OH)",
        "eval_report": "eval_reports/eval_report_gpt-5-mini_maxiter_100_N_v0.51.1-no-hint-run_1.csv",
        "log_dir": "logs/run_evaluation/ground_truth5/gpt-5-mini_maxiter_100_N_v0.51.1-no-hint-run_1",
    },
    {
        "name": "Claude 3.7 Sonnet (OH)",
        "eval_report": "eval_reports/eval_report_us.anthropic.claude-3-7-sonnet-20250219-v1_0_maxiter_100_N_v0.51.1-no-hint-run_1.csv",
        "log_dir": "logs/run_evaluation/ground_truth5/us.anthropic.claude-3-7-sonnet-20250219-v1_0_maxiter_100_N_v0.51.1-no-hint-run_1",
    },
    # SWE-agent
    {
        "name": "Gemini 2.5 Flash (SWE-agent)",
        "eval_report": "eval_reports/eval_report_default_sweperf_gemini__gemini--gemini-2.5-flash__t-0.00__p-1.00__c-1.00___swefficiency_full_test.csv",
        "log_dir": "logs/run_evaluation/ground_truth5/default_sweperf_gemini__gemini--gemini-2.5-flash__t-0.00__p-1.00__c-1.00___swefficiency_full_test",
    },
    {
        "name": "GPT-5 Mini (SWE-agent)",
        "eval_report": "eval_reports/eval_report_default_sweperf_openai__openai--gpt-5-mini__t-1.00__p-1.00__c-1.00___swefficiency_full_test.csv",
        "log_dir": "logs/run_evaluation/ground_truth5/default_sweperf_openai__openai--gpt-5-mini__t-1.00__p-1.00__c-1.00___swefficiency_full_test",
    },
    {
        "name": "Claude 3.7 Sonnet (SWE-agent)",
        "eval_report": "eval_reports/eval_report_default_sweperf_claude__anthropic--claude-3-7-sonnet-20250219__t-0.00__p-1.00__c-1.00___swefficiency_full_test.csv",
        "log_dir": "logs/run_evaluation/ground_truth5/default_sweperf_claude__anthropic--claude-3-7-sonnet-20250219__t-0.00__p-1.00__c-1.00___swefficiency_full_test",
    },
]

from pathlib import Path

import pandas as pd

interesting_patches_dir = "analysis/adhoc/patch_analysis/interesting_patches"

for metadata in run_metadata:
    # Load the evaluation report.
    eval_report = pd.read_csv(metadata["eval_report"])

    # Filter to only >= 1.0 human_speedup_ratio.
    filtered_report = eval_report[eval_report["human_speedup_ratio"] >= 1.0]

    print(f"Run: {metadata['name']}")
    print(f"Total patches with human_speedup_ratio >= 1.0: {len(filtered_report)}")

    for _, row in filtered_report.iterrows():
        instance_id = row["instance_id"]
        human_speedup_ratio = row["human_speedup_ratio"]
        print(f"Instance ID: {instance_id}, Human Speedup Ratio: {human_speedup_ratio}")

        # Load the corresponding log file.
        log_dir = Path(metadata["log_dir"]) / row["instance_id"]
        patch_file = log_dir / f"patch.diff"

        if patch_file.exists():
            with open(patch_file, "r") as f:
                patch_content = f.read()

            # Add a header to the patch content.
            patch_content = (
                f"# Instance ID: {instance_id}\n# Human Speedup Ratio: {human_speedup_ratio}\n=================================\n"
                + patch_content
            )

            # Save the interesting patch to a separate file for further analysis.
            interesting_patch_path = (
                Path(interesting_patches_dir)
                / f"{metadata['name'].replace(' ', '_').replace('(', '').replace(')', '')}"
                / f"{instance_id}_patch.diff"
            )
            interesting_patch_path.parent.mkdir(parents=True, exist_ok=True)
            with open(interesting_patch_path, "w") as f:
                f.write(patch_content)

            print(f"Saved interesting patch to: {interesting_patch_path}")
