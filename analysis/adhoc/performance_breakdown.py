from pathlib import Path
import pandas as pd

reports_dir = Path("eval_reports2")

for report_file in sorted(reports_dir.glob("*.csv")):
    if "gold" in report_file.name:
        continue
    
    df = pd.read_csv(report_file)
    
    # Compute proportion of instances with correctness < 1.0.
    total_instances = len(df)
    incorrect_instances = (df["correctness"] < 1.0).sum()
    proportion_incorrect = incorrect_instances / total_instances if total_instances > 0 else 0.0
    
    # Compute proportion of instances with correctness == 1.0 and raw_pred_speedup_ratio < 1.0.
    correct_but_no_speedup = ((df["correctness"] == 1.0) & (df["raw_pred_speedup_ratio"] < 1.0)).sum()
    proportion_correct_but_no_speedup = correct_but_no_speedup / total_instances if total_instances > 0 else 0.0
    
    # Compute proportion of instances with correctness == 1.0 and raw_pred_speedup_ratio >= 1.0 but human_speedup_ratio < 1.0.
    correct_with_speedup_but_human_no_speedup = ((df["correctness"] == 1.0) & (df["raw_pred_speedup_ratio"] >= 1.0) & (df["human_speedup_ratio"] < 1.0)).sum()
    proportion_correct_with_speedup_but_human_no_speedup = correct_with_speedup_but_human_no_speedup / total_instances if total_instances > 0 else 0.0
    
    # Compute proportin of instances with human_speedup_ratio >= 1.0.
    human_speedup_or_better = (df["human_speedup_ratio"] >= 1.0).sum()
    proportion_human_speedup_or_better = human_speedup_or_better / total_instances if total_instances > 0 else 0.0
    
    print(f"Report: {report_file.name}")
    print(f"  Total instances: {total_instances}")
    print(f"  Proportion incorrect (correctness < 1.0): {proportion_incorrect:.4f}")
    print(f"  Proportion correct but no speedup (correctness == 1.0 and raw_pred_speedup_ratio < 1.0): {proportion_correct_but_no_speedup:.4f}")
    print(f"  Proportion correct with speedup but human no speedup (correctness == 1.0 and raw_pred_speedup_ratio >= 1.0 but human_speedup_ratio < 1.0): {proportion_correct_with_speedup_but_human_no_speedup:.4f}")
    print(f"  Proportion with human speedup or better (human_speedup_ratio >= 1.0): {proportion_human_speedup_or_better:.4f}")
    print()