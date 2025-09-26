import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

raw_openhands_traj_dir = Path("predictions/openhands")
eval_reports = Path("eval_reports")

raw_trajectory_files_oh = {
    "gpt-5-mini_oh": "gpt5mini_raw.jsonl",
    "claude-3-7-sonnet_oh": "claude37sonnet_raw.jsonl",
    "gemini-2.5-flash_oh": "gemini25flash_raw.jsonl",
}

polished_trajectory_files_oh = {
    "deepseek-v3.1_oh": "predictions/converted/oh_deepseekv31_traj.jsonl",
}

raw_trajectory_names = {
    "gpt-5-mini_oh": "GPT-5 Mini (OH)",
    "claude-3-7-sonnet_oh": "Claude 3-7 Sonnet (OH)",
    "gemini-2.5-flash_oh": "Gemini 2.5 Flash (OH)",
    "deepseek-v3.1_oh": "DeepSeek V3.1 (OH)",
}

eval_files_oh = {
    "gpt-5-mini_oh": eval_reports
    / "eval_report_gpt-5-mini_maxiter_100_N_v0.51.1-no-hint-run_1.csv",
    "claude-3-7-sonnet_oh": eval_reports
    / "eval_report_us.anthropic.claude-3-7-sonnet-20250219-v1_0_maxiter_100_N_v0.51.1-no-hint-run_1.csv",
    "gemini-2.5-flash_oh": eval_reports
    / "eval_report_gemini-2.5-flash_maxiter_100_N_v0.51.1-no-hint-run_1.csv",
    "gpt-5-mini_sweagent": eval_reports
    / "eval_report_default_sweperf_openai__openai--gpt-5-mini__t-1.00__p-1.00__c-1.00___swefficiency_full_test.csv",
    "claude-3-7-sonnet_sweagent": eval_reports
    / "eval_report_default_sweperf_claude__anthropic--claude-3-7-sonnet-20250219__t-0.00__p-1.00__c-1.00___swefficiency_full_test.csv",
    "gemini-2.5-flash_sweagent": eval_reports
    / "eval_report_default_sweperf_gemini__gemini--gemini-2.5-flash__t-0.00__p-1.00__c-1.00___swefficiency_full_test.csv",
    "deepseek-v3.1_oh": eval_reports
    / "eval_report_deepseek-reasoner_maxiter_100_N_v0.51.1-no-hint-run_1.csv",
}

# model name, marker, color
names = [
    ("gpt-5-mini_oh", "o", "black"),
    ("claude-3-7-sonnet_oh", "^", "#c15f3c"),
    ("gemini-2.5-flash_oh", "s", "#088cfb"),
    ("deepseek-v3.1_oh", "D", "#4d6bfe"),
]


# ---- helper: centered line+marker plot (no external deps) ----
def _plot_median_markers_vertical_centered(
    medians_by_model, bin_labels, filename, styles, *, tick_span=0.35
):
    """
    medians_by_model: dict[str, pd.Series] indexed by bin_labels
    styles: dict[str, dict(marker=..., color=..., alpha?)]
    """
    n_bins = len(bin_labels)
    x = np.linspace(-tick_span / 2.0, +tick_span / 2.0, n_bins)

    fig, ax = plt.subplots(figsize=(5, 3))

    ymax = 0.0
    for m, series in medians_by_model.items():
        vals = series.reindex(bin_labels).astype(float).to_numpy()
        mask = np.isfinite(vals)
        xv, yv = x[mask], vals[mask]

        st = styles.get(m, {})
        marker = st.get("marker", "o")
        color = st.get("color", None)
        alpha = st.get("alpha", 0.7)

        ax.scatter(
            xv,
            yv,
            marker=marker,
            c=color,
            alpha=alpha,
            s=55,
            linewidths=0.8,
            zorder=3,
            label=raw_trajectory_names.get(m, m),
        )
        ax.plot(
            xv, yv, linestyle=st.get("linestyle", "-"), color=color, alpha=0.6, zorder=2
        )

        if yv.size:
            ymax = max(ymax, float(np.nanmax(yv)) * 1.10)

    ax.set_xlim(-0.5, 0.5)  # keep ticks centered & compact
    ax.margins(x=0)
    ax.set_ylim(bottom=0.0, top=105 if ymax > 0 else 1.0)

    # Add a red dashed horizontal line at y=105
    ax.axhline(
        y=100,
        color="red",
        linestyle="--",
        linewidth=2.0,
        alpha=0.7,
        zorder=1,
        label="Eval. Limit (100 turns)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=12)
    ax.tick_params(axis="y", labelsize=11)

    ax.set_xlabel("Speedup Ratio (Per Instance)", fontsize=14, weight="bold")
    ax.set_ylabel("Median Trajectory\nLength", fontsize=14, weight="bold")

    ax.grid(True, axis="y", alpha=0.5, linestyle="--", linewidth=0.6, zorder=0)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--", linewidth=0.6, zorder=0)
    ax.legend(
        fontsize=9,
        loc="upper right",
        bbox_to_anchor=(1, 0.95),
        ncol=2,
        columnspacing=1.2,
        handletextpad=0.3,
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {Path(filename).resolve()}")


# ---- scatter (unchanged; widened xlim so >1.0 shows) ----
plt.figure(figsize=(8, 6))

bucket_labels = [
    "Very Weak\n< 0.25",
    "Weak\n[0.25,0.5)",
    "Moderate\n[0.5, 1.0)",
    "Strong\n≥ 1.0",
]
bucket_edges = [
    0.0,
    0.25,
    0.5,
    1.0,
    np.inf,
]  # [0,0.25], (0.25,0.5], (0.5,1.0], (1.0, inf]

medians_by_model = {}
all_dfs = []
styles = {m: {"marker": mk, "color": c} for (m, mk, c) in names}

for name, marker, color in names:
    eval_report = pd.read_csv(eval_files_oh[name])
    eval_report = eval_report[eval_report["correctness"] == 1.0]

    trajectory_lengths = {}

    if name not in raw_trajectory_files_oh:
        with open(polished_trajectory_files_oh[name], "r") as f:
            for line in f:
                instance_info = json.loads(line)
                if instance_info["trajectory_length"] > 1:
                    print(instance_info["trajectory_length"])
                trajectory_lengths[instance_info["instance_id"]] = instance_info[
                    "trajectory_length"
                ]
    else:
        # load trajectories
        trajectory_file = raw_openhands_traj_dir / raw_trajectory_files_oh[name]
        with open(trajectory_file, "r") as f:
            for line in f:
                traj = json.loads(line)

                history = traj.get("history") or []
                history = [
                    h
                    for h in history
                    if h.get("action") is not None and h.get("source") == "agent"
                ]
                traj_len = len(history)
                trajectory_lengths[traj["instance_id"]] = traj_len

    # collect rows
    rows = []
    instance_ids = set(eval_report["instance_id"])
    for instance_id in list(instance_ids):
        r = eval_report.loc[eval_report["instance_id"] == instance_id]
        if len(r) == 0:
            continue
        human_speedup_ratio = r["human_speedup_ratio"].values[0]
        traj_len = trajectory_lengths.get(instance_id, 0.0)

        rows.append(
            {
                "model": name,
                "instance_id": instance_id,
                "trajectory_length": traj_len,
                "human_speedup_ratio": human_speedup_ratio,
            }
        )

    df = pd.DataFrame(rows).dropna(subset=["human_speedup_ratio"])
    if df.empty:
        print(f"[warn] No matched instances for {name}")
        continue

    # scatter for context
    plt.scatter(
        df["human_speedup_ratio"].values,
        df["trajectory_length"].values,
        alpha=0.5,
        marker=marker,
        label=name,
        color=color,
    )

    # bucket & MEDIAN per bucket
    df["bucket"] = pd.cut(
        df["human_speedup_ratio"].astype(float),
        bins=bucket_edges,
        labels=bucket_labels,
        right=True,
        include_lowest=True,
    )

    med = df.groupby("bucket")["trajectory_length"].median().reindex(bucket_labels)
    medians_by_model[name] = med

    all_dfs.append(
        df[["instance_id", "trajectory_length", "human_speedup_ratio", "bucket"]]
    )

plt.xlabel("Speedup Ratio", fontsize=22)
plt.ylabel("Median Trajectory Length (# Actions)", fontsize=22)

# Set xticks and yticks font size
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.legend()
plt.tight_layout()
# plt.savefig("docs/assets/figures/trajectory_length_vs_speedup.png", dpi=300, bbox_inches="tight", pad_inches=0)
plt.close()

# ---- OVERALL median per bucket (optional extra curve) ----
# overall = (
#     pd.concat(all_dfs, ignore_index=True)
#     if len(all_dfs) else
#     pd.DataFrame(columns=["instance_id","trajectory_length","human_speedup_ratio","bucket"])
# )
# if not overall.empty:
#     overall_med = overall.groupby("bucket")["trajectory_length"].median().reindex(bucket_labels)
#     medians_by_model["OVERALL"] = overall_med
#     styles.setdefault("OVERALL", {"marker": "D", "color": "#666666", "alpha": 0.8})

# ---- write CSV (rows=buckets, cols=models; values=median traj len) ----
summary_median_df = pd.DataFrame({m: s for m, s in medians_by_model.items()})
summary_median_df.index.name = "human_speedup_bucket"
# summary_median_df.to_csv("median_trajectory_len_by_speedup_bucket.csv")

print("\nMedian trajectory length by human speedup bucket (rows) and model (cols):")
print(summary_median_df.round(2))

# ---- centered line+marker plot (replaces bar plot) ----
_plot_median_markers_vertical_centered(
    medians_by_model,
    bucket_labels,
    filename="docs/assets/figures/median_traj_len_by_speedup_bucket.png",
    styles=styles,
    tick_span=0.8,  # tighten or loosen horizontal spacing of tick band
)
