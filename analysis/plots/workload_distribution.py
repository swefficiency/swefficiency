from pathlib import Path

import numpy as np
import tqdm
from datasets import load_dataset

from swefficiency.harness.run_validation import parse_perf_summary

# ===================== MANUAL LAYOUT CONFIG =====================
# Give each subplot's width in *inches*. If None, all subplots are equal width.
# Example: [10, 5, 5] makes ax0 twice as wide as ax1/ax2.
SUBPLOT_WIDTHS_IN = [6.5, 6.5, 4.0]  # Set to None for equal widths

# If FIG_WIDTH is None and SUBPLOT_WIDTHS_IN is set, we'll use the sum of those widths.
# If both are None, default width is 18 inches.
FIG_WIDTH = None
FIG_HEIGHT = 4.5

PURPLE_COLOR = "#A64CA6"
GOLD_COLOR = "#FFC94D"

# Layout behavior:
USE_CONSTRAINED = True  # If False, we'll use fig.subplots_adjust with WSPACE below
WSPACE = 0.04  # Only used when USE_CONSTRAINED = False
# ================================================================

# ---------------- Load data ----------------
ds = load_dataset("swefficiency-anon/swefficiency", split="test")
gold_dir = Path("logs/run_evaluation/ground_truth5/gold")

pre_edit_workload_runtimes = []
post_edit_workload_runtimes = []
speedups = []

for instance in tqdm.tqdm(ds, total=len(ds), desc="Processing instances"):
    instance_id = instance["instance_id"]
    gold_run_entry = gold_dir / instance_id / "perf_summary.txt"
    if not gold_run_entry.exists():
        print(f"Instance {instance_id} has no gold perf summary.", flush=True)
        continue

    gold_perf_info = parse_perf_summary(gold_run_entry.read_text())
    gold_speedup_ratio = gold_perf_info["before_mean"] / gold_perf_info["after_mean"]

    pre_edit_workload_runtimes.append(gold_perf_info["before_mean"])
    post_edit_workload_runtimes.append(gold_perf_info["after_mean"])
    speedups.append(gold_speedup_ratio)


#!/usr/bin/env python3
import json
import textwrap
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

# Ensure hatch lines are visible
mpl.rcParams["hatch.linewidth"] = 1.5

# ---------------- Figure & axes with manual sizing ----------------
if SUBPLOT_WIDTHS_IN is not None:
    # Guard against zero/negatives; treat the given inches as width ratios too.
    width_ratios = [max(0.1, float(w)) for w in SUBPLOT_WIDTHS_IN]
    fig_width = FIG_WIDTH if FIG_WIDTH is not None else sum(width_ratios)
else:
    width_ratios = [1, 1, 1]
    fig_width = FIG_WIDTH if FIG_WIDTH is not None else 18.0

fig = plt.figure(figsize=(fig_width, FIG_HEIGHT), constrained_layout=USE_CONSTRAINED)
gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=width_ratios)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax0 = fig.add_subplot(gs[0, 2])


if not USE_CONSTRAINED:
    fig.subplots_adjust(wspace=WSPACE)

# ---------------- Subplot 1: horizontal bar counts ----------------
JSONL_PATH = "analysis/llm/outputs/diff_classification_results.jsonl"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


field = "classification"
counter = Counter()
for obj in read_jsonl(JSONL_PATH):
    if field in obj and obj[field] is not None:
        counter[str(obj[field])] += 1

items = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
labels, counts = zip(*items) if items else ([], [])

print("\nCounts:")
if labels:
    width = max(len(s) for s in labels)
    for lab, c in items:
        print(f"{lab:<{width}}  {c}")

positions = range(len(labels))

import textwrap

MAX_LABEL_CHARS = 23


def _wrap(label):
    if len(label) <= MAX_LABEL_CHARS:
        return label
    # Replace underscores with spaces for nicer wrapping
    label = label.replace("_", " ")
    return textwrap.fill(label, width=MAX_LABEL_CHARS, break_long_words=False)


# Relabel the labels:
def relabel(text):
    if text.startswith("Code Simplification"):
        return "Simplify Code"
    if text.startswith("Algorithmic"):
        return "Algorithmic Change"
    if text.startswith("Memory"):
        return "Memory Efficiency"
    if text.startswith("Caching & Reuse"):
        return "Caching & Reuse"
    if text.startswith("Concurrency"):
        return "Parallelism"
    if text.startswith("Configuration"):
        return "Config Tuning"
    if text.startswith("Compiler"):
        return "Low-level Tuning"

    return text


labels = [relabel(l) for l in labels]

wrapped_labels = [_wrap(l) for l in labels]


bars = ax0.barh(positions, counts, alpha=1.0, color=PURPLE_COLOR)
ax0.set_yticks(list(positions), wrapped_labels, fontsize=20)
ax0.set_xlabel("# of Instances", fontsize=20)
ax0.invert_yaxis()
if counts:
    ax0.set_xlim(0, max(counts) * 1.2)
    for bar, count in zip(bars, counts):
        ax0.text(
            bar.get_width() + (0.01 * max(counts)),
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            ha="left",
            fontsize=14,
            fontweight="bold",
        )

ax0.set_title("Gold Patch Categories", fontsize=24, x=-0.05)
ax0.tick_params(axis="x", labelsize=13)

# ---------------- Subplot 2: runtimes hist ----------------
all_runtimes = pre_edit_workload_runtimes + post_edit_workload_runtimes
if (
    all_runtimes
    and min(all_runtimes) > 0
    and max(all_runtimes) > 0
    and min(all_runtimes) != max(all_runtimes)
):
    bins_rt = np.logspace(np.log10(min(all_runtimes)), np.log10(max(all_runtimes)), 30)
else:
    bins_rt = 30  # fallback if empty or degenerate

ax1.hist(
    pre_edit_workload_runtimes,
    bins=bins_rt,
    alpha=1.0,
    color=PURPLE_COLOR,
    label="Before Gold Patch",
    zorder=1,
)
ax1.hist(
    post_edit_workload_runtimes,
    bins=bins_rt,
    histtype="step",
    label="After Gold Patch",
    hatch="//",
    linewidth=1.5,
    facecolor="none",
    edgecolor=GOLD_COLOR,
    zorder=2,
)
ax1.set_title("Workload Runtimes", fontsize=24)
ax1.set_xlabel("Runtime (seconds)", fontsize=20)
ax1.set_ylabel("# of Instances", fontsize=20)
ax1.set_xscale("log")
ax1.tick_params(labelsize=13)
ax1.legend(fontsize=16, loc="upper left")
ax1.grid(True)
ax1.set_axisbelow(True)

ax1.xaxis.set_major_locator(LogLocator(base=10))
ax1.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.2))
ax1.xaxis.set_minor_formatter(NullFormatter())
ax1.grid(True, which="major", axis="x", alpha=0.6)
ax1.grid(True, which="minor", axis="x", alpha=0.25)

# ---------------- Subplot 3: speedups hist ----------------
if (
    speedups
    and min(speedups) > 0
    and max(speedups) > 0
    and min(speedups) != max(speedups)
):
    bins_sp = np.logspace(np.log10(min(speedups)), np.log10(max(speedups)), 30)
else:
    bins_sp = 30  # fallback

ax2.hist(speedups, bins=bins_sp, alpha=1.0, color=GOLD_COLOR, zorder=1)
ax2.set_title("Gold Patch Speedup", fontsize=24)
ax2.set_xlabel("Speedup Factor (before/after)", fontsize=20)
# ax2.set_ylabel('Number of Instances', fontsize=22)
ax2.set_xscale("log")
ax2.tick_params(labelsize=13)
ax2.grid(True)
ax2.set_axisbelow(True)

ax2.xaxis.set_major_locator(LogLocator(base=10))
ax2.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10, 2) * 0.1))
ax2.xaxis.set_minor_formatter(NullFormatter())
ax2.grid(True, which="major", axis="x", alpha=0.6)
ax2.grid(True, which="minor", axis="x", alpha=0.25)

# plt.tight_layout()
fig.savefig(
    "assets/figures/workload_distribution.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.1,
)
plt.close(fig)
