import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------------------------------------------------------
# Inputs (reuse your list; add/remove models as needed)
# -----------------------------------------------------------------------------
eval_reports = [
    (
        "Claude 3.7 Sonnet (OH)",
        "eval_reports/eval_report_us.anthropic.claude-3-7-sonnet-20250219-v1_0_maxiter_100_N_v0.51.1-no-hint-run_1.csv",
    ),
    (
        "GPT-5 Mini (OH)",
        "eval_reports/eval_report_gpt-5-mini_maxiter_100_N_v0.51.1-no-hint-run_1.csv",
    ),
    (
        "Gemini 2.5 Flash (OH)",
        "eval_reports/eval_report_gemini-2.5-flash_maxiter_100_N_v0.51.1-no-hint-run_1.csv",
    ),
]

# -----------------------------------------------------------------------------
# Style mapping (colors per family; marker/linestyle kept for future line plots)
# -----------------------------------------------------------------------------
_model_styles = {
    "GPT-5 Mini (OH)": dict(color="black", marker="o", alpha=0.7),
    "GPT-5 Mini (SWE-agent)": dict(
        color="black", marker="^", alpha=0.7, linestyle="dashed"
    ),
    "Claude 3.7 Sonnet (OH)": dict(color="#c15f3c", marker="o", alpha=0.7),
    "Claude 3.7 Sonnet (SWE-agent)": dict(
        color="#c15f3c", marker="^", alpha=0.7, linestyle="dashed"
    ),
    "Gemini 2.5 Flash (OH)": dict(color="#088cfb", marker="o", alpha=0.7),
    "Gemini 2.5 Flash (SWE-agent)": dict(
        color="#088cfb", marker="^", alpha=0.7, linestyle="dashed"
    ),
}


def _style_key(name: str) -> str:
    """Normalize 'OH Foo' -> 'Foo (OH)' so it matches _model_styles keys."""
    name = name.strip()
    if "(" in name and name.endswith(")"):
        return name
    tokens = name.split()
    if tokens and tokens[0] in ("OH", "SWE-agent"):
        variant = tokens[0]
        base = name[len(variant) :].strip()
        return f"{base} ({variant})"
    return name


# -----------------------------------------------------------------------------
# Config: bucket definition + plotting
# -----------------------------------------------------------------------------
# Choose one of: "quantile", "log", "thresholds"
BIN_MODE = "thresholds"

# For BIN_MODE == "quantile" or "log"
N_BUCKETS = 6
START_QUANTILE = 0.05  # used only for "log"

# For BIN_MODE == "thresholds": provide one or more cut points.
THRESHOLDS = [0.25, 0.5, 1.0]  # split at 0.5
THRESHOLD_LABELS = [
    f"Very Weak\nPerf.\n< {THRESHOLDS[0]}",
    f"Weak\nPerf.\n[{THRESHOLDS[0]}, {THRESHOLDS[1]})",
    f"Moderate\nPerf.\n[{THRESHOLDS[1]}, {THRESHOLDS[2]})",
    f"Strong\nPerf.\n≥ {THRESHOLDS[2]}",
]  # shown on the y-axis

# Which side of threshold is closed/inclusive for pd.cut
THRESHOLD_SIDE = "right"  # "right" or "left"

OUTDIR = Path("assets/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

# One row, three columns: each entry is (metric_column, per-axes x-label, y-scale)
# Note: x-label here is what you asked to show per subplot; bins are on the x-axis.
METRICS = [
    ("pre_edit_runtime", "Pre-edit runtime (s)", "linear"),
    ("gold_speedup_ratio", "Gold speedup factor", "linear"),
    ("patch_length", "Gold patch size (# lines)", "linear"),
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _load_df(path):
    df = pd.read_csv(path)
    cols_needed = [
        "human_speedup_ratio",
        "gold_speedup_ratio",
        "pre_edit_runtime",
        "patch_length",
    ]
    for c in cols_needed:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {path}")
    valid = (
        np.isfinite(df["human_speedup_ratio"])
        & (df["human_speedup_ratio"] > 0)
        & np.isfinite(df["gold_speedup_ratio"])
        & (df["gold_speedup_ratio"] > 0)
        & np.isfinite(df["pre_edit_runtime"])
        & (df["pre_edit_runtime"] > 0)
        & np.isfinite(df["patch_length"])
        & (df["patch_length"] > 0)
    )
    return df.loc[valid, cols_needed].copy()


def _fmt_num(x):
    return f"{x:.3g}"


def _make_global_bins(
    hsr_all,
    n_buckets,
    mode="quantile",
    start_q=0.05,
    *,
    thresholds=None,
    right_inclusive=True,
):
    if mode == "thresholds":
        if thresholds is None or len(thresholds) == 0:
            raise ValueError(
                "BIN_MODE='thresholds' requires THRESHOLDS to be non-empty."
            )
        t = np.unique(np.asarray(thresholds, dtype=float))
        edges = np.r_[[-np.inf], t, [np.inf]]
        return edges

    hsr_all = np.asarray(hsr_all, dtype=float)
    hsr_all = hsr_all[np.isfinite(hsr_all) & (hsr_all > 0)]
    if hsr_all.size == 0:
        raise ValueError(
            "No positive finite human_speedup_ratio values to define bins."
        )

    if mode == "quantile":
        cats = pd.qcut(pd.Series(hsr_all), q=n_buckets, duplicates="drop")
        intervals = cats.cat.categories
        edges = np.r_[[intervals[0].left], [iv.right for iv in intervals]]
    elif mode == "log":
        lo = np.quantile(hsr_all, start_q)
        hi = np.max(hsr_all)
        if not (np.isfinite(lo) and np.isfinite(hi) and lo > 0 and hi > lo):
            raise ValueError("Could not compute valid log bin edges.")
        edges = np.geomspace(lo, hi, n_buckets + 1)
    else:
        raise ValueError("BIN_MODE must be 'quantile', 'log', or 'thresholds'.")
    edges = np.unique(edges)
    if edges.size < 2:
        raise ValueError("Insufficient unique edges to form bins.")
    return edges


def _label_bins(edges, *, mode, right_inclusive=True, custom_labels=None):
    if custom_labels is not None:
        if len(custom_labels) != (len(edges) - 1):
            raise ValueError("THRESHOLD_LABELS length must be len(THRESHOLDS)+1.")
        return list(custom_labels)

    labels = []
    for a, b in zip(edges[:-1], edges[1:]):
        if np.isneginf(a) and np.isposinf(b):
            labels.append("All")
            continue
        if mode == "thresholds":
            if np.isneginf(a):
                comp = "≤" if right_inclusive else "<"
                labels.append(f"{comp} {_fmt_num(b)}×")
            elif np.isposinf(b):
                comp = ">" if right_inclusive else "≥"
                labels.append(f"{comp} {_fmt_num(a)}×")
            else:
                left_comp = ">" if right_inclusive else "≥"
                right_comp = "≤" if right_inclusive else "<"
                labels.append(
                    f"{left_comp} {_fmt_num(a)}× – {right_comp} {_fmt_num(b)}×"
                )
        else:
            left = "-∞" if np.isneginf(a) else _fmt_num(a)
            right = "∞" if np.isposinf(b) else _fmt_num(b)
            labels.append(f"{left}–{right}×")
    return labels


def _bucket_and_aggregate(df, edges, *, right_inclusive=True):
    """Bin by human_speedup_ratio and compute geometric means per bin of each metric."""

    def gmean(series: pd.Series) -> float:
        a = np.asarray(series, dtype=float)
        a = a[np.isfinite(a) & (a > 0)]
        if a.size == 0:
            return np.nan
        return float(np.exp(np.mean(np.log(a))))

    # Print bucket counts for info
    cats = pd.cut(
        df["human_speedup_ratio"],
        bins=edges,
        include_lowest=True,
        right=right_inclusive,
    )
    counts = cats.value_counts(sort=False, dropna=False)
    print("Bucket counts:")
    for interval, count in zip(counts.index, counts.to_numpy()):
        print(f"  {interval}: {count}")

    cats = pd.cut(
        df["human_speedup_ratio"],
        bins=edges,
        include_lowest=True,
        right=right_inclusive,
    )
    tmp = pd.DataFrame(
        {
            "bin": cats,
            "patch_length": df["patch_length"].to_numpy(),
            "gold_speedup_ratio": df["gold_speedup_ratio"].to_numpy(),
            "pre_edit_runtime": df["pre_edit_runtime"].to_numpy(),
        }
    )
    out = (
        tmp.groupby("bin", observed=True)
        .agg(
            {
                "patch_length": gmean,
                "gold_speedup_ratio": gmean,
                "pre_edit_runtime": gmean,
            }
        )
        .reindex(pd.CategoricalIndex(cats.cat.categories), fill_value=np.nan)
    )
    return out  # index: bins


# -----------------------------------------------------------------------------
# Load, pool bins, aggregate per model
# -----------------------------------------------------------------------------
dfs = {}
all_hsr = []
for model, path in eval_reports:
    try:
        df = _load_df(path)
    except Exception as e:
        print(f"[WARN] Skipping {model} ({path}): {e}")
        continue
    dfs[model] = df
    all_hsr.append(df["human_speedup_ratio"].to_numpy())

if not dfs:
    raise SystemExit("No valid reports loaded; nothing to plot.")

right_inclusive = THRESHOLD_SIDE == "right"

edges = _make_global_bins(
    np.concatenate(all_hsr),
    N_BUCKETS,
    mode=BIN_MODE,
    start_q=START_QUANTILE,
    thresholds=THRESHOLDS,
    right_inclusive=right_inclusive,
)

bin_labels = _label_bins(
    edges,
    mode=BIN_MODE,
    right_inclusive=right_inclusive,
    custom_labels=THRESHOLD_LABELS,
)

# Aggregate -> per metric arrays aligned to bins
agg = {
    model: _bucket_and_aggregate(df, edges, right_inclusive=right_inclusive)
    for model, df in dfs.items()
}

# # --- Make "Strong" appear first (left) and "Weak" to its right ----------------
# # For any binning mode, pd.cut categories are ascending by HSR. Reversing rows
# # makes higher-HSR (strong) bins come first on the x-axis for column plots.
# def _strong_first(agg_dict, labels):
#     labels_rev = list(labels[::-1])
#     agg_rev = {m: df.iloc[::-1].copy() for m, df in agg_dict.items()}
#     return agg_rev, labels_rev

# agg, bin_labels = _strong_first(agg, bin_labels)


# --- Plotter: horizontal grouped bars with top→bottom order matching colors --
def _plot_subplots_grouped_bars_horizontal(agg, bin_labels, metrics, filename):
    xlabel_font_size = 22
    ylabel_font_size = 22
    sub_xlabel_font_size = 14
    sub_ylabel_font_size = 14
    xtick_font_size = 10
    ytick_font_size = 10

    models = list(
        agg.keys()
    )  # keep this in the legend & bar order you want (colors map to these)
    n_bins = len(bin_labels)
    n_models = len(models)

    y = np.arange(n_bins)
    group_height = 0.8
    bar_h = group_height / max(n_models, 1)

    # Offsets so j=0 is TOP, j=1 is next, etc. (top→bottom matches models/colors)
    offsets = ((n_models - 1) / 2.0 - np.arange(n_models)) * bar_h

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(max(12, 4 * len(metrics)), 3),
        sharey=True,
        constrained_layout=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    added_legend = False
    for k, (metric, x_label, x_scale) in enumerate(metrics):
        ax = axes[k]
        for j, m in enumerate(models):
            vals = np.asarray(agg[m][metric].to_numpy(), dtype=float)
            ypos = y + offsets[j]

            style = _model_styles.get(_style_key(m), {})
            color = style.get("color", None)
            alpha = style.get("alpha", 0.9)

            vals_plot = np.where(
                np.isfinite(vals) & ((vals > 0) if x_scale == "log" else True),
                vals,
                np.nan,
            )
            ax.barh(ypos, vals_plot, height=bar_h, label=m, color=color, alpha=alpha)

        # y ticks = difficulty bins (show labels on the left subplot only)
        ax.set_yticks(y)
        ax.set_yticklabels(
            bin_labels, fontsize=sub_ylabel_font_size, multialignment="center"
        )  # will show: "Below 0.5", "Above 0.5"

        # ax.grid(True, axis="x", alpha=0.3)
        if x_scale == "log":
            ax.set_xscale("log")

        ax.set_xlabel(x_label, fontsize=sub_xlabel_font_size)
        ax.margins(y=0.10)

        ax.tick_params(axis="x", labelsize=xtick_font_size)
        ax.tick_params(axis="y", labelsize=ytick_font_size)

        if k == 0 and not added_legend:
            ax.legend(fontsize=10, loc="best")
            added_legend = True

    fig.supylabel("Speedup Ratio", fontsize=ylabel_font_size, weight="bold")
    fig.supxlabel("Difficulty Measure", fontsize=xlabel_font_size, weight="bold")
    fig.subplots_adjust(bottom=0.18)
    out_path = OUTDIR / filename
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved subplot figure to: {out_path.resolve()}")


# --- Plotter: VERTICAL grouped bars (bins on x; left→right matches models/colors) --
def _plot_subplots_grouped_bars_vertical(agg, bin_labels, metrics, filename):
    xlabel_font_size = 22
    ylabel_font_size = 22
    sub_xlabel_font_size = 14
    sub_ylabel_font_size = 14
    xtick_font_size = 10
    ytick_font_size = 10

    models = list(
        agg.keys()
    )  # keep this in the legend & bar order you want (colors map to these)
    n_bins = len(bin_labels)
    n_models = len(models)

    x = np.arange(n_bins)
    group_width = 0.8
    bar_w = group_width / max(n_models, 1)

    # Offsets so j=0 is leftmost, then rightward (left→right matches models/colors)
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * bar_w

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(max(12, 4 * len(metrics)), 4.0),
        sharex=True,
        constrained_layout=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    added_legend = False
    for k, (metric, y_label, y_scale) in enumerate(metrics):
        ax = axes[k]
        for j, m in enumerate(models):
            vals = np.asarray(agg[m][metric].to_numpy(), dtype=float)
            xpos = x + offsets[j]

            style = _model_styles.get(_style_key(m), {})
            color = style.get("color", None)
            alpha = style.get("alpha", 0.9)

            # For log scale, drop nonpositive values
            vals_plot = np.where(
                np.isfinite(vals) & ((vals > 0) if y_scale == "log" else True),
                vals,
                np.nan,
            )
            ax.bar(xpos, vals_plot, width=bar_w, label=m, color=color, alpha=alpha)

        # x ticks = difficulty bins
        ax.set_xticks(x)
        ax.set_xticklabels(
            bin_labels, fontsize=sub_xlabel_font_size, rotation=0, ha="center"
        )

        if y_scale == "log":
            ax.set_yscale("log")

        # Metric label now belongs on the y-axis for each subplot
        ax.set_ylabel(y_label, fontsize=sub_ylabel_font_size)

        ax.margins(x=0.10)
        ax.tick_params(axis="x", labelsize=xtick_font_size)
        ax.tick_params(axis="y", labelsize=ytick_font_size)

        if k == 0:
            ax.legend(fontsize=10, loc="best")
            added_legend = True

    # Global labels: bins (HSR) are on X now; Y is per-subplot metric value
    fig.supxlabel(
        "Speedup Ratio (Per Instance)", fontsize=xlabel_font_size, weight="bold"
    )

    # shift down ylabel.
    fig.supylabel(
        "Average Difficulty Measure", fontsize=ylabel_font_size, weight="bold"
    )
    fig.subplots_adjust(bottom=0.22)

    out_path = OUTDIR / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"Saved subplot figure to: {out_path.resolve()}")


# -----------------------------------------------------------------------------
# Plot a single 1×3 subplot figure (vertical)
# -----------------------------------------------------------------------------
_plot_subplots_grouped_bars_vertical(
    agg=agg,
    bin_labels=bin_labels,
    metrics=METRICS,
    filename=f"hsr_{BIN_MODE}_bins_subplots_vertical.png",
)

print(f"Saved plots to: {OUTDIR.resolve()}")

# Style mapping: same color per foundation model family; different marker/alpha per variant
_model_styles = {
    "GPT-5 Mini (OH)": dict(color="black", marker="^", alpha=0.7),
    "GPT-5 Mini (SWE-agent)": dict(
        color="black", marker="o", alpha=0.9, linestyle="dashed"
    ),
    "Claude 3.7 Sonnet (OH)": dict(color="#c15f3c", marker="o", alpha=0.7),
    "Claude 3.7 Sonnet (SWE-agent)": dict(
        color="#c15f3c", marker="o", alpha=0.9, linestyle="dashed"
    ),
    "Gemini 2.5 Flash (OH)": dict(color="#088cfb", marker="s", alpha=0.7),
    "Gemini 2.5 Flash (SWE-agent)": dict(
        color="#088cfb", marker="o", alpha=0.9, linestyle="dashed"
    ),
}


# --- Plotter: centered markers (no lines), one vertical stack per x tick -----
# --- Plotter: centered markers (no jitter), compact tick spacing -------------
def _plot_markers_vertical_centered(
    agg, bin_labels, metrics, filename, *, tick_span=0.8
):
    """
    tick_span ∈ (0, 1]: fraction of axis width used by all x ticks.
    Smaller = ticks closer together (still centered). Try 0.25–0.45.
    """
    xlabel_font_size = 18
    ylabel_font_size = 18
    sub_xlabel_font_size = 14
    sub_ylabel_font_size = 14
    xtick_font_size = 10
    ytick_font_size = 10

    models = list(agg.keys())
    n_bins = len(bin_labels)

    # Place ticks in a compact, centered band; we'll keep x-limits fixed to [-0.5, 0.5]
    x = np.linspace(-tick_span / 2.0, +tick_span / 2.0, n_bins)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(metrics),
        figsize=(12, 3.5),
        sharex=True,
        constrained_layout=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for k, (metric, y_label, y_scale) in enumerate(metrics):
        ax = axes[k]
        ymin = 1.0 if y_scale == "log" else 0.0
        ymax = 0.0

        for m in models:
            vals = np.asarray(agg[m][metric].to_numpy(), dtype=float)
            mask = np.isfinite(vals) & ((vals > 0) if y_scale == "log" else True)
            xv, yv = x[mask], vals[mask]

            style = _model_styles.get(_style_key(m), {})
            marker = style.get("marker", "o")
            color = style.get("color", None)
            alpha = style.get("alpha", 0.9)

            # Points
            ax.scatter(
                xv,
                yv,
                marker=marker,
                c=color,
                alpha=alpha,
                s=55,
                linewidths=0.8,
                zorder=3,
                label=m,
            )
            # Connecting line (respect optional linestyle from style)
            ax.plot(
                xv,
                yv,
                linestyle=style.get("linestyle", "-"),
                color=color,
                alpha=0.5,
                zorder=2,
            )

            if np.any(np.isfinite(vals)):
                ymax = max(ymax, float(np.nanmax(vals)) * 1.10)

        if k == 2:  # optional fixed range for patch length
            ymin, ymax = 30, 80
            ax.yaxis.set_label_coords(-0.11, 0)

        ax.set_ylim(bottom=ymin, top=ymax)
        ax.set_xlim(-0.5, 0.5)  # keep ticks centered & compact
        ax.margins(x=0)

        ax.set_ylabel(y_label, fontsize=sub_ylabel_font_size)
        if y_scale == "log":
            ax.set_yscale("log")
        ax.tick_params(axis="y", labelsize=ytick_font_size)
        ax.grid(True, axis="y", alpha=0.5, linestyle="--", linewidth=0.6, zorder=0)
        ax.grid(True, axis="x", alpha=0.5, linestyle="--", linewidth=0.6, zorder=0)

        ax.legend(fontsize=10, loc="upper right")

        ax.yaxis.set_label_coords(-0.12, 0.48)

    # Shared x ticks/labels
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        bin_labels, fontsize=sub_xlabel_font_size, rotation=0, ha="center"
    )
    axes[-1].tick_params(axis="x", labelsize=xtick_font_size)

    fig.supxlabel(
        "Speedup Ratio (Per Instance)", fontsize=xlabel_font_size, weight="bold"
    )
    fig.supylabel("Difficulty Metric", fontsize=ylabel_font_size, weight="bold")

    # plt.tight_layout()

    out_path = OUTDIR / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved subplot figure to: {out_path.resolve()}")


# -----------------------------------------------------------------------------
# Plot the centered-markers version
# -----------------------------------------------------------------------------
_plot_markers_vertical_centered(
    agg=agg,
    bin_labels=bin_labels,
    metrics=METRICS,
    filename=f"hsr_{BIN_MODE}_bins_markers_centered.png",
)


print(f"Saved plots to: {OUTDIR.resolve()}")
