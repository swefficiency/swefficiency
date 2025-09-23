import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogFormatter, LogLocator

# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
# Model eval reports
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
    (
        "DeepSeek V3.1 (OH)",
        "eval_reports/eval_report_deepseek-reasoner_maxiter_100_N_v0.51.1-no-hint-run_1.csv",
    ),
    (
        "GPT-5 Mini (SWE-agent)",
        "eval_reports/eval_report_default_sweperf_openai__openai--gpt-5-mini__t-1.00__p-1.00__c-1.00___swefficiency_full_test.csv",
    ),
    (
        "Claude 3.7 Sonnet (SWE-agent)",
        "eval_reports/eval_report_default_sweperf_claude__anthropic--claude-3-7-sonnet-20250219__t-0.00__p-1.00__c-1.00___swefficiency_full_test.csv",
    ),
    (
        "Gemini 2.5 Flash (SWE-agent)",
        "eval_reports/eval_report_default_sweperf_gemini__gemini--gemini-2.5-flash__t-0.00__p-1.00__c-1.00___swefficiency_full_test.csv",
    ),
]

# Style mapping: same color per foundation model family; different marker/alpha per variant
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
    "DeepSeek V3.1 (OH)": dict(color="#2b4eff", marker="D", alpha=0.7),
}

# Monkey-patch Axes.plot so existing plotting code (which sets marker='o') is overridden
import matplotlib.axes as _maxes

_original_plot = _maxes.Axes.plot


def _styled_plot(self, *args, **kwargs):
    label = kwargs.get("label")
    if label in _model_styles:
        style = _model_styles[label]
        # Force our styles
        kwargs["color"] = style["color"]
        kwargs["marker"] = style["marker"]
        kwargs["linestyle"] = style.get("linestyle", "solid")

        # Preserve user alpha if explicitly provided, else apply ours
        if "alpha" not in kwargs:
            kwargs["alpha"] = style["alpha"]
        # Make lines a bit thicker for visibility
        kwargs.setdefault("linewidth", 1.0)
    return _original_plot(self, *args, **kwargs)


_maxes.Axes.plot = _styled_plot

# Number of points per curve; increase for smoother lines
NUM_POINTS_LOG = 10

# If True, every model uses the same global log-spaced thresholds (fair comparison).
# If False, each model uses its own min/max (curves can start/stop at different x).
ALIGN_MODELS = True

# Start the grid at this quantile of the sort column.
# - If ALIGN_MODELS=True, we compute this quantile over the *pooled* x from all models.
# - If ALIGN_MODELS=False, we compute it per-model (each curve starts at its own quantile).
# Set to None to start at the minimum positive value.
START_QUANTILE = 0.15  # e.g., 1st percentile; must be < 1.0 (or set to None)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def harmonic_mean(x):
    """Harmonic mean of positive, finite values."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return np.nan
    return x.size / np.sum(1.0 / x)


def _prep_model_df(report_path, sort_col):
    """Load -> filter -> sort -> precompute cumulative stats for fast HM queries."""
    df = pd.read_csv(report_path)

    # Log x-axis requires positive thresholds; HM requires positive ratios
    valid = (
        np.isfinite(df.get(sort_col, np.nan))
        & (df[sort_col] > 0)
        & np.isfinite(df.get("human_speedup_ratio", np.nan))
        & (df["human_speedup_ratio"] > 0)
    )
    df = df[valid].sort_values(by=sort_col).reset_index(drop=True)
    if len(df) == 0:
        return None

    s = df[sort_col].to_numpy(dtype=float)  # sorted thresholds
    r = df["human_speedup_ratio"].to_numpy(dtype=float)  # corresponding ratios
    inv = 1.0 / r
    csum_inv = np.cumsum(inv)  # prefix sums of 1/ratio
    return {"x_sorted": s, "csum_inv": csum_inv}


def _quantile_or_min(a, q):
    """Return quantile(q) if q is not None, else min(a). Clamp q into [0, 1-1e-6]."""
    if a.size == 0:
        return np.nan
    if q is None:
        return np.min(a)
    q = float(q)
    q = min(max(q, 0.0), 1.0 - 1e-6)  # ensure strictly < 1 for geomspace
    return np.quantile(a, q)


def _global_log_grid(model_data_list, num_points, start_quantile):
    """Compute a single global log-spaced grid across all models."""
    xs = [
        d["x_sorted"]
        for d in model_data_list
        if d is not None and d["x_sorted"].size > 0
    ]
    if not xs:
        return np.array([])
    all_x = np.concatenate(xs)
    all_x = all_x[all_x > 0]
    if all_x.size == 0:
        return np.array([])

    lo = _quantile_or_min(all_x, start_quantile)
    hi = np.max(all_x)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo > 0 and hi > 0 and lo < hi):
        return np.array([])

    return np.geomspace(lo, hi, num_points)


def _per_model_log_grid(d, num_points, start_quantile):
    """Model-specific log-spaced grid from its own (quantile→max)."""
    s = d["x_sorted"]
    if s.size == 0:
        return np.array([])
    lo = _quantile_or_min(s, start_quantile)
    hi = s[-1]
    if not (np.isfinite(lo) and np.isfinite(hi) and lo > 0 and hi > 0 and lo < hi):
        return np.array([])
    return np.geomspace(lo, hi, num_points)


def _hm_up_to_thresholds(d, thresholds):
    """
    For each threshold t, include rows with x_sorted <= t.
    If k rows are included, HM = k / csum_inv[k-1].
    """
    if d is None or thresholds.size == 0:
        return np.array([]), np.array([])

    s = d["x_sorted"]
    csum_inv = d["csum_inv"]

    k = np.searchsorted(s, thresholds, side="right")  # count of elements <= t
    y = np.full_like(thresholds, np.nan, dtype=float)
    mask = k > 0
    y[mask] = k[mask] / csum_inv[k[mask] - 1]

    return thresholds, y


def compute_curve_logspaced(
    selected_eval_reports,
    sort_col,
    num_points=30,
    align_models=True,
    start_quantile=None,
):
    """
    Build curves using log-spaced thresholds, optionally starting at a given quantile.
    - If align_models=True: one global grid, first x is pooled quantile.
    - If align_models=False: per-model grids, first x is that model's quantile.
    Returns: dict[model] -> (x_values, y_values)
    """
    # Prep per-model data
    per_model = {}
    for model_name, report_path in selected_eval_reports:
        per_model[model_name] = _prep_model_df(report_path, sort_col)

    # Choose threshold grid(s)
    global_grid = (
        _global_log_grid(list(per_model.values()), num_points, start_quantile)
        if align_models
        else None
    )

    curves = {}
    for model_name, d in per_model.items():
        if d is None:
            curves[model_name] = (np.array([]), np.array([]))
            continue

        thresholds = (
            global_grid
            if align_models
            else _per_model_log_grid(d, num_points, start_quantile)
        )
        x, y = _hm_up_to_thresholds(d, thresholds)
        curves[model_name] = (x, y)

    return curves


# -----------------------------------------------------------------------------
# Compute curves for each x-axis definition
# -----------------------------------------------------------------------------
pre_edit_xy = compute_curve_logspaced(
    eval_reports,
    "pre_edit_runtime",
    num_points=NUM_POINTS_LOG,
    align_models=ALIGN_MODELS,
    start_quantile=START_QUANTILE,
)
gold_xy = compute_curve_logspaced(
    eval_reports,
    "gold_speedup_ratio",
    num_points=NUM_POINTS_LOG,
    align_models=ALIGN_MODELS,
    start_quantile=START_QUANTILE,
)
patchlen_xy = compute_curve_logspaced(
    eval_reports,
    "patch_length",
    num_points=NUM_POINTS_LOG,
    align_models=ALIGN_MODELS,
    start_quantile=START_QUANTILE,
)

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

# Bold font for axis labels
fig.supylabel("Speedup Ratio", fontsize=22, weight="bold")
fig.supxlabel("Benchmark Difficulty", fontsize=22, weight="bold")


def style_log_xaxis(ax):
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(LogFormatter(base=10))
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(fontsize=12)


# 1) Thresholds on pre-edit runtime
ax = axes[0]

for model, (x, y) in pre_edit_xy.items():
    if x.size:
        m = ~np.isnan(y)
        ax.plot(x[m], y[m], label=model)
style_log_xaxis(ax)
ax.set_xlabel("Pre-edit workload runtime (s)", fontsize=18)
# ax.set_title('Speedup vs runtime threshold')

# 2) Thresholds on gold speedup ratio
ax2 = axes[2]

for model, (x, y) in gold_xy.items():
    if x.size:
        m = ~np.isnan(y)
        ax2.plot(x[m], y[m], label=model)
style_log_xaxis(ax2)
ax2.set_xlabel("Gold patch speedup", fontsize=18)
# ax2.set_ylabel('Speedup Ratio (SR @ 1)')
# ax2.set_title('Speedup vs gold-speedup threshold')

# 3) Thresholds on patch length
ax3 = axes[1]

for model, (x, y) in patchlen_xy.items():
    if x.size:
        m = ~np.isnan(y)
        ax3.plot(x[m], y[m], label=model)
style_log_xaxis(ax3)
ax3.set_xlabel("Gold patch length (# lines)", fontsize=18)
# ax3.set_ylabel('Aggregate speedup ratio (harmonic mean ≤ x)')
# ax3.set_title('Speedup vs patch-length threshold')
ax3.set_ylim(bottom=-0.0015, top=0.03)

plt.savefig("assets/figures/scaling_trends_all.png", dpi=300)
plt.close(fig)
