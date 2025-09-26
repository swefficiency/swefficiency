import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogFormatter, LogLocator

# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
eval_reports = [
    (
        "OH GPT-5 Mini",
        "eval_reports/eval_report_gpt-5-mini_maxiter_100_N_v0.51.1-no-hint-run_1.csv",
    ),
    (
        "OH Claude 3.7 Sonnet",
        "eval_reports/eval_report_us.anthropic.claude-3-7-sonnet-20250219-v1_0_maxiter_100_N_v0.51.1-no-hint-run_1.csv",
    ),
    (
        "OH Gemini 2.5 Flash",
        "eval_reports/eval_report_gemini-2.5-flash_maxiter_100_N_v0.51.1-no-hint-run_1.csv",
    ),
]

# Choose ONE eval report (by index or by name). Defaults to the first.
EVAL_REPORT_INDEX = 1
# Or: SELECTED_REPORT_NAME = "OH GPT-5 Mini"

# Binning / robustness
NUM_BINS_LOG = 40  # number of log-spaced bins per subplot
START_QUANTILE = 0.15  # start bins at this quantile (None -> min positive)
MIN_COUNT_PER_BIN = 3  # require at least this many tasks to compute HM in a bin


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _select_report(reports, idx=0, name=None):
    if name is not None:
        for n, p in reports:
            if n == name:
                return n, p
    return reports[idx]


def harmonic_mean(x):
    """Harmonic mean of positive, finite values."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return np.nan
    return x.size / np.sum(1.0 / x)


def _pos_finite(a):
    a = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    return a[np.isfinite(a) & (a > 0)]


def _quantile_or_min(a, q):
    if a.size == 0:
        return np.nan
    if q is None:
        return float(np.min(a))
    q = float(np.clip(q, 0.0, 1.0 - 1e-6))  # strictly < 1 for geomspace
    return float(np.quantile(a, q))


def _log_bins(a, num_bins, start_quantile):
    """Return log-spaced bin edges (length num_bins+1)."""
    if a.size == 0:
        return np.array([])
    lo = _quantile_or_min(a, start_quantile)
    hi = float(np.max(a))
    if not (np.isfinite(lo) and np.isfinite(hi) and lo > 0 and hi > 0 and lo < hi):
        return np.array([])
    return np.geomspace(lo, hi, num_bins + 1)


def _bin_harmonic_means(x, ratios, bins, min_count=1):
    """
    Compute harmonic mean of ratios for each [bin[i], bin[i+1]) interval.
    Returns: left_edges, widths, hm_per_bin (NaN for bins with <min_count)
    """
    if bins.size < 2:
        return np.array([]), np.array([]), np.array([])

    # digitize: bin index for each x (0..n_bins-1), right edge exclusive
    idx = np.digitize(x, bins, right=False) - 1
    n_bins = bins.size - 1
    lefts = bins[:-1]
    rights = bins[1:]
    widths = rights - lefts

    hm = np.full(n_bins, np.nan, dtype=float)
    for b in range(n_bins):
        m = idx == b
        if np.count_nonzero(m) >= min_count:
            hm[b] = harmonic_mean(ratios[m])
    return lefts, widths, hm


def style_log_xaxis(ax):
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(LogFormatter(base=10))


# -----------------------------------------------------------------------------
# Load and prep data
# -----------------------------------------------------------------------------
model_name, report_path = _select_report(
    eval_reports, idx=EVAL_REPORT_INDEX
)  # or name=SELECTED_REPORT_NAME
df = pd.read_csv(report_path)

# Filter positives/finites
pre_runtime = _pos_finite(df.get("pre_edit_runtime", np.nan))
gold_sr_x = _pos_finite(df.get("gold_speedup_ratio", np.nan))
patch_len = _pos_finite(df.get("patch_length", np.nan))
ratios = _pos_finite(df.get("human_speedup_ratio", np.nan))


# Also need aligned masks so x and ratios correspond row-wise
# Re-read columns together to preserve alignment, then filter:
def _aligned_xy(colname):
    col = pd.to_numeric(df.get(colname, np.nan), errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(df.get("human_speedup_ratio", np.nan), errors="coerce").to_numpy(
        dtype=float
    )
    m = np.isfinite(col) & (col > 0) & np.isfinite(r) & (r > 0)
    return col[m], r[m]


x_runtime, r_runtime = _aligned_xy("pre_edit_runtime")
x_gold, r_gold = _aligned_xy("gold_speedup_ratio")
x_patch, r_patch = _aligned_xy("patch_length")

# Compute log-spaced bins
bins_runtime = _log_bins(x_runtime, NUM_BINS_LOG, START_QUANTILE)
bins_gold = _log_bins(x_gold, NUM_BINS_LOG, START_QUANTILE)
bins_patch = _log_bins(x_patch, NUM_BINS_LOG, START_QUANTILE)

# Bin-wise harmonic means
l_rt, w_rt, hm_rt = _bin_harmonic_means(
    x_runtime, r_runtime, bins_runtime, MIN_COUNT_PER_BIN
)
l_pl, w_pl, hm_pl = _bin_harmonic_means(x_patch, r_patch, bins_patch, MIN_COUNT_PER_BIN)
l_gd, w_gd, hm_gd = _bin_harmonic_means(x_gold, r_gold, bins_gold, MIN_COUNT_PER_BIN)

# -----------------------------------------------------------------------------
# Plot (3 log-binned bar charts; bar height = HM in bin)
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

fig.supylabel("Speedup Ratio (SR@1, harmonic mean per bin)", fontsize=16)
fig.supxlabel("Task Complexity", fontsize=16)
fig.suptitle(f"{model_name} — Log-binned bar plots of SR@1", fontsize=14)

# 1) Pre-edit runtime
ax = axes[0]
if l_rt.size:
    ax.bar(l_rt, hm_rt, width=w_rt, align="edge", edgecolor="black")
style_log_xaxis(ax)
ax.set_xlabel("≤ Pre-edit workload runtime (s)", fontsize=12)

# 2) Patch length
ax2 = axes[1]
if l_pl.size:
    ax2.bar(l_pl, hm_pl, width=w_pl, align="edge", edgecolor="black")
style_log_xaxis(ax2)
ax2.set_xlabel("≤ Gold patch length (# lines)", fontsize=12)

# 3) Gold patch speedup factor (x-axis is the gold factor; bar height is HM of *human* SR)
ax3 = axes[2]
if l_gd.size:
    ax3.bar(l_gd, hm_gd, width=w_gd, align="edge", edgecolor="black")
style_log_xaxis(ax3)
ax3.set_xlabel("≤ Gold patch speedup factor", fontsize=12)

plt.savefig("docs/assets/figures/scaling_trends_single.png", dpi=300)
plt.close(fig)
