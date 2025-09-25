import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tqdm
from datasets import load_dataset

from swefficiency.harness.run_validation import parse_perf_summary

# ----------------------------
# Config
# ----------------------------
NUM_BINS = 20
TARGET_N = 100  # keep it < 100; adjust as you like
RANDOM_SEED = 42  # reproducibility
GOLD_RESULT_FOLDER = Path("logs/run_evaluation/ground_truth5/gold")
SPLIT = "test"  # change if needed
DATASET_ID = "swefficiency/swefficiency"  # as in your snippet

rng = np.random.default_rng(RANDOM_SEED)


# ----------------------------
# Helpers
# ----------------------------
def _logspace_edges(values: np.ndarray, num_bins: int) -> np.ndarray:
    """Compute log-spaced bin edges for strictly positive data, with robust fallbacks."""
    vpos = values[values > 0]
    if vpos.size == 0:
        # Fallback if everything is non-positive (shouldn't happen with these metrics)
        vmin, vmax = 1e-6, 1.0
    else:
        vmin, vmax = vpos.min(), vpos.max()
        if vmin == vmax:
            # Avoid degenerate edges; widen slightly
            vmin *= 0.9
            vmax *= 1.1
    return np.logspace(np.log10(vmin), np.log10(vmax), num_bins + 1)


def _assign_bins(
    values: Dict[str, float], num_bins: int
) -> Tuple[List[List[str]], Dict[str, int]]:
    """
    Assign instance_ids to log-spaced bins based on their numeric values.
    Returns:
      - bins: list of lists of instance_ids per bin index [0..num_bins-1]
      - idx_map: instance_id -> bin index
    """
    ids = np.array(list(values.keys()))
    vals = np.array([values[i] for i in ids], dtype=float)

    edges = _logspace_edges(vals, num_bins)

    # Push non-positive values just inside the first edge to be captured by bin 0
    min_pos = edges[0]
    safe_vals = np.where(vals > 0, vals, min_pos * 0.999)

    # np.digitize with internal edges produces bin indices 0..num_bins-1
    bin_idx = np.digitize(safe_vals, edges[1:-1], right=False)

    bins = [[] for _ in range(num_bins)]
    for iid, b in zip(ids, bin_idx):
        bins[b].append(iid)

    idx_map = {iid: int(b) for iid, b in zip(ids, bin_idx)}
    return bins, idx_map


def _multinomial_allocation(
    counts: List[int], total: int, rng: np.random.Generator
) -> List[int]:
    """Allocate 'total' draws across bins with probability proportional to 'counts'."""
    counts = np.array(counts, dtype=float)
    if counts.sum() == 0:
        # No items; return all zero
        return [0] * len(counts)
    probs = counts / counts.sum()
    draws = rng.multinomial(total, probs)
    return draws.tolist()


def _sample_from_bins(
    bins: List[List[str]], alloc: List[int], rng: np.random.Generator
) -> List[str]:
    """Sample alloc[k] items from bins[k] without replacement (or all if not enough)."""
    selected = []
    shortfall = 0
    surplus_pool = []

    for k, want in enumerate(alloc):
        pool = bins[k]
        if want <= 0 or len(pool) == 0:
            continue
        if want >= len(pool):
            # take all, accumulate any "unused capacity" to redistribute later
            selected.extend(pool)
            shortfall += want - len(pool)
        else:
            chosen = rng.choice(pool, size=want, replace=False).tolist()
            selected.extend(chosen)
            # Keep the unchosen in a surplus pool in case we need to top up later
            remaining = list(set(pool) - set(chosen))
            surplus_pool.extend(remaining)

    # If we wanted more than we could take from some sparse bins, top up from surplus_pool
    if shortfall > 0 and surplus_pool:
        top_up = min(shortfall, len(surplus_pool))
        selected.extend(rng.choice(surplus_pool, size=top_up, replace=False).tolist())

    return selected


def _inverse_density_weights(
    sigs: Dict[str, Tuple[int, int, int]], bins_by_metric: Dict[str, List[List[str]]]
) -> Dict[str, float]:
    """Weight each instance by inverse product of its bin densities across all three metrics."""
    # Precompute bin sizes
    sizes = {
        "size_of_patch": [len(b) for b in bins_by_metric["size_of_patch"]],
        "pre_edit_workload_runtime": [
            len(b) for b in bins_by_metric["pre_edit_workload_runtime"]
        ],
        "improvement": [len(b) for b in bins_by_metric["improvement"]],
    }
    weights = {}
    for iid, (b0, b1, b2) in sigs.items():
        denom = (
            (sizes["size_of_patch"][b0] or 1)
            * (sizes["pre_edit_workload_runtime"][b1] or 1)
            * (sizes["improvement"][b2] or 1)
        )
        weights[iid] = 1.0 / denom
    return weights


# ----------------------------
# Collect metrics
# ----------------------------
ds = load_dataset(DATASET_ID, split=SPLIT)

key_info = {}
for d in ds:
    if d["instance_id"] not in os.listdir(GOLD_RESULT_FOLDER):
        continue

    instance_id = d["instance_id"]

    # Metric 1: size_of_patch
    patch = d.get("patch", "") or ""
    size_of_patch = len(str(patch).splitlines())

    # Parse performance summary
    perf_summary = GOLD_RESULT_FOLDER / instance_id / "perf_summary.txt"
    if not perf_summary.exists():
        # Skip gracefully if missing
        continue

    try:
        parsed = parse_perf_summary(perf_summary.read_text())
    except Exception:
        # Skip instances with unparsable summaries
        continue

    before = float(parsed.get("before_mean", np.nan))
    after = float(parsed.get("after_mean", np.nan))
    if not np.isfinite(before) or not np.isfinite(after) or after <= 0:
        # Guard against bad numbers
        continue

    pre_edit_workload_runtime = before  # Metric 2
    improvement = before / after  # Metric 3 (speedup factor)

    key_info[instance_id] = {
        "size_of_patch": size_of_patch,
        "pre_edit_workload_runtime": pre_edit_workload_runtime,
        "improvement": improvement,
    }

if not key_info:
    raise RuntimeError(
        "No valid instances collected. Check paths and data assumptions."
    )

# ----------------------------
# Build bins for each metric
# ----------------------------
metrics = {
    "size_of_patch": {iid: m["size_of_patch"] for iid, m in key_info.items()},
    "pre_edit_workload_runtime": {
        iid: m["pre_edit_workload_runtime"] for iid, m in key_info.items()
    },
    "improvement": {iid: m["improvement"] for iid, m in key_info.items()},
}

bins_by_metric = {}
bin_index_by_metric = {}
for name, mapping in metrics.items():
    bins, idx_map = _assign_bins(mapping, NUM_BINS)
    bins_by_metric[name] = bins
    bin_index_by_metric[name] = idx_map

# ----------------------------
# Allocate & sample per metric
# ----------------------------
all_ids = list(key_info.keys())
N = min(TARGET_N, len(all_ids))
per_metric_quota = [N // 3, N // 3, N - 2 * (N // 3)]  # sums to N

selected_set = set()

for quota, name in zip(
    per_metric_quota, ["size_of_patch", "pre_edit_workload_runtime", "improvement"]
):
    bin_counts = [len(b) for b in bins_by_metric[name]]
    alloc = _multinomial_allocation(bin_counts, quota, rng)
    picks = _sample_from_bins(bins_by_metric[name], alloc, rng)
    selected_set.update(picks)

# ----------------------------
# Top-up phase (if union < N)
# Use inverse-density weighting across the 3D bin signature to promote rare combos.
# ----------------------------
if len(selected_set) < N:
    # Compute 3D signature (bin indices across all three metrics) per instance
    signatures = {}
    for iid in all_ids:
        b0 = bin_index_by_metric["size_of_patch"][iid]
        b1 = bin_index_by_metric["pre_edit_workload_runtime"][iid]
        b2 = bin_index_by_metric["improvement"][iid]
        signatures[iid] = (b0, b1, b2)

    weights = _inverse_density_weights(signatures, bins_by_metric)
    remaining = np.array([iid for iid in all_ids if iid not in selected_set])
    if remaining.size > 0:
        w = np.array([weights[iid] for iid in remaining], dtype=float)
        if w.sum() == 0:
            w = None  # fall back to uniform
        needed = N - len(selected_set)
        needed = min(needed, remaining.size)
        extra = rng.choice(
            remaining,
            size=needed,
            replace=False,
            p=(w / w.sum()) if w is not None else None,
        ).tolist()
        selected_set.update(extra)

# If we somehow overshot (shouldn't happen), trim randomly
if len(selected_set) > N:
    selected = rng.choice(list(selected_set), size=N, replace=False).tolist()
else:
    selected = list(selected_set)

# ----------------------------
# Output
# ----------------------------
selected_key_info = {iid: key_info[iid] for iid in selected}

# Save IDs and metrics to disk for reproducibility
out_dir = Path("analysis/adhoc/make_lite/lite_selection")
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "selected_instance_ids.json", "w") as f:
    json.dump(selected, f, indent=2)

with open(out_dir / "selected_metrics.json", "w") as f:
    json.dump(selected_key_info, f, indent=2)

# Print a brief summary
print(f"Selected {len(selected)} instances (target {N}, total pool {len(all_ids)})")
print(
    f"Saved to: {out_dir/'selected_instance_ids.json'} and {out_dir/'selected_metrics.json'}"
)

# Optional: show per-metric selected coverage by bin (coarse sanity check)
for name in ["size_of_patch", "pre_edit_workload_runtime", "improvement"]:
    bins = bins_by_metric[name]
    idx_map = bin_index_by_metric[name]
    sel_bins = [idx_map[iid] for iid in selected if iid in idx_map]
    hist = np.bincount(sel_bins, minlength=NUM_BINS)
    print(f"[{name}] selected-per-bin:", hist.tolist())

# Filter dataset to selected instances and save as a new dataset (with split "lite")
lite_ds = ds.filter(lambda example: example["instance_id"] in selected)

# Make sure if current swefficiency_lite exists, we use the same instance_ids
try:
    existing = load_dataset("swefficiency/swefficiency_lite", split="test")
    existing_ids = set(d["instance_id"] for d in existing)
    new_ids = set(d["instance_id"] for d in lite_ds)
    if existing_ids != new_ids:
        raise ValueError(
            "Instance IDs in existing swefficiency_0lite differ from newly selected IDs."
        )
except Exception:
    if "swefficiency_lite" in os.environ.get("HF_DATASETS_OFFLINE", ""):
        raise

# Upload to Hugging Face Hub
lite_ds.push_to_hub("swefficiency/swefficiency_lite", split="test", private=False)
