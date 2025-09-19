from pathlib import Path

import datasets
import numpy as np
import tqdm

GOLD_RUN = Path("logs/run_evaluation/ground_truth5/gold")
SONNET_RUN = Path(
    "logs/run_evaluation/ground_truth5/us.anthropic.claude-3-7-sonnet-20250219-v1_0_maxiter_100_N_v0.51.1-no-hint-run_1"
)

ds = datasets.load_dataset("swefficiency/swefficiency", split="test")


def read_text(p: Path) -> str:
    if p.exists():
        with open(p, "r") as f:
            return f.read()
    return ""


def parse_modified_files(patch_text: str) -> set[str]:
    """
    Parse file paths from `diff --git a/<old> b/<new>` lines.
    Prefer the 'b/<new>' path so renames are attributed to the new filename.
    """
    files = set()
    for line in patch_text.splitlines():
        if line.startswith("diff --git"):
            # Expected tokens: diff --git a/foo b/foo
            parts = line.split()
            if len(parts) >= 4:
                bpath = parts[3]
                if bpath.startswith("b/"):
                    bpath = bpath[2:]
                files.add(bpath)
    return files


macro_jaccards = []
macro_precisions = []
macro_recalls = []
macro_f1s = []
exact_matches = 0
any_overlap = 0

sum_intersections = 0
sum_unions = 0

evaluated = 0
skipped = 0  # if both empty (shouldn't happen for gold, but be robust)

from collections import defaultdict

file_extensions = set()
keep_extensions = {
    ".py",
    ".ini",
    ".f",
    ".pyx",
    ".h",
    ".build",
    ".cfg",
    ".cpp",
    ".json",
    ".pyi",
    ".pxd",
    ".yaml",
    ".toml",
    ".cxx",
}
extension_counter = defaultdict(int)

for d in tqdm.tqdm(ds, total=len(ds), desc="Computing file-wise metrics"):
    instance_id = d["instance_id"]

    gold_patch = read_text(GOLD_RUN / instance_id / "patch.diff")
    pred_patch = read_text(SONNET_RUN / instance_id / "patch.diff")

    G = parse_modified_files(gold_patch)
    P = parse_modified_files(pred_patch)

    # Keep certain extensions.
    G = {f for f in G if f.endswith(tuple(keep_extensions))}
    P = {f for f in P if f.endswith(tuple(keep_extensions))}

    # Track counter for extension changes.
    for f in G.union(P):
        extension_counter[Path(f).suffix] += 1

    # If both are empty, skip (no edit signal)
    if not G and not P:
        skipped += 1
        continue

    evaluated += 1
    inter = G & P
    union = G | P

    # Hit / non-coincidence
    if len(inter) > 0:
        any_overlap += 1

    # Exact set match (ignore the degenerate both-empty case above)
    if G == P and G:
        exact_matches += 1

    # Jaccard
    j = (len(inter) / len(union)) if union else 1.0
    macro_jaccards.append(j)

    # Precision / Recall / F1 (macro)
    if P:
        prec = len(inter) / len(P)
    else:
        prec = 0.0 if G else 1.0  # if both empty, we'd have skipped

    if G:
        rec = len(inter) / len(G)
    else:
        rec = 1.0 if not P else 0.0

    if prec + rec > 0:
        f1 = 2 * prec * rec / (prec + rec)
    else:
        f1 = 0.0

    macro_precisions.append(prec)
    macro_recalls.append(rec)
    macro_f1s.append(f1)

    # Micro (weighted) Jaccard components
    sum_intersections += len(inter)
    sum_unions += len(union)

print(extension_counter)

# Aggregate
hit_rate = any_overlap / evaluated if evaluated else 0.0
non_coincidence_rate = 1.0 - hit_rate
exact_match_rate = exact_matches / evaluated if evaluated else 0.0

macro_jacc = float(np.sum(macro_jaccards) / len(ds)) if macro_jaccards else 0.0
macro_prec = float(np.sum(macro_precisions) / len(ds)) if macro_precisions else 0.0
macro_rec = float(np.sum(macro_recalls) / len(ds)) if macro_recalls else 0.0
macro_f1 = float(np.sum(macro_f1s) / len(ds)) if macro_f1s else 0.0
micro_jacc = (sum_intersections / sum_unions) if sum_unions else 0.0

print(f"Evaluated instances: {evaluated} (skipped: {skipped})")
print(f"Non-coincidence rate (zero file overlap): {non_coincidence_rate:.3f}")
print(f"Hit rate (any overlap): {hit_rate:.3f}")
print(f"Exact file-set match rate: {exact_match_rate:.3f}")
print(f"Macro Jaccard: {macro_jacc:.3f}")
print(f"Micro (weighted) Jaccard: {micro_jacc:.3f}")
print(
    f"Macro Precision / Recall / F1: {macro_prec:.3f} / {macro_rec:.3f} / {macro_f1:.3f}"
)
