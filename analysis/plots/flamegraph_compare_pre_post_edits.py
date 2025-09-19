#!/usr/bin/env python3
"""
Flame Graph Compare: Pre-edit vs Expert Post-edit vs LLM Post-edit

This script renders three aligned flame graphs from cProfile .prof files and
colors boxes by whether a file was attended by the Expert, the LLM, both, or
neither. It focuses on the subtree rooted at (*/workload.py, workload).

Key features
- Left: pre-edit flame graph for the workload subtree only.
- Right (top): expert post-edit, reusing the same color mapping.
- Right (bottom): LLM post-edit, reusing the same color mapping.
- Consistent x-scale across panels using *pre-edit* width.
- Depth limit via --max-depth (default 5).
- Stable sibling ordering by label (or time) for better visual comparability.
- Optional per-call normalization via --single-call to mimic a single
  invocation of workload() when your driver repeats it many times.

Usage
------
python flamegraph_compare_pre_post_edits.py \
  --instance pandas-dev__pandas-52381 \
  --gold-root logs/run_evaluation/profile_runs/gold \
  --llm-root  logs/run_evaluation/profile_runs/default_sweperf_claude__anthropic--claude-3-7-sonnet-20250219__t-0.00__p-1.00__c-1.00___swefficiency_full_test \
  --expert-files path/to/expert_files.txt \
  --llm-files    path/to/llm_files.txt \
  --max-depth 5 \
  --single-call \
  --child-sort label \
  --prune-frac 0.05 \
  --out flame_compare.png

The expert/LLM file lists can be absolute, relative, or basenames; matching is
flexible (exact, suffix, or basename).
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import pstats
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable, Set

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FuncKey = Tuple[str, int, str]  # (filename, lineno, funcname)

# -----------------------------
# Helpers for file categorization
# -----------------------------

def _normalize_path(p: str) -> str:
    try:
        return os.path.normpath(p)
    except Exception:
        return p

class PathMatcher:
    """Flexible path membership test against an attended-files list."""
    def __init__(self, paths: Iterable[str]):
        cleaned: Set[str] = set()
        for raw in paths:
            s = raw.strip()
            if not s:
                continue
            cleaned.add(_normalize_path(s))
        self.paths = cleaned
        self.basenames = {os.path.basename(p) for p in cleaned}

    def contains(self, candidate: str) -> bool:
        if not candidate:
            return False
        c = _normalize_path(candidate)
        b = os.path.basename(c)
        if b in self.basenames:
            return True
        for p in self.paths:  # suffix or reverse-suffix match
            if c.endswith(p) or p.endswith(c):
                return True
        return c in self.paths

# -----------------------------
# PStats → callee map
# -----------------------------

def load_stats(path: Path) -> pstats.Stats:
    st = pstats.Stats(str(path))
    return st

@dataclass
class Edge:
    parent: FuncKey
    child: FuncKey
    calls: int
    tot_time: float  # edge tottime (caller→callee)
    cum_time: float  # edge cumtime (caller→callee)

@dataclass
class Node:
    func: FuncKey
    tt: float
    ct: float
    cc: int
    nc: int
    edges: List[Edge] = field(default_factory=list)


def build_callee_map(stats: pstats.Stats) -> Tuple[Dict[FuncKey, Node], Dict[FuncKey, Dict[FuncKey, Edge]]]:
    """Create Node objects and a parent→child map with per-edge times.

    stats.stats[func] = (cc, nc, tt, ct, callers)
    callers: Dict[caller_func, (callcount, reccalls, tottime, cumtime)]
    We invert callers to callee map.
    """
    nodes: Dict[FuncKey, Node] = {}
    callee_map: Dict[FuncKey, Dict[FuncKey, Edge]] = defaultdict(dict)

    for func, data in stats.stats.items():
        cc, nc, tt, ct, callers = data
        nodes[func] = Node(func=func, tt=tt, ct=ct, cc=cc, nc=nc)

    for child, data in stats.stats.items():
        cc, nc, tt, ct, callers = data
        for parent, edge_vals in callers.items():
            callcount = int(edge_vals[0]) if len(edge_vals) > 0 else 0
            tottime = float(edge_vals[-2]) if len(edge_vals) >= 2 else 0.0
            cumtime = float(edge_vals[-1]) if len(edge_vals) >= 1 else 0.0
            e = Edge(parent=parent, child=child, calls=callcount, tot_time=tottime, cum_time=cumtime)
            callee_map[parent][child] = e

    for p, chmap in callee_map.items():
        if p in nodes:
            nodes[p].edges = list(chmap.values())

    return nodes, callee_map

# -----------------------------
# Find workload root
# -----------------------------

def is_workload(func: FuncKey) -> bool:
    filename, lineno, funcname = func
    return funcname == "workload" and os.path.basename(filename) == "workload.py"


def find_workload_root(stats: pstats.Stats) -> Optional[FuncKey]:
    candidates = [f for f in stats.stats.keys() if is_workload(f)]
    if candidates:
        candidates.sort(key=lambda f: stats.stats[f][3], reverse=True)
        return candidates[0]
    approx = [f for f in stats.stats.keys() if f[2] == "workload" and "workload.py" in f[0]]
    if approx:
        approx.sort(key=lambda f: stats.stats[f][3], reverse=True)
        return approx[0]
    return None

# -----------------------------
# Render structures
# -----------------------------
@dataclass
class RenderBox:
    x: float
    y: int
    w: float
    h: float
    label: str
    color_key: str
    file: str
    func: FuncKey


COLOR_MAP = {
    "expert": "#1b9e77",  # teal
    "llm":    "#d95f02",  # orange
    "both":   "#7570b3",  # purple
    "neither":"#e6e6e6",  # light gray
}

class FlameBuilder:
    def __init__(self, nodes: Dict[FuncKey, Node], callee_map: Dict[FuncKey, Dict[FuncKey, Edge]],
                 expert: PathMatcher, llm: PathMatcher, *, per_call: bool, child_sort: str = "label"):
        self.nodes = nodes
        self.callee_map = callee_map
        self.expert = expert
        self.llm = llm
        self.per_call = per_call
        self.child_sort = child_sort

    def _effective_ct(self, f: FuncKey) -> float:
        n = self.nodes.get(f)
        if not n:
            return 0.0
        if not self.per_call:
            return n.ct
        return n.ct / max(1, n.nc)

    def color_key_for_file(self, file_path: str) -> str:
        e = self.expert.contains(file_path)
        l = self.llm.contains(file_path)
        if e and l:
            return "both"
        if e:
            return "expert"
        if l:
            return "llm"
        return "neither"

    def build_boxes(self, root: FuncKey, y0: int = 0, x0: float = 0.0,
                    width_override: Optional[float] = None,
                    max_depth: int = 64,
                    min_box_width: float = 1e-6,
                    verbose: bool = False) -> Tuple[List[RenderBox], float, int]:
        boxes: List[RenderBox] = []
        maxy = 0

        def walk(func: FuncKey, x: float, y: int, avail_w: float, depth: int) -> float:
            nonlocal boxes, maxy
            maxy = max(maxy, y)
            filename, lineno, fname = func
            label = f"{fname} — {os.path.basename(filename)}:{lineno}"
            color_key = self.color_key_for_file(filename)
            if avail_w < min_box_width:
                return x
            boxes.append(RenderBox(x=x, y=y, w=avail_w, h=1.0, label=label,
                                   color_key=color_key, file=filename, func=func))
            if depth >= max_depth - 1:
                return x + avail_w

            # Children: use fractions of parent aggregate ct to partition width
            parent_ct_agg = self.nodes[func].ct if func in self.nodes else 0.0
            edges = list(self.callee_map.get(func, {}).values())
            if self.child_sort == "time":
                edges.sort(key=lambda e: e.cum_time, reverse=True)
            else:  # stable by label
                edges.sort(key=lambda e: (e.child[2], os.path.basename(e.child[0]), e.child[1]))

            used = 0.0
            for e in edges:
                ch = e.child
                if parent_ct_agg <= 0.0:
                    continue
                ratio = max(0.0, e.cum_time) / parent_ct_agg
                w_scaled = avail_w * ratio
                if w_scaled < min_box_width:
                    continue
                ch_x = x + used
                walk(ch, ch_x, y + 1, w_scaled, depth + 1)
                used += w_scaled

            # Exclusive time remainder
            excl = max(0.0, avail_w - used)
            if excl >= min_box_width:
                pseudo = (filename, lineno, f"{fname} [self]")
                label2 = f"{fname} [self] — {os.path.basename(filename)}:{lineno}"
                boxes.append(RenderBox(x=x + used, y=y + 1, w=excl, h=1.0, label=label2,
                                       color_key=color_key, file=filename, func=pseudo))
            return x + avail_w

        root_w = self._effective_ct(root)
        if width_override is not None:
            root_w = width_override
        walk(root, x0, y0, root_w, depth=0)
        depth = maxy - y0 + 1
        if verbose:
            print(f"Depth: {depth}, total width: {root_w:.6f}s, boxes: {len(boxes)}")
        return boxes, root_w, depth

# -----------------------------
# Rendering helpers
# -----------------------------

def draw_flame(ax, boxes: List[RenderBox], title: str, xlim: float,
               show_labels: bool = True, label_min_frac: float = 0.03):
    ax.set_title(title, fontsize=12, pad=6)
    ax.set_xlim(0, max(1e-9, xlim))
    max_y = 0
    for b in boxes:
        rect = mpatches.Rectangle((b.x, b.y), b.w, b.h, linewidth=0.3,
                                  edgecolor="black", facecolor=COLOR_MAP.get(b.color_key, "#cccccc"))
        ax.add_patch(rect)
        max_y = max(max_y, b.y + 1)
        if show_labels and b.w >= label_min_frac * xlim:
            cx = b.x + b.w / 2.0
            cy = b.y + b.h / 2.0
            ax.text(cx, cy, b.label, ha="center", va="center", fontsize=7, clip_on=True)
    ax.set_ylim(0, max_y + 0.5)
    ax.invert_yaxis()  # classic flame (root at bottom)
    ax.set_yticks([])
    ax.set_xlabel("seconds (cumulative)")


def legend_fig(fig):
    patches = [mpatches.Patch(color=COLOR_MAP[k], label=k) for k in ["expert", "llm", "both", "neither"]]
    fig.legend(handles=patches, loc="lower center", ncol=4, frameon=False)

# -----------------------------
# Main driver
# -----------------------------

def read_list_file(path: Optional[str]) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        print(f"[warn] list file not found: {p}")
        return []
    return [line.rstrip("\n") for line in p.read_text(encoding="utf-8").splitlines()]


def resolve_profile_paths(args) -> Tuple[Path, Path, Path, Path]:
    base_gold = Path(args.gold_root)
    base_llm = Path(args.llm_root)
    inst = args.instance
    gold_pre = base_gold / inst / "workload_preedit_cprofile.prof"
    gold_post = base_gold / inst / "workload_postedit_cprofile.prof"
    llm_pre = base_llm / inst / "workload_preedit_cprofile.prof"
    llm_post = base_llm / inst / "workload_postedit_cprofile.prof"
    return gold_pre, gold_post, llm_pre, llm_post


def seconds_fmt(s: float) -> str:
    return f"{s:.3f}s"


def main():
    ap = argparse.ArgumentParser(description="Render pre/post flame graphs with expert/LLM overlays")
    ap.add_argument("--instance", type=str, required=True,
                    help="Instance name, e.g., pandas-dev__pandas-52381")
    ap.add_argument("--gold-root", type=str, required=True,
                    help="Path to GOLD profile dir (contains <instance>/preedit & postedit)")
    ap.add_argument("--llm-root", type=str, required=True,
                    help="Path to LLM profile dir (contains <instance>/preedit & postedit)")
    ap.add_argument("--expert-files", type=str, default=None,
                    help="Newline-delimited file list for expert-attended files")
    ap.add_argument("--llm-files", type=str, default=None,
                    help="Newline-delimited file list for LLM-attended files")
    ap.add_argument("--out", type=str, default="flame_compare.png")
    ap.add_argument("--figwidth", type=float, default=16.0)
    ap.add_argument("--figheight", type=float, default=9.0)
    ap.add_argument("--max-depth", type=int, default=5,
                    help="Maximum flame depth to draw (levels including root; default 5)")
    ap.add_argument("--single-call", action="store_true",
                    help="Normalize widths to one workload() invocation (per-call)")
    ap.add_argument("--child-sort", choices=["label", "time"], default="label",
                    help="Order siblings by label (stable) or by descending time")
    ap.add_argument("--prune-frac", type=float, default=0.05,
                    help="Prune boxes narrower than this fraction of the expert improvement (default 0.05). Set 0 to disable.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    gold_pre, gold_post, llm_pre, llm_post = resolve_profile_paths(args)

    # Load stats
    st_pre = load_stats(gold_pre)
    st_exp = load_stats(gold_post)
    st_lpre = load_stats(llm_pre)  # not used for layout; kept for parity if needed later
    st_llm = load_stats(llm_post)

    # Roots
    root_pre = find_workload_root(st_pre)
    if not root_pre:
        raise SystemExit("Could not find workload() in pre-edit stats")

    nodes_pre, cmap_pre = build_callee_map(st_pre)
    nodes_exp, cmap_exp = build_callee_map(st_exp)
    nodes_llm, cmap_llm = build_callee_map(st_llm)

    root_exp = root_pre if root_pre in nodes_exp else find_workload_root(st_exp)
    root_llm = root_pre if root_pre in nodes_llm else find_workload_root(st_llm)
    if not root_exp:
        raise SystemExit("Could not find workload() in expert post-edit stats")
    if not root_llm:
        raise SystemExit("Could not find workload() in LLM post-edit stats")

    # Attended file matchers
    expert_match = PathMatcher(read_list_file(args.expert_files))
    llm_match = PathMatcher(read_list_file(args.llm_files))

    # Builders
    fb_pre = FlameBuilder(nodes_pre, cmap_pre, expert_match, llm_match,
                          per_call=args.single_call, child_sort=args.child_sort)
    fb_exp = FlameBuilder(nodes_exp, cmap_exp, expert_match, llm_match,
                          per_call=args.single_call, child_sort=args.child_sort)
    fb_llm = FlameBuilder(nodes_llm, cmap_llm, expert_match, llm_match,
                          per_call=args.single_call, child_sort=args.child_sort)

    # Determine widths and a common x-axis limit (same for all panels)
    pre_root_w = fb_pre._effective_ct(root_pre)
    exp_root_w = fb_exp._effective_ct(root_exp)
    llm_root_w = fb_llm._effective_ct(root_llm)
    xlim_ref = max(pre_root_w, exp_root_w, llm_root_w)

    # Pruning threshold based on expert improvement
    improvement = max(0.0, pre_root_w - exp_root_w)
    prune_abs = improvement * max(0.0, args.prune_frac)

    boxes_pre, _, _ = fb_pre.build_boxes(root_pre, y0=0, x0=0.0,
                                         width_override=None,
                                         max_depth=args.max_depth,
                                         min_box_width=prune_abs,
                                         verbose=args.verbose)
    boxes_exp, _, _ = fb_exp.build_boxes(root_exp, y0=0, x0=0.0,
                                         width_override=None,
                                         max_depth=args.max_depth,
                                         min_box_width=prune_abs,
                                         verbose=args.verbose)
    boxes_llm, _, _ = fb_llm.build_boxes(root_llm, y0=0, x0=0.0,
                                         width_override=None,
                                         max_depth=args.max_depth,
                                         min_box_width=prune_abs,
                                         verbose=args.verbose)

    # Figure layout
    fig = plt.figure(figsize=(args.figwidth, args.figheight))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], height_ratios=[1.0, 1.0], wspace=0.2, hspace=0.25)

    ax_left = fig.add_subplot(gs[:, 0])
    ax_rt = fig.add_subplot(gs[0, 1])
    ax_rb = fig.add_subplot(gs[1, 1])

    # Titles (per-call aware)
    def title_for(nodes: Dict[FuncKey, Node], root: FuncKey, label: str) -> str:
        n = nodes[root]
        val = (n.ct / max(1, n.nc)) if args.single_call else n.ct
        delta = val - pre_root_w
        suffix = " (per-call)" if args.single_call else ""
        return f"{label}{suffix}: {seconds_fmt(val)} (Δ {seconds_fmt(delta)})" if label != "Pre-edit workload" else f"{label}{suffix}: {seconds_fmt(val)}"

    t_pre = title_for(nodes_pre, root_pre, "Pre-edit workload")
    t_exp = title_for(nodes_exp, root_exp, "Expert post-edit")
    t_llm = title_for(nodes_llm, root_llm, "LLM post-edit")

    draw_flame(ax_left, boxes_pre, title=t_pre, xlim=xlim_ref)
    draw_flame(ax_rt, boxes_exp, title=t_exp, xlim=xlim_ref)
    draw_flame(ax_rb, boxes_llm, title=t_llm, xlim=xlim_ref)

    for ax in (ax_left, ax_rt, ax_rb):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(False)

    legend_fig(fig)
    fig.suptitle("Workload Flame Graphs — Expert vs LLM edits (colors = attended files)", fontsize=14, y=0.98)

    out = Path(args.out)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved → {out.resolve()}")


if __name__ == "__main__":
    main()
