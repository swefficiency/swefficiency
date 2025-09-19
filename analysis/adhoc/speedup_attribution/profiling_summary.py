#!/usr/bin/env python3
"""
Minimal profiling analysis for:
  1) file_overlap_jaccard
  2) ERC (Expert-Relative Coverage): file-level & function-level
  3) LossDecomposition: WrongFileLoss, InFileLoss

Noise control:
- improvement_threshold_frac_of_expert (default 0.02 = 2% of expert total gain)
- improvement_threshold_seconds (absolute floor, default 0.0)
- improvement_cumratio_cap (fraction of total positive gain to keep, default 1.0)

Inputs:
- cProfile .prof files for expert and LLM, pre- and post-edit.

Outputs:
- JSONL with metrics per instance_id.

Author: cleaned & minimized per user request.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Set, Iterable, Optional, Any, List
from collections import defaultdict
from pathlib import Path
import os
import pstats
import json

FuncKey = Tuple[str, int, str]  # (file, line, funcname)

from typing import Optional, Set

def _parse_modified_files_from_patch(patch_text: Optional[str]) -> Set[str]:
    """
    Extract a set of normalized file paths modified in a unified diff.
    Normalization matches _normalize_file: last two path components, '/' separators.
    """
    files: Set[str] = set()
    if not patch_text:
        return files

    for line in patch_text.splitlines():
        line = line.strip()

        # Lines like: "diff --git a/path/to/file.py b/path/to/file.py"
        if line.startswith("diff --git "):
            parts = line.split()
            # Expect the last two tokens to be a/<path> b/<path>
            for tok in parts[-2:]:
                if tok.startswith("a/") or tok.startswith("b/"):
                    path = tok[2:]
                    files.add(_normalize_file(path))
            continue

        # Lines like: "+++ b/path/to/file.py" or "--- a/path/to/file.py"
        if line.startswith("+++") or line.startswith("---"):
            path = line[4:].strip()
            if path == "/dev/null":
                continue
            if path.startswith("a/") or path.startswith("b/"):
                path = path[2:]
            files.add(_normalize_file(path))

    return files


# ----------------------------
# Loading & basic structures
# ----------------------------

@dataclass
class FuncStat:
    ncalls: int
    tottime: float
    cumtime: float

def _normalize_file(path: str) -> str:
    """Normalize to last two path components, forward slashes."""
    if not path:
        return path
    path = os.path.normpath(path)
    parts = path.split(os.sep)
    tail = os.sep.join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    return tail.replace("\\", "/")

def _load_pstats(prof_path: str) -> Dict[FuncKey, FuncStat]:
    """Load a cProfile .prof file into a dict of FuncKey -> FuncStat,
    restricted to the subtree rooted at (file='tmp/workload.py', fname='workload')."""
    st = pstats.Stats(prof_path)

    # Build caller -> set(callee) adjacency from the callers info in st.stats
    caller_to_callees: Dict[Tuple[str, int, str], Set[Tuple[str, int, str]]] = {}
    for func, data in st.stats.items():
        cc, nc, tt, ct, callers = data
        for caller in (callers or {}):
            caller_to_callees.setdefault(caller, set()).add(func)

    # Identify root(s): match by normalized file and function name (line can vary)
    root_file_norm = _normalize_file("tmp/workload.py")
    root_fname = "workload"
    roots = {
        func for func in st.stats.keys()
        if _normalize_file(func[0]) == root_file_norm and func[2] == root_fname
    }
    if not roots:
        return {}

    # DFS/BFS to collect the reachable subtree from the root(s)
    reachable: Set[Tuple[str, int, str]] = set()
    stack = list(roots)
    while stack:
        f = stack.pop()
        if f in reachable:
            continue
        reachable.add(f)
        for cal in caller_to_callees.get(f, ()):
            if cal not in reachable:
                stack.append(cal)

    # Emit stats only for the reachable functions (including the root)
    out: Dict[FuncKey, FuncStat] = {}
    for func in reachable:
        # Some callers may reference functions not present in st.stats (rare). Guard for safety.
        if func not in st.stats:
            continue
        file, line, fname = func
        file_norm = _normalize_file(file)
        _cc, nc, tt, ct, _callers = st.stats[func]
        out[(file_norm, int(line), str(fname))] = FuncStat(
            ncalls=int(nc), tottime=float(tt), cumtime=float(ct)
        )
    return out

# ----------------------------
# Deltas & selection
# ----------------------------

def _delta_tottime(pre: Dict[FuncKey, FuncStat],
                   post: Dict[FuncKey, FuncStat]) -> Dict[FuncKey, float]:
    """Δtottime (pre - post) per function; positive means faster."""
    keys = set(pre.keys()) | set(post.keys())
    d: Dict[FuncKey, float] = {}
    for k in keys:
        pr = pre.get(k, FuncStat(0, 0.0, 0.0)).tottime
        po = post.get(k, FuncStat(0, 0.0, 0.0)).tottime
        d[k] = pr - po
    return d

def _delta_cumtime(pre: Dict[FuncKey, FuncStat],
                   post: Dict[FuncKey, FuncStat]) -> Dict[FuncKey, float]:
    """Δcumtime (pre - post) per function; positive means faster."""
    keys = set(pre.keys()) | set(post.keys())
    d: Dict[FuncKey, float] = {}
    for k in keys:
        pr = pre.get(k, FuncStat(0, 0.0, 0.0)).cumtime
        po = post.get(k, FuncStat(0, 0.0, 0.0)).cumtime
        d[k] = pr - po
    return d

def _get_workload_cumtime_delta(pre: Dict[FuncKey, FuncStat], post: Dict[FuncKey, FuncStat]) -> float:
    """Return Δcumtime of function named 'workload', or 0.0 if not found."""
    
    for k, stat in pre.items():
        if k[0] == "tmp/workload.py" and k[2] == "workload":
            pre_cum = stat.cumtime
            post_cum = post.get(k, FuncStat(0, 0.0, 0.0)).cumtime
            return pre_cum - post_cum
    return 0.0


def _get_workload_pre_cumtime(pre: Dict[FuncKey, FuncStat]) -> float:
    """Return PRE cumtime of function named 'workload', or 0.0 if not found."""
    for k, stat in pre.items():
        if k[0] == "tmp/workload.py" and k[2] == "workload":
            return stat.cumtime
    return 0.0

def _select_functions(
    pre: Dict[FuncKey, FuncStat],
    post: Dict[FuncKey, FuncStat],
    prof_path_for_graph: str,
    improvement_threshold_abs: float,
    improvement_cumratio_cap: float = 1.0,
    allowed_files: Optional[Set[str]] = None,
) -> Set[FuncKey]:
    """
    Select 'core' functions by Δcumtime (pre - post), digging as deep as possible
    in the call tree rooted at (tmp/workload.py, workload) to avoid double-counting
    parent improvements.

    Strategy:
      1) Build callers_map from the *pre* profile, derive depths from 'workload'
         (caller distance), and restrict to nodes reachable from 'workload'.
      2) Candidates = reachable funcs with positive Δcumtime >= threshold
         (and in allowed_files if provided).
      3) Sort candidates by:
            - deeper depth first (prefer deepest),
            - larger % of overall workload time (tie-break),
            - larger absolute Δcumtime (final tie-break).
      4) Greedily pick a candidate unless any of its ancestors is already selected.
         After picking a node, mark *all* its ancestors as banned to avoid
         selecting callers of this deepest improvement.
      5) Stop when selected cumulative Δcumtime reaches the cap fraction.

    Notes:
      * This uses the PRE graph to determine ancestry/depths (stable choice).
      * "% of overall time" uses PRE workload cumtime as denominator.
    """
    # Δcumtime map
    delta_cum = _delta_cumtime(pre, post)

    # Build call graph + depths from 'workload'
    callers_map = _load_callers_map(prof_path_for_graph)  # callee -> set(callers)
    depths, found_wl = _compute_depths_from_named_roots(callers_map, {"workload"})

    # If we couldn't find 'workload' by name, fall back to using everything with depth 0
    reachable_nodes: Set[FuncKey] = set(depths.keys()) if found_wl else set(callers_map.keys())
    if not reachable_nodes:
        reachable_nodes = set(delta_cum.keys())  # last-ditch fallback

    # PRE workload cumtime (for % tie-breaks)
    workload_key = None
    for k in pre.keys():
        if k[0] == "tmp/workload.py" and k[2] == "workload":
            workload_key = k
            break
    workload_pre_cum = pre.get(workload_key, FuncStat(0, 0.0, 0.0)).cumtime if workload_key else 0.0
    denom = workload_pre_cum if workload_pre_cum > 0.0 else 1e-12

    # Allowed files filter set (if provided)
    def _allowed(k: FuncKey) -> bool:
        return (allowed_files is None) or (k[0] in allowed_files)

    # Candidates: positive Δcumtime above threshold, reachable from workload, file-allowed
    cands: List[Tuple[FuncKey, float]] = [
        (k, d)
        for k, d in delta_cum.items()
        if d > 0.0
        and d >= improvement_threshold_abs
        and k in reachable_nodes
        and _allowed(k)
    ]
    if not cands:
        return set()

    # Total positive mass for capping
    total_pos = sum(d for _k, d in cands) or 1e-12
    cap_mass = max(0.0, min(1.0, improvement_cumratio_cap)) * total_pos

    # Convenience reverse edges (caller ancestry traversal)
    # callers_map: callee -> set(callers)
    def ancestors_of(node: FuncKey) -> Set[FuncKey]:
        """All (transitive) callers of 'node' within the reachable subgraph."""
        out: Set[FuncKey] = set()
        stack = list(callers_map.get(node, ()))
        while stack:
            cur = stack.pop()
            if cur in out:
                continue
            if cur in reachable_nodes:  # keep ancestry within the same component
                out.add(cur)
                stack.extend(callers_map.get(cur, ()))
        return out

    # Precompute depths (default 0 for nodes missing depth if workload not found)
    def depth_of(k: FuncKey) -> int:
        return depths.get(k, 0)

    # Compute percent-of-overall improvement for tie-breaks
    def share_of(k: FuncKey, d: float) -> float:
        return d / denom

    # Sort: deeper first, then higher share, then higher absolute Δcumtime
    cands.sort(key=lambda kv: (depth_of(kv[0]), share_of(kv[0], kv[1]), kv[1]), reverse=True)

    selected: Set[FuncKey] = set()
    banned_ancestors: Set[FuncKey] = set()  # callers of any selected node
    acc = 0.0

    for k, d in cands:
        if k in banned_ancestors:
            continue
        # select k
        selected.add(k)
        acc += d

        # ban all of k's ancestors (callers) to avoid double-counting parent improvements
        banned_ancestors.update(ancestors_of(k))

        # cap check
        if acc >= cap_mass - 1e-12:
            break

    # Ensure at least the single best candidate is included if nothing selected (edge case)
    if not selected and cands:
        selected.add(cands[0][0])

    return selected


# ----------------------------
# Metrics
# ----------------------------

def _build_file_set(funcs: Iterable[FuncKey]) -> Set[str]:
    return {k[0] for k in funcs}

def _erc_and_losses(s_exp: Dict[FuncKey, float], E_llm: Set[FuncKey]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    ERC_file/func computed over EXPERT mass defined as Δcumtime improvements.
    ERC_file: fraction of expert total Δcumtime gain within files chosen by LLM.
    ERC_func: fraction of expert total Δcumtime gain within functions chosen by LLM.
    Losses: WrongFileLoss = 1 - ERC_file
            InFileLoss   = max(0, ERC_file - ERC_func)
    """ 
    S_exp = sum(s_exp.values()) or 1e-12

    # Expert mass by file
    s_exp_by_file = defaultdict(float)
    for k, mass in s_exp.items():
        s_exp_by_file[k[0]] += mass

    # Files/functions chosen by LLM
    llm_files = _build_file_set(E_llm)

    s_exp_in_llm_files = sum(s_exp_by_file.get(f, 0.0) for f in llm_files)
    s_exp_in_llm_funcs = sum(s_exp.get(k, 0.0) for k in E_llm)

    ERC_file = s_exp_in_llm_files / S_exp
    ERC_func = s_exp_in_llm_funcs / S_exp

    WrongFileLoss = 1.0 - ERC_file
    InFileLoss = max(0.0, ERC_file - ERC_func)

    return (
        {"file": ERC_file, "func": ERC_func},
        {"WrongFileLoss": WrongFileLoss, "InFileLoss": InFileLoss},
    )

def _file_overlap_jaccard(E_exp: Set[FuncKey], E_llm: Set[FuncKey]) -> Optional[float]:
    """
    Jaccard over the set of edited files: |F_exp ∩ F_llm| / |F_exp ∪ F_llm|
    """
    F_exp = _build_file_set(E_exp)
    F_llm = _build_file_set(E_llm)
    union = F_exp | F_llm
    if not union:
        return None
    inter = F_exp & F_llm
    return len(inter) / len(union)

### ADD YOUR CODE HERE ###

from collections import deque, defaultdict
from typing import Dict, Set, Optional, Tuple


def _load_callers_map(prof_path: str) -> Dict[FuncKey, Set[FuncKey]]:
    """
    Build a mapping: callee FuncKey -> set of caller FuncKeys from a .prof file.
    Paths are normalized to match FuncKeys elsewhere.
    """
    st = pstats.Stats(prof_path)
    callers_map: Dict[FuncKey, Set[FuncKey]] = {}

    # First pass: add callees with their callers
    for func, data in st.stats.items():
        file, line, fname = func
        key: FuncKey = (_normalize_file(file), int(line), str(fname))
        _cc, _nc, _tt, _ct, callers = data
        callee_callers: Set[FuncKey] = set()
        if callers:
            for caller_func in callers.keys():
                cfile, cline, cfname = caller_func
                callee_callers.add((_normalize_file(cfile), int(cline), str(cfname)))
        callers_map[key] = callee_callers

    # Second pass: ensure every caller that appeared is present as a node (possibly root) in the map
    for callee, cset in list(callers_map.items()):
        for caller in cset:
            if caller not in callers_map:
                callers_map[caller] = set()

    return callers_map


def _compute_depths_from_named_roots(
    callers_map: Dict[FuncKey, Set[FuncKey]],
    target_names: Iterable[str],
) -> Tuple[Dict[FuncKey, int], bool]:
    """
    BFS depths starting from any function whose name is in target_names
    (case-insensitive). Returns (depths, found_any).
    Depths map only includes nodes reachable from those sources.
    """
    names_lc = {n.lower() for n in target_names}

    # Build caller -> callee adjacency and collect all nodes
    out_edges: Dict[FuncKey, Set[FuncKey]] = defaultdict(set)
    nodes: Set[FuncKey] = set(callers_map.keys())
    for callee, callers in callers_map.items():
        for caller in callers:
            out_edges[caller].add(callee)
            nodes.add(caller)

    # Find sources named "workload"
    sources = {k for k in nodes if k[2].lower() in names_lc}
    found_any = bool(sources)
    if not found_any:
        return {}, False

    depths: Dict[FuncKey, int] = {}
    q = deque()
    for s in sources:
        depths[s] = 0
        q.append(s)

    while q:
        u = q.popleft()
        for v in out_edges.get(u, ()):
            if v not in depths:
                depths[v] = depths[u] + 1
                q.append(v)

    return depths, True


def _weighted_avg_depth_and_share(
    selected_funcs: Set[FuncKey],
    delta_map: Dict[FuncKey, float],
    depths: Dict[FuncKey, int],
) -> Tuple[Optional[float], float]:
    """
    Weighted average (by positive Δtottime) of depth over selected functions,
    but only over funcs that have a defined depth (i.e., are reachable).
    Returns (avg_depth or None, reachable_weight_share in [0,1]).
    """
    total_w = 0.0
    in_w = 0.0
    num = 0.0

    for k in selected_funcs:
        w = max(0.0, delta_map.get(k, 0.0))
        if w <= 0.0:
            continue
        total_w += w
        d = depths.get(k)
        if d is not None:
            in_w += w
            num += w * d

    if in_w <= 0.0:
        return None, 0.0
    return num / in_w, (in_w / total_w) if total_w > 0.0 else 1.0


### END YOUR CODE HERE ###


# ----------------------------
# Driver
# ----------------------------

from typing import Optional, Dict, Any, Set

def analyze_localization(
    expert_pre_path: str,
    expert_post_path: str,
    llm_pre_path: str,
    llm_post_path: str,
    improvement_threshold_seconds: float = 0.0,
    improvement_threshold_frac_of_expert: float = 0.02,  # 2% default per-function floor
    improvement_cumratio_cap: float = 1.0,
    expert_patch_text: Optional[str] = None,
    llm_patch_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes only:
      - file_overlap_jaccard
      - ERC (file, func)
      - LossDecomposition (WrongFileLoss, InFileLoss)

    Patch-based filtering:
      * expert_patch_text: unified diff for expert's edit; restricts Expert selection to those files.
      * llm_patch_text:    unified diff for LLM's edit;    restricts LLM selection to those files.
    """
    # Load profiles
    exp_pre = _load_pstats(expert_pre_path)
    exp_post = _load_pstats(expert_post_path)
    llm_pre = _load_pstats(llm_pre_path)
    llm_post = _load_pstats(llm_post_path)

    # Per-function Δtottime (for diagnostic/aux weighting) and Δcumtime (for selection + ERC)
    d_exp_tt  = _delta_tottime(exp_pre, exp_post)
    d_llm_tt  = _delta_tottime(llm_pre, llm_post)
    d_exp_cum = _delta_cumtime(exp_pre, exp_post)
    d_llm_cum = _delta_cumtime(llm_pre, llm_post)

    # Get cumulative time improvement of "workload" in expert/LLM pre->post,
    # and each actor's PRE workload cumtime for normalization.
    workload_improvement = _get_workload_cumtime_delta(exp_pre, exp_post)
    workload_improvement_llm = _get_workload_cumtime_delta(llm_pre, llm_post)
    expert_wl_pre_cum = _get_workload_pre_cumtime(exp_pre)
    llm_wl_pre_cum    = _get_workload_pre_cumtime(llm_pre)

    # Absolute floor for both actors (seconds)
    rel_part = (improvement_threshold_frac_of_expert * workload_improvement) if improvement_threshold_frac_of_expert > 0.0 else 0.0
    improvement_threshold_abs = max(improvement_threshold_seconds, rel_part)

    # Parse patch files (normalize to match FuncKey[0] normalization)
    def _maybe_files(patch_text: Optional[str]) -> Optional[Set[str]]:
        if not patch_text:
            return None
        files = _parse_modified_files_from_patch(patch_text)
        return files if files else None  # treat empty as no filter

    expert_allowed_files = _maybe_files(expert_patch_text)
    llm_allowed_files    = _maybe_files(llm_patch_text)

    # Select function sets, with per-side patch filters
    E_exp = _select_functions(
        exp_pre,                 # use PRE/POST stats to compute Δcumtime internally
        exp_post,
        expert_pre_path,         # build call graph & depths from PRE profile
        improvement_threshold_abs,
        improvement_cumratio_cap,
        allowed_files=expert_allowed_files,
    )

    E_llm = _select_functions(
        llm_pre,
        llm_post,
        llm_pre_path,
        improvement_threshold_abs,
        improvement_cumratio_cap,
        allowed_files=llm_allowed_files,
    )
    
    expert_selected_files = _build_file_set(E_exp)

    s_exp: Dict[FuncKey, float] = {
        k: d for k, d in d_exp_cum.items()
        if d >= improvement_threshold_abs and k[0] in expert_selected_files
    }
    erc_baseline_kind = "expert_selected"

    # Metrics
    jaccard_files = _file_overlap_jaccard(E_exp, E_llm)
    ERC, Losses = _erc_and_losses(s_exp, E_llm)

    exp_callers = _load_callers_map(expert_pre_path)
    llm_callers = _load_callers_map(llm_pre_path)

    # NEW: compute weighted average optimization depths from pre-edit call graphs
    exp_depths_wl, exp_wl_found = _compute_depths_from_named_roots(exp_callers, {"workload"})
    llm_depths_wl, llm_wl_found = _compute_depths_from_named_roots(llm_callers, {"workload"})

    avg_depth_expert_wl, share_exp = _weighted_avg_depth_and_share(E_exp, d_exp_tt, exp_depths_wl)

    avg_depth_llm_wl,   share_llm = _weighted_avg_depth_and_share(E_llm, d_llm_tt, llm_depths_wl)

    # Normalize depth weights by each actor's PRE workload runtime
    # to reduce run-to-run scaling variance (percent of pre-edit runtime).
    if expert_wl_pre_cum > 0:
        d_exp_tt_norm = {k: v / expert_wl_pre_cum for k, v in d_exp_tt.items()}
        avg_depth_expert_wl, share_exp = _weighted_avg_depth_and_share(E_exp, d_exp_tt_norm, exp_depths_wl)
    if llm_wl_pre_cum > 0:
        d_llm_tt_norm = {k: v / llm_wl_pre_cum for k, v in d_llm_tt.items()}
        avg_depth_llm_wl,   share_llm = _weighted_avg_depth_and_share(E_llm, d_llm_tt_norm, llm_depths_wl)

    return {
        "Improvement": {
            "preedit_runtime_expert": sum(st.tottime for st in exp_pre.values()),
            "postedit_runtime_expert": sum(st.tottime for st in exp_post.values()),
            "preedit_runtime_llm": sum(st.tottime for st in llm_pre.values()),
            "postedit_runtime_llm": sum(st.tottime for st in llm_post.values()),
            "expert_cumtime_workload": workload_improvement,
            "llm_cumtime_workload": workload_improvement_llm,
            # New: percent-of-PRE-runtime (each actor normalized by its own PRE workload time)
            "expert_workload_pre_cum": expert_wl_pre_cum,
            "llm_workload_pre_cum": llm_wl_pre_cum,
            "expert_workload_speedup_ratio_of_pre": (workload_improvement / expert_wl_pre_cum) if expert_wl_pre_cum > 0 else None,
            "llm_workload_speedup_ratio_of_pre":    (workload_improvement_llm / llm_wl_pre_cum) if llm_wl_pre_cum > 0 else None,
            # Optional: whole-program tottime speedup as percent of PRE workload runtime (same denominator choice)
            "expert_total_speedup_ratio_of_pre": (
                ((sum(st.tottime for st in exp_pre.values()) - sum(st.tottime for st in exp_post.values())) / expert_wl_pre_cum)
                if expert_wl_pre_cum > 0 else None
            ),
            "llm_total_speedup_ratio_of_pre": (
                ((sum(st.tottime for st in llm_pre.values()) - sum(st.tottime for st in llm_post.values())) / llm_wl_pre_cum)
                if llm_wl_pre_cum > 0 else None
            ),
        },
        "file_overlap_jaccard": jaccard_files,
        "Files": {
            "expert": sorted(_build_file_set(E_exp)),
            "llm": sorted(_build_file_set(E_llm)),
        },
        "ERC": ERC,
        "LossDecomposition": Losses,
        "AvgOptimizationDepthFromWorkload": {  # <-- NEW block
            "expert_pre": avg_depth_expert_wl,
            "llm_pre": avg_depth_llm_wl,
            "workload_found": {"expert": exp_wl_found, "llm": llm_wl_found},
            "reachable_weight_share": {"expert": share_exp, "llm": share_llm},
            "definition": "Minimum caller distance from any function named 'workload' in the PRE profile; weighted by Δtottime of selected improvements. Only improvements reachable from 'workload' contribute; 'reachable_weight_share' reports the weight coverage.",
        },
        "selection_params": {
            "improvement_threshold_seconds": improvement_threshold_seconds,
            "improvement_threshold_frac_of_expert": improvement_threshold_frac_of_expert,
            "effective_threshold_seconds": improvement_threshold_abs,
            "improvement_cumratio_cap": improvement_cumratio_cap,
            "expert_patch_filter_active": bool(expert_allowed_files),
            "llm_patch_filter_active": bool(llm_allowed_files),
            "erc_baseline": erc_baseline_kind,
        },
    }


# ----------------------------
# Batch wrapper & CLI-style main
# ----------------------------

# Set these once in your driver script / notebook
GOLD_RUN_DIR = Path("logs/run_evaluation/profile_runs/gold")

def wrapper(
    instance: Dict[str, Any],
    improvement_threshold_seconds: float = 0.0,
    improvement_threshold_frac_of_expert: float = 0.001,  # 2%
    improvement_cumratio_cap: float = 1.0,
) -> Optional[Dict[str, Any]]:
    instance_id = instance["instance_id"]
    
    gold_pre_path = GOLD_RUN_DIR / instance_id / "workload_preedit_cprofile.prof"
    gold_post_path = GOLD_RUN_DIR / instance_id / "workload_postedit_cprofile.prof"
    pred_pre_path = PRED_RUN_DIR / instance_id / "workload_preedit_cprofile.prof"
    pred_post_path = PRED_RUN_DIR / instance_id / "workload_postedit_cprofile.prof"
    expert_patch_text = GOLD_RUN_DIR / instance_id / "patch.diff"
    llm_patch_text = PRED_RUN_DIR / instance_id / "patch.diff"

    if not (gold_pre_path.exists() and gold_post_path.exists() and pred_pre_path.exists() and pred_post_path.exists()):
        return None

    res = analyze_localization(
        expert_pre_path=str(gold_pre_path),
        expert_post_path=str(gold_post_path),
        llm_pre_path=str(pred_pre_path),
        llm_post_path=str(pred_post_path),
        improvement_threshold_seconds=improvement_threshold_seconds,
        improvement_threshold_frac_of_expert=improvement_threshold_frac_of_expert,
        improvement_cumratio_cap=improvement_cumratio_cap,
        expert_patch_text=expert_patch_text.read_text() if expert_patch_text.exists() else None,
        llm_patch_text=llm_patch_text.read_text() if llm_patch_text.exists() else None,
    )
    new_res = {
        "instance_id": instance_id,
        **res,
    }
    
    return new_res

model_name_to_run_results ={
    "claude37sonnet_sweagent": (
        Path("eval_reports2/eval_report_default_sweperf_claude__anthropic--claude-3-7-sonnet-20250219__t-0.00__p-1.00__c-1.00___swefficiency_full_test.csv"),
        Path("logs/run_evaluation/profile_runs/default_sweperf_claude__anthropic--claude-3-7-sonnet-20250219__t-0.00__p-1.00__c-1.00___swefficiency_full_test")
    ),
    "claude37sonnet_oh": (
        Path("eval_reports2/eval_report_us.anthropic.claude-3-7-sonnet-20250219-v1_0_maxiter_100_N_v0.51.1-no-hint-run_1.csv"),
        Path("logs/run_evaluation/profile_runs/us.anthropic.claude-3-7-sonnet-20250219-v1_0_maxiter_100_N_v0.51.1-no-hint-run_1")
    ),
    "gpt5mini_oh": (
        Path("eval_reports2/eval_report_gpt-5-mini_maxiter_100_N_v0.51.1-no-hint-run_1.csv"),
        Path("logs/run_evaluation/profile_runs/gpt-5-mini_maxiter_100_N_v0.51.1-no-hint-run_1")
    ),
    "gemini25flash_oh": (
        Path("eval_reports2/eval_report_gemini-2.5-flash_maxiter_100_N_v0.51.1-no-hint-run_1.csv"),
        Path("logs/run_evaluation/profile_runs/gemini-2.5-flash_maxiter_100_N_v0.51.1-no-hint-run_1")
    ),
}

def main(model_name):
    import datasets
    import tqdm
    import pandas as pd

    ds = datasets.load_dataset("swefficiency/swefficiency", split="test")
    
    # Filter out edits where we did not improve or correctness degraded.
    global PRED_RUN_DIR
    run_results_path, PRED_RUN_DIR = model_name_to_run_results[model_name]
    
    run_results = pd.read_csv(run_results_path)
    selected_instance_ids = set(
        run_results[
            (run_results["correctness"] == 1.0) &
            (run_results["pred_speedup_ratio"] >= 1.0)
        ]["instance_id"].tolist()
    )
    
    print(f"Total test instances: {len(ds)}")
    ds = ds.filter(lambda x: x["instance_id"] in selected_instance_ids)
    print(f"Filtered to {len(ds)} instances with correctness=1.0 and speedup>=1.0")

    results: List[Dict[str, Any]] = []
    for item in tqdm.tqdm(ds, total=len(ds), desc="Analyzing profiles"):
        r = wrapper(
            item,
            improvement_threshold_seconds=0.0,
            improvement_threshold_frac_of_expert=0.05,  # 2% per-function floor
            improvement_cumratio_cap=1.0,               # keep all above threshold
        )
        if r is not None:
            results.append(r)

    # Sort final results by jaccard index (lowest to highest), then by AvgOptimizationDepthFromWorkload"]["expert_pre"]
    results.sort(
        key=lambda x: (x.get("file_overlap_jaccard", float("inf")) or 0.0, x.get("AvgOptimizationDepthFromWorkload", {}).get("expert_pre", float("inf")) or float("inf"))
    )

    # Save results
    print(f"Saving {len(results)} results for model {model_name}...")
    out_path = f"analysis/adhoc/speedup_attribution/output/profilng_analysis_results_{model_name}.jsonl"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Results saved to {out_path}")

    # (Optional) quick dataset means for the requested metrics
    if results:
        import statistics as stats
        instance_ids = [r["instance_id"] for r in results]
        erc_file = [r["ERC"]["file"] for r in results]
        erc_func = [r["ERC"]["func"] for r in results]
        wfl = [r["LossDecomposition"]["WrongFileLoss"] for r in results]
        ifl = [r["LossDecomposition"]["InFileLoss"] for r in results]
        fj = [r["file_overlap_jaccard"] for r in results if r["file_overlap_jaccard"] is not None]

        def mean_or_nan(x): return float('nan') if not x else stats.fmean(x)

        print(f"Analyzed {len(instance_ids)} instances.")
        print(f"Average ERC (file): {mean_or_nan(erc_file):.4f}")
        print(f"Average ERC (func): {mean_or_nan(erc_func):.4f}")
        print(f"Average WrongFileLoss: {mean_or_nan(wfl):.4f}")
        print(f"Average InFileLoss: {mean_or_nan(ifl):.4f}")
        print(f"Average file_overlap_jaccard (defined only): {mean_or_nan(fj):.4f}")
        
        import statistics as stats
        
        depth_wl_exp = [r["AvgOptimizationDepthFromWorkload"]["expert_pre"]
                        for r in results
                        if r.get("AvgOptimizationDepthFromWorkload", {}).get("expert_pre") is not None]
        depth_wl_llm = [r["AvgOptimizationDepthFromWorkload"]["llm_pre"]
                        for r in results
                        if r.get("AvgOptimizationDepthFromWorkload", {}).get("llm_pre") is not None]

        print(f"Weighted avg optimization depth from 'workload' (expert): {mean_or_nan(depth_wl_exp):.2f}")
        print(f"Weighted avg optimization depth from 'workload' (LLM):    {mean_or_nan(depth_wl_llm):.2f}")

if __name__ == "__main__":
    for model_name in model_name_to_run_results.keys():
        print(f"Running analysis for model: {model_name}")
        main(model_name)
