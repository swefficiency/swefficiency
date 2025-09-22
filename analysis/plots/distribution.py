import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import datasets
from matplotlib import pyplot as plt

ds = datasets.load_dataset("swefficiency-anon/swefficiency", split="test")


def get_repo_count(ds):
    repos = []
    for item in ds:
        repo = item["repo"].split("/")[-1]
        repos.append(repo)

    return Counter(repos)


import math

import matplotlib.pyplot as plt


def make_pie_chart_with_callouts(repo_counts, title_suffix: str = "") -> plt.Figure:
    labels = list(repo_counts.keys())
    sizes = list(repo_counts.values())

    # use Set3 pastel palette
    cmap = plt.cm.Set3
    colors = cmap(range(len(labels)))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    wedges, _ = ax.pie(
        sizes,
        startangle=90,
        colors=colors,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    ax.axis("equal")

    # Add labels with arrows pointing close to wedges
    for i, wedge in enumerate(wedges):
        angle = (wedge.theta2 + wedge.theta1) / 2.0
        x = math.cos(math.radians(angle))
        y = math.sin(math.radians(angle))

        # keep labels close to pie
        r = 1.2
        ha = "left" if x > 0 else "right"
        ax.annotate(
            f"{labels[i]} ({sizes[i]})",
            xy=(x, y),
            xytext=(r * x, r * y),
            ha=ha,
            va="center",
            arrowprops=dict(
                arrowstyle="-", connectionstyle="arc3,rad=0", lw=0.8, color="black"
            ),
            fontsize=20,
        )

    # ax.set_title(
    #     "Distribution of SWE-Perf across open-source GitHub repositories" + title_suffix,
    #     pad=25
    # )
    fig.tight_layout()
    return fig


repo_count = get_repo_count(ds)
fig = make_pie_chart_with_callouts(repo_count)
fig.savefig("swefficiency_distribution_test_set.png", bbox_inches="tight", dpi=300)


# %%

# -----------------------------
# Helpers to compute statistics
# -----------------------------


def _safe_len(x) -> Optional[int]:
    if x is None:
        return None
    try:
        return len(x)
    except Exception:
        return None


def _parse_git_diff(patch_text: Optional[str]) -> Tuple[int, int, int]:
    """
    Very lightweight git-diff parser.
    Returns (lines_edited, files_edited, funcs_edited).
    - lines_edited: count of '+' or '-' lines (excluding headers like '+++'/'---' and '@@')
    - files_edited: number of unique files touched (based on 'diff --git' or '+++ b/...').
    - funcs_edited: number of unique function/class names that appear in changed lines
                    (naive heuristic searching for 'def ' or 'class ' in +/- lines).
    """
    if not patch_text:
        return 0, 0, 0

    lines_edited = 0
    files: Set[str] = set()
    funcs: Set[str] = set()

    current_file = None
    for line in patch_text.splitlines():
        # Track files
        if line.startswith("diff --git "):
            # Example: diff --git a/sklearn/x.py b/sklearn/x.py
            parts = line.split()
            # Try to grab the "b/<path>" part
            try:
                b_idx = parts.index([p for p in parts if p.startswith("b/")][0])
                current_file = parts[b_idx][2:]
                files.add(current_file)
            except Exception:
                current_file = None
        elif line.startswith("+++ "):
            # Example: +++ b/sklearn/x.py or +++ /dev/null
            if line.startswith("+++ b/"):
                files.add(line[6:].strip())
                current_file = line[6:].strip()
            else:
                current_file = None

        # Count edited lines (exclude headers)
        if line.startswith(("+", "-")) and not (line.startswith(("+++", "---", "@@"))):
            lines_edited += 1

            # Naive function/class name extraction on changed lines
            # This is a heuristic; it counts definitions touched.
            text = line[1:]  # drop +/- for matching
            for marker in ("def ", "class "):
                if marker in text:
                    # pull the token after marker as the "name"
                    try:
                        rest = text.split(marker, 1)[1].strip()
                        name = (
                            rest.split("(")[0].split(":")[0].split()[0]
                        )  # crude but works okay
                        if name:
                            funcs.add((current_file or ""), name)
                    except Exception:
                        pass

    return lines_edited, len(files), len(funcs)


def _mean(nums: List[float]) -> Optional[float]:
    nums = [n for n in nums if n is not None and not math.isnan(n)]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _fmt_number(x: Optional[float], k_suffix: bool = False, pct: bool = False) -> str:
    if x is None:
        return "—"
    if pct:
        return f"{x:.1f}%"
    if k_suffix and x >= 1000:
        return f"{x/1000:.0f}k"
    if float(x).is_integer():
        return f"{int(x)}"
    return f"{x:.1f}"


# -----------------------------
# Aggregation from dataset rows
# -----------------------------


def aggregate_metrics(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute all summary stats needed for the table and pie chart."""
    repos = []
    tests_per_inst: List[Optional[int]] = []
    lines_edited_per_inst: List[int] = []
    files_edited_per_inst: List[int] = []
    funcs_edited_per_inst: List[int] = []
    speedup_per_inst: List[Optional[float]] = []
    workload_lines_per_inst: List[int] = []

    total_instances = 0

    for r in rows:
        total_instances += 1
        repo = r.get("repo")
        if repo:
            repos.append(repo)

        # covering_tests -> # Related tests
        tests_per_inst.append(_safe_len(r.get("PASS_TO_PASS")))

        # patch -> edit stats
        le, fe, fu = _parse_git_diff(r.get("patch"))
        lines_edited_per_inst.append(le)
        files_edited_per_inst.append(fe)
        funcs_edited_per_inst.append(fu)

        workload_lines_per_inst.append(_safe_len(r.get("workload", "").splitlines()))

        # performance (if provided)
        sp = r.get("speedup")
        if isinstance(sp, (int, float)):
            speedup_per_inst.append(float(sp))
        else:
            speedup_per_inst.append(None)

    repo_counts = Counter(repos)

    metrics = {
        "n_instances": total_instances,
        "n_repos": len(set(repos)),
        "repo_counts": repo_counts,
        "tests_min": min([x for x in tests_per_inst if x is not None], default=None),
        "tests_mean": _mean([x for x in tests_per_inst if x is not None]),
        "tests_max": max([x for x in tests_per_inst if x is not None], default=None),
        "workload_min": (
            min(workload_lines_per_inst) if workload_lines_per_inst else None
        ),
        "workload_mean": _mean(workload_lines_per_inst),
        "workload_max": (
            max(workload_lines_per_inst) if workload_lines_per_inst else None
        ),
        "lines_edit_min": min(lines_edited_per_inst) if lines_edited_per_inst else None,
        "lines_edit_mean": _mean(lines_edited_per_inst),
        "lines_edit_max": max(lines_edited_per_inst) if lines_edited_per_inst else None,
        "files_edit_min": min(files_edited_per_inst) if files_edited_per_inst else None,
        "files_edit_mean": _mean(files_edited_per_inst),
        "files_edit_max": max(files_edited_per_inst) if files_edited_per_inst else None,
        "funcs_edit_mean": _mean(funcs_edited_per_inst),
        "funcs_edit_max": max(funcs_edited_per_inst) if funcs_edited_per_inst else None,
        # Keep speedup as-is; many users store >1 == faster; adapt if you prefer percentage.
        "speedup_mean": _mean([x for x in speedup_per_inst if x is not None]),
        "speedup_max": max(
            [x for x in speedup_per_inst if x is not None], default=None
        ),
    }
    return metrics


def make_summary_table_latex(
    metrics: Dict[str, Any],
    caption: str = "Average and maximum numbers characterizing different attributes of SWE-Perf.",
    label: str = "tab:sweperf-summary",
) -> str:
    """Return LaTeX code for the summary table (Booktabs)."""

    def _fmt_number(
        x: Optional[float], k_suffix: bool = False, pct: bool = False
    ) -> str:
        if x is None:
            return "—"
        if pct:
            return f"{x:.1f}\\%"
        if k_suffix and x >= 1000:
            return f"{x/1000:.0f}k"
        if float(x).is_integer():
            return f"{int(x)}"
        return f"{x:.1f}"

    rows: List[Tuple[str, str, str, str]] = []

    # Size
    rows.append(
        ("Size", r"\# Instances", _fmt_number(metrics.get("n_instances")), "")
    )  # total
    rows.append(("Size", r"\# Repos", _fmt_number(metrics.get("n_repos")), ""))

    # Codebase (placeholders)
    rows.append(("Codebase", r"\# Files (non-test)", "—", "—"))
    rows.append(("Codebase", r"\# Lines (non-test)", "—", "—"))

    # Expert Patch
    rows.append(
        (
            "Expert Patch",
            r"\# Lines edited",
            _fmt_number(metrics.get("lines_edit_mean")),
            _fmt_number(metrics.get("lines_edit_max")),
        )
    )
    rows.append(
        (
            "Expert Patch",
            r"\# Files edited",
            _fmt_number(metrics.get("files_edit_mean")),
            _fmt_number(metrics.get("files_edit_max")),
        )
    )
    rows.append(
        (
            "Expert Patch",
            r"\# Func. Edited",
            _fmt_number(metrics.get("funcs_edit_mean")),
            _fmt_number(metrics.get("funcs_edit_max")),
        )
    )

    # Tests
    rows.append(
        (
            "Tests",
            r"\# Related tests",
            _fmt_number(metrics.get("tests_mean")),
            _fmt_number(metrics.get("tests_max")),
        )
    )
    rows.append(("Tests", "Original runtime / s", "—", "—"))

    # Functions
    rows.append(("Functions", r"\# Oracle", "—", "—"))
    rows.append(("Functions", r"\# Realistic", "—", "—"))

    # Performance
    if (
        metrics.get("speedup_mean") is not None
        or metrics.get("speedup_max") is not None
    ):
        rows.append(
            (
                "Performance",
                "Speedup",
                _fmt_number(metrics.get("speedup_mean")),
                _fmt_number(metrics.get("speedup_max")),
            )
        )
    else:
        rows.append(("Performance", "Ratio", "—", "—"))

    # Build LaTeX (Booktabs + a little spacing between category blocks)
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\begin{tabular}{@{}llrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"Category & Metric & Mean & Max \\")
    lines.append(r"\midrule")

    # Insert \addlinespace when category changes to mimic block separation
    prev_cat = None
    for cat, metric, mean, mx in rows:
        if prev_cat is not None and cat != prev_cat:
            lines.append(r"\addlinespace")
        lines.append(f"{cat} & {metric} & {mean} & {mx} \\\\")
        prev_cat = cat

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


metrics = aggregate_metrics(ds)
latex_table = make_summary_table_latex(metrics)

print(metrics)

with open("swefficiency_summary_table.tex", "w") as f:
    f.write(latex_table)
