# Update the chart per feedback:
# - Put percentages on top of bars
# - Keep model coloring, but simplify legend to only show "Cause" with solid (wrong-file) vs hatched (in-file)
# - Rename y-axis to "% of expert speedup missed"
# - Remove the title
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors
from matplotlib.ticker import PercentFormatter
import numpy as np
import os

# === Spacing controls ===
center_spacing = 0.75   # 1.0 = default center-to-center; <1.0 = tighter, >1.0 = looser
outer_pad = 0.5      # extra space at the plot edges, as a fraction of center_spacing
                         # use a tuple (left, right) for asymmetric padding
bar_width = 0.4         # keep the same bar thickness

_model_styles = {
    "GPT-5 Mini (OH)":               dict(color="black", marker="o", alpha=0.9),
    "GPT-5 Mini (SWE-agent)":        dict(color="black", marker="^", alpha=0.9, linestyle="dashed"),
    "Claude 3.7 Sonnet (OH)":        dict(color="#c15f3c", marker="o", alpha=0.9),
    "Claude 3.7 Sonnet (SWE-agent)": dict(color="#c15f3c", marker="^", alpha=0.9, linestyle="dashed"),
    "Gemini 2.5 Flash (OH)":         dict(color="#088cfb", marker="o", alpha=0.9),
    "Gemini 2.5 Flash (SWE-agent)":  dict(color="#088cfb", marker="^", alpha=0.9, linestyle="dashed"),
}

data = [
    {"model": "Claude 3.7 Sonnet (OH)", "WrongFileLoss": 0.389, "InFileLoss": 0.297},
    {"model": "GPT-5 Mini (OH)", "WrongFileLoss": 0.449, "InFileLoss": 0.274},
    {"model": "Gemini 2.5 Flash (OH)", "WrongFileLoss": 0.451, "InFileLoss": 0.283},
]

def wrap_labels(labels, width):
    import textwrap
    return ["\n".join(textwrap.wrap(label, width)) for label in labels]

def lighten_color(c, factor=0.55):
    rgb = mcolors.to_rgb(c)
    return tuple(ch + (1.0 - ch) * factor for ch in rgb)

labels = [d["model"] for d in data]
wrong_file = [d["WrongFileLoss"] for d in data]
in_file = [d["InFileLoss"] for d in data]
totals = [wf + inf for wf, inf in zip(wrong_file, in_file)]
colors = [_model_styles[label]["color"] for label in labels]
light_colors = [lighten_color(c, 0.55) for c in colors]

fig, ax = plt.subplots(figsize=(6, 4))

# === Positions with configurable center spacing ===
x = np.arange(len(labels)) * center_spacing

# Bars
bars_wrongfile = ax.bar(x, wrong_file, label="Wrong-file", color=colors, width=bar_width)
plt.rcParams['hatch.linewidth'] = 2.0
bars_infile = ax.bar(
    x, in_file, bottom=wrong_file, label="Wrong-function (in-file)",
    color=light_colors, hatch="//", width=bar_width
)

# Totals as percentages
for xi, total in zip(x, totals):
    ax.text(xi, total + 0.01, f"{total*100:.1f}%", ha="center", va="bottom", fontsize=14)

# Y axis format/label
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_ylabel("% of expert\nspeedup missed", fontsize=18, fontweight="bold")
ax.tick_params(axis='y', labelsize=14)

# X ticks
ax.set_xticks(x)
ax.set_xticklabels(wrap_labels(labels, 14), ha="center", fontsize=14)
ax.set_xlabel("LM System", fontsize=18, fontweight="bold")

# Limits/grid
ax.set_ylim(0, 1.0)
ax.grid(axis="y", linestyle="--", alpha=0.4)

# === Outer padding (in units of center_spacing) ===
if isinstance(outer_pad, (tuple, list, np.ndarray)) and len(outer_pad) == 2:
    left_pad_frac, right_pad_frac = outer_pad
else:
    left_pad_frac = right_pad_frac = float(outer_pad)

left_pad = left_pad_frac * center_spacing
right_pad = right_pad_frac * center_spacing

ax.set_xlim(x[0] - bar_width/2 - left_pad, x[-1] + bar_width/2 + right_pad)
ax.margins(x=0)  # disable automatic extra margins so our manual padding is used

# Legend: simple "Cause" with solid vs hatch
legend_patches = [
    mpatches.Patch(facecolor="0.5", label="Wrong-file"),
    mpatches.Patch(facecolor="0.85", hatch="///", label="Wrong-function (in-file)")
]
ax.legend(handles=legend_patches, loc="upper left")

plt.tight_layout()

# Save
out_dir = "assets/figures"
os.makedirs(out_dir, exist_ok=True)
png_path = os.path.join(out_dir, "openhands_speedup_missed_stacked.png")
fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
