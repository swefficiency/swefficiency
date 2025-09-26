#!/usr/bin/env python3
import json
from collections import Counter

import matplotlib.pyplot as plt

# Hardcoded path
JSONL_PATH = "analysis/llm/outputs/diff_classification_results.jsonl"


def read_jsonl(path):
    """Yield parsed JSON objects from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    field = "classification"  # the field to count
    counter = Counter()

    for obj in read_jsonl(JSONL_PATH):
        if field in obj and obj[field] is not None:
            counter[str(obj[field])] += 1

    if not counter:
        print("No entries found to count.")
        return

    # Sort by count (descending)
    items = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
    labels, counts = zip(*items)

    # Print counts to stdout
    print("\nCounts:")
    width = max(len(s) for s in labels)
    for lab, c in items:
        print(f"{lab:<{width}}  {c}")

    # Plot horizontal bar chart
    plt.figure(figsize=(10, max(4, 0.4 * len(labels))))
    positions = range(len(labels))
    bars = plt.barh(positions, counts, color="purple", alpha=0.7)
    plt.yticks(positions, labels, fontsize=14)
    plt.xlabel("# of Instances", fontsize=14)
    # plt.title(f"Counts of '{field}'")
    plt.gca().invert_yaxis()  # largest at the top

    # Extend x-axis so labels fit
    plt.xlim(0, max(counts) * 1.15)  # add 15% margin

    # Add numeric labels at the end of bars (floating)
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_width() + (0.01 * max(counts)),  # slight offset past the bar
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            ha="left",
            fontsize=14,
            fontweight="bold",
        )

    plt.tight_layout()

    # Save the plot
    plt.savefig("docs/assets/figures/diff_classification_counts.png")


if __name__ == "__main__":
    main()
