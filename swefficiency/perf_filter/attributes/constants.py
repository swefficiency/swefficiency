"""
Defining repo specific filters.
"""

import re

VERBATIM_KEYWORDS = [
    "PERF",
    "OPTIM",
]

BASE_PERF_KEYWORDS = [
    "performance",
    "speedup",
    "speeds up",
    "speed-up",
    "speed up",
    "faster",
    "memory",
    "optimize",
    "optimization",
    "profiling",
    "accelerate",
    "fast",
    "runtime",
    "efficiency",
    "benchmark",
    "latency",
    "throughput",
    "multithreading",
    "parallel",
    "concurrency",
    "concurrent",
    "profiling",
    "CPU usage",
    "memory usage",
    "resource usage",
    "cache",
    "caching",
    # Other common ones related to timing and benchmarks.
    "timeit",
    "asv",
]


def check_labels(pull: dict, value: list[str]) -> bool:
    labels = [label["name"].lower() for label in pull["labels"]]
    return any(v in label for v in value for label in labels)


def remove_markdown_comments(input_str):
    # Use a regular expression to find and remove Markdown comments
    return re.sub(r"<!--.*?-->", "", input_str, flags=re.DOTALL)


def filter_base(pull: dict, keywords=BASE_PERF_KEYWORDS):
    pull_body = pull["body"] or ""
    pull_title = pull["title"] or ""
    # pull_body = ""

    for item in [pull_body, pull_title]:
        item_lower = remove_markdown_comments(item.lower())

        if any(kw in item_lower for kw in BASE_PERF_KEYWORDS):
            return True

        if any(kw in item for kw in VERBATIM_KEYWORDS):
            return True

    return False


def filter_content(issue_text, keywords=BASE_PERF_KEYWORDS):
    if not issue_text:
        return False

    issue_text = issue_text.lower()
    issue_text = remove_markdown_comments(issue_text)
    if any(kw in issue_text for kw in keywords):
        return True
    return False


def filter_sklearn(pull: dict):
    pr_title = pull["title"].lower()

    # 1. If labelled with "Performance" label, probably perf related :-).
    if check_labels(pull, ["performance"]):
        return True

    # 2. Otherwise, check if pull title has "PERF" or  (commonly observed.)
    if any(kw in pr_title for kw in ["eff", "perf"]):
        return True

    # 3. Other filters?

    return False


def filter_astropy(pull: dict):
    pr_title = pull["title"].lower()

    # 1. If labelled with "Performance" label, probably perf related :-).
    if check_labels(pull, ["performance"]):
        return True

    # 2. Otherwise, check if pull title has "PERF" or  (commonly observed.)
    if any(kw in pr_title for kw in ["eff", "perf", "speed up"]):
        return True

    # 3. Other filters?

    return False


def filter_matplotlib(pull: dict):
    pr_title = pull["title"].lower()

    # 1. If labelled with "Performance" label, probably perf related :-).
    if check_labels(pull, ["performance"]):
        return True

    # 2. Otherwise, check if pull title has "PERF" or  (commonly observed.)
    if any(kw in pr_title for kw in ["perf"]):
        return True

    # 3. Other filters?

    return False


# TODO: Come up with filters per repo!


def filter_pylint(pull: dict):
    pr_title = pull["title"].lower()

    # 1. If labelled with "Performance" label, probably perf related :-).
    if check_labels(pull, ["performance"]):
        return True

    # 2. Otherwise, check if pull title has "PERF" or  (commonly observed.)
    if any(kw in pr_title for kw in ["perf"]):
        return True

    # 3. Other filters?

    return False


def filter_seaborn(pull: dict):
    pr_title = pull["title"].lower()

    # 1. If labelled with "Performance" label, probably perf related :-).
    if check_labels(pull, ["perf"]):
        return True

    # 2. Otherwise, check if pull title has "PERF" or  (commonly observed.)
    if any(kw in pr_title for kw in ["perf"]):
        return True

    # 3. Other filters?

    return False


def filter_sphinx(pull: dict):
    pr_title = pull["title"].lower()

    # 1. If labelled with "Performance" label, probably perf related :-).
    if check_labels(pull, ["type:performance"]):
        return True

    # 2. Otherwise, check if pull title has "PERF" or  (commonly observed.)
    if any(kw in pr_title for kw in ["perf"]):
        return True

    # 3. Other filters?
    return False


def filter_sympy(pull: dict):
    pr_title = pull["title"].lower()

    # 1. If labelled with "Performance" label, probably perf related :-).
    if check_labels(pull, ["performance"]):
        return True

    # 2. Otherwise, check if pull title has "PERF" or  (commonly observed.)
    if any(kw in pr_title for kw in ["perf"]):
        return True

    # 3. Other filters?
    return False


def filter_xarray(pull: dict):
    pr_title = pull["title"].lower()

    # 1. If labelled with "Performance" label, probably perf related :-).
    if check_labels(pull, ["topic-performance"]):
        return True

    # 2. Otherwise, check if pull title has "PERF" or  (commonly observed.)
    if any(kw in pr_title for kw in ["perf", "speed up"]):
        return True

    # 3. Other filters?
    return False


# Aggregate all filters in single map access.
REPO_PERF_FILTERS = {
    "default": filter_base,
    "astropy": filter_astropy,
    "scikit-learn": filter_sklearn,
    "matplotlib": filter_matplotlib,
    # No django or flask, manual kw search did not yield much.
    "pylint": filter_pylint,
    # No pytest or requests, manual kw search did not yield much.
    "seaborn": filter_seaborn,
    "sphinx": filter_sphinx,
    "sympy": filter_sympy,
    "xarray": filter_xarray,
    # TODO: Rest of SWE-Gym?
}

# SWE-GYM


def filter_pandas(pull: dict):
    pr_title = pull["title"]

    # 1. If labelled with "Performance" label, probably perf related :-).
    if check_labels(pull, ["performance"]):
        return True

    keywords = ["perf", "speed up", "efficiency", "performance"]
    # keywords = ["PERF", "performance"]
    if any(kw in pr_title for kw in keywords):
        return True

    # 3. Other filters?
    return False


def filter_dask(pull: dict):
    pr_title = pull["title"].lower()

    # 2. Otherwise, check if pull title has "PERF" or  (commonly observed.)
    if any(
        kw in pr_title
        for kw in [
            "perf",
            "speed up",
            "efficiency",
            "remove",
            "avoid",
            "overhead",
            "memory",
        ]
    ):
        return True

    # 3. Other filters?
    return False


def filter_numpy(pull: dict):
    pr_title = pull["title"]

    keywords = ["perf", "speed up", "efficiency", "performance"]
    if any(kw in pr_title for kw in keywords):
        return True

    # 3. Other filters?
    return False


def filter_statsmodels(pull: dict):
    pr_title = pull["title"]

    # 1. If labelled with "Performance" label, probably perf related :-).
    if check_labels(pull, ["performance"]):
        return True

    keywords = ["perf", "speed up", "efficiency", "performance"]
    # keywords = ["PERF", "performance"]
    if any(kw in pr_title for kw in keywords):
        return True

    # 3. Other filters?
    return False


REPO_PERF_FILTERS.update(
    {
        "pandas": filter_pandas,
        "dask": filter_dask,
        "numpy": filter_numpy,
        "scipy": filter_numpy,
        "statsmodels": filter_statsmodels,
    }
)


def filter_pillow(pull: dict):
    pr_title = pull["title"]

    # 1. If labelled with "Performance" label, probably perf related :-).
    if check_labels(pull, ["performance"]):
        return True

    keywords = ["perf", "speed", "efficiency", "performance"]
    # keywords = ["PERF", "performance"]
    if any(kw in pr_title for kw in keywords):
        return True

    # 3. Other filters?
    return False


def filter_spacy(pull: dict):
    pr_title = pull["title"]

    # 1. If labelled with "Performance" label, probably perf related :-).
    if check_labels(pull, ["perf"]):
        return True

    keywords = ["perf", "speed", "efficiency", "performance"]
    # keywords = ["PERF", "performance"]
    if any(kw in pr_title for kw in keywords):
        return True

    # 3. Other filters?
    return False


def filter_numba(pull: dict):
    pr_title = pull["title"]

    # 1. If labelled with "Performance" label, probably perf related :-).
    if check_labels(pull, ["performance"]):
        return True

    keywords = ["perf", "speed", "efficiency", "performance"]
    # keywords = ["PERF", "performance"]
    if any(kw in pr_title for kw in keywords):
        return True

    # 3. Other filters?
    return False


def filter_gensim(pull: dict):
    pr_title = pull["title"]

    # 1. If labelled with "Performance" label, probably perf related :-).
    if check_labels(pull, ["performance"]):
        return True

    keywords = ["perf", "speed", "efficiency", "performance"]
    # keywords = ["PERF", "performance"]
    if any(kw in pr_title for kw in keywords):
        return True

    # 3. Other filters?
    return False


def filter_scikit_image(pull: dict):
    pr_title = pull["title"]

    # 1. If labelled with "Performance" label, probably perf related :-).
    if check_labels(pull, ["performance"]):
        return True

    keywords = ["perf", "speed", "efficiency", "performance"]
    # keywords = ["PERF", "performance"]
    if any(kw in pr_title for kw in keywords):
        return True

    # 3. Other filters?
    return False


REPO_PERF_FILTERS.update(
    {
        "pillow": filter_pillow,
        "spacy": filter_spacy,
        "numba": filter_numba,
        "gensim": filter_gensim,
        "scikit-image": filter_scikit_image,
    }
)

#
