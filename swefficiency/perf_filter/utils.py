import argparse
import json
import os

import pandas as pd


def extract_edits(patch: str) -> list[tuple[str, str, str]]:
    """
    Assume diff is in unified format and extract which files are affected.

    diff --git a/src/sqlfluff/core/templaters/placeholder.py b/src/sqlfluff/core/templaters/placeholder.py
    --- a/src/sqlfluff/core/templaters/placeholder.py
    +++ b/src/sqlfluff/core/templaters/placeholder.py
    """

    split_by_diff_git = patch.split("diff --git")
    edits = []

    assert len(split_by_diff_git) > 1, "Patch does not contain any diff --git"
    assert split_by_diff_git[0] == "", "Patch does not start with diff --git"

    for diff_git in split_by_diff_git[1:]:
        diff_git = diff_git.strip()

        lines = diff_git.split("\n")
        source_file_name = lines[0].split()[1]
        dest_file_name = lines[1].split()[1]

        remaining_lines = "\n".join(lines[2:])
        edits.append((source_file_name, dest_file_name, remaining_lines))

    return edits


def read_jsonl(jsonl_path: str, to_df=False):
    """Helper util to read in instances easily."""

    if to_df:
        df = pd.read_json(jsonl_path, lines=True)
        return df
    else:
        jsonl_items = []
        with open(jsonl_path, "r") as f:
            file_iter = iter(f)
            while True:
                try:
                    data = next(file_iter, None)
                    if not data:
                        break
                    jsonl_items.append(json.loads(data))
                except:
                    pass

        return jsonl_items


def get_gh_tokens(env_var_name="GITHUB_TOKENS"):
    gh_tokens = os.environ.get(env_var_name).split(",")
    return gh_tokens


def is_doc_file(file_path: str) -> bool:
    return file_path.endswith(".md") or file_path.endswith(".rst")


def has_lock_file_change(file_path: str) -> bool:
    return file_path.endswith(".lock")
