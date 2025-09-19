# %%

import os
import subprocess

import shutil

from ghapi.core import GhApi

import requests

import tqdm

import json
import pandas as pd

import functools


# Get all files with coverage info.
# find logs/run_evaluation/ -type f -name covering_tests.txt | grep "debugging_coverage_" | xargs -n1 dirname

tasks_dir = "artifacts/tasks"

base_dir = "logs/run_evaluation"
output_dir = "scripts/perf"
output_file = os.path.join(output_dir, "pr_to_info.jsonl")


@functools.lru_cache(maxsize=None)
def get_repo_dataset(repo_name: str):
    return pd.read_json(
        os.path.join(tasks_dir, f"{repo_name}-task-instances.jsonl.all"),
        lines=True,
    )


gh_api = GhApi(token=os.environ.get("GITHUB_TOKEN"))


def get_url(url, github_token=os.environ.get("GITHUB_TOKEN"), params=None):

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",  # Good practice to specify API version
    }
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    response = requests.get(url, headers=headers, params=params)
    return response.json()


pr_to_info = {}

if not os.path.exists(output_file):
    f = open(output_file, "w")

    for root, dirs, files in tqdm.tqdm(os.walk(base_dir)):
        has_covering_tests = any(f.startswith("covering_tests.txt") for f in files)
        has_prefix = "debugging_coverage_" in root

        if has_prefix and has_covering_tests:

            folder_name = root.split("/")[-1]
            owner = folder_name.split("__")[0]
            repo = folder_name.split("__")[1].rsplit("-", 1)[0]
            number = int(folder_name.rsplit("-", 1)[-1])

            print(f"Owner: {owner}, Repo: {repo}, Number: {number}")

            repo_dataset_df = get_repo_dataset(repo)
            instance = repo_dataset_df[
                repo_dataset_df["instance_id"] == folder_name
            ].iloc[0]

            try:

                pr_info = gh_api.pulls.get(owner=owner, repo=repo, pull_number=number)

                # Get body, PR comments, issue text, issue comments.
                pr_link = pr_info["html_url"]
                body = pr_info["body"]
                review_comments_url = pr_info["review_comments_url"]
                comments_url = pr_info["comments_url"]

                raw_pr_comments = get_url(review_comments_url)

                raw_issue_comments = []
                if instance["issue_numbers"]:
                    for issue_number in instance["issue_numbers"]:
                        raw_issue_comments.extend(
                            gh_api.issues.list_comments(
                                owner=owner, repo=repo, issue_number=issue_number
                            )
                        )

                pr_comments = [comment["body"] for comment in raw_pr_comments]
                issue_comments = [comment["body"] for comment in raw_issue_comments]

                info = {
                    "owner": owner,
                    "repo": repo,
                    "number": number,
                    "title": pr_info["title"] or "",
                    "pr_body": body or "",
                    "pr_comments": pr_comments,
                    "issue_body": instance["problem_statement"] or "",
                    "issue_comments": issue_comments,
                    "html_url": pr_link,
                    "original_folder": root,
                }

                # Json flush to output file.
                print(json.dumps(info), file=f, flush=True)

            except Exception as e:
                continue


# %%

# Load JSONL file.
import json
import pandas as pd
import re

information_df = pd.read_json(
    output_file,
    lines=True,
)


def extract_markdown_code_blocks(md_text):
    """
    Extracts all code blocks from a markdown string.

    Args:
        md_text (str): The markdown string.

    Returns:
        List[str]: A list of code block strings (without the backticks or language tags).
    """
    if md_text is None:
        return []

    # Regex pattern for fenced code blocks
    pattern = r"^```(?!suggestion)(?:\w+)?\s*\n(.*?)(?=^```)\n?```"
    result = re.findall(pattern, md_text, re.DOTALL | re.MULTILINE)
    return result


def extract_markdown_from_row(row):
    body_code_blocks = extract_markdown_code_blocks(row["pr_body"])
    comments_code_blocks = [
        extract_markdown_code_blocks(comment) for comment in row["pr_comments"]
    ]
    issue_code_blocks = extract_markdown_code_blocks(row["issue_body"])
    issue_comments_code_blocks = [
        extract_markdown_code_blocks(comment) for comment in row["issue_comments"]
    ]

    # print(row['pr_body'])
    result = [
        *body_code_blocks,
        *sum(comments_code_blocks, []),
        *issue_code_blocks,
        *sum(issue_comments_code_blocks, []),
    ]

    return result


# Create a new column, which concatenates a list of code blocks running the above method over 'pr_body', each elem
information_df["code_blocks"] = information_df.apply(extract_markdown_from_row, axis=1)

# Filter out rows with empty codeblocks.
# information_df = information_df[information_df['code_blocks'].apply(len) > 0]

# Show just html_url column.
information_df["html_url"].to_list()
# %%

# For each element in this list, copy the original folder

data_dir = "data"

for i, row in information_df.iterrows():

    original_folder = row["original_folder"]
    instance_id = f"{row['owner']}__{row['repo']}-{row['number']}"
    working_dir = os.path.join(data_dir, instance_id)

    # Check if "workload.py" is in directory.
    if os.path.exists(os.path.join(original_folder, "workload.py")):
        continue

    # shutil.rmtree(working_dir, ignore_errors=True)
    os.makedirs(working_dir, exist_ok=True)
    shutil.copytree(original_folder, working_dir, dirs_exist_ok=True)

    code_block_file = os.path.join(working_dir, "2_code_blocks.txt")
    with open(code_block_file, "w") as f:
        f.write(
            "\n=========================================\n".join(row["code_blocks"])
        )

    comments_file = os.path.join(working_dir, "1_comments.txt")

    with open(comments_file, "w") as f:
        components = [
            "PR URL:",
            row["html_url"] or "",
            "PR Title:",
            row["title"] or "",
            "PR Body:",
            row["pr_body"] or "",
            "PR Comments:",
            "\n".join(row["pr_comments"]),
            "Issue Body:",
            row["issue_body"] or "",
            "Issue Comments:",
            "\n".join(row["issue_comments"]),
        ]

        f.write("\n\n".join(components))
