import json
import os
import re
import requests
import sys

from pathlib import Path

from datetime import datetime

from ghapi.core import GhApi

sys.path.append("../../harness")
from swefficiency.versioning.utils import get_instances

PATH_TASK_STATSMODELS = (
    "artifacts/1_attributes/statsmodels-task-instances_attribute.jsonl"
)

# Get raw matplotlib dataset
data_tasks = get_instances(PATH_TASK_STATSMODELS)

# Get releases for statsmodels, which correspond to verions.
gh_api = GhApi(token=os.environ["GITHUB_TOKEN"])

raw_releases = gh_api.repos.list_releases("statsmodels", "statsmodels", per_page=100)

PATTERN = r"^(Version (\d+\.\d+\.\d+) Release|Release (\d+\.\d+\.\d+))$"

keep_major_minor = lambda x, sep: ".".join(x.strip().split(sep)[:2])

versions_to_release_date = {}

for raw_release in raw_releases:
    release_name = raw_release["name"]

    # Of the form "2019-06-07T06:11:59Z", convert to datetime.
    release_date = raw_release["created_at"].split("T")[0]

    # Find a match and extract version number if matched.
    match = re.match(PATTERN, release_name)
    if match:
        version = match.group(2) or match.group(3)

        version = keep_major_minor(version, ".")
        versions_to_release_date[version] = min(
            versions_to_release_date.get(version, release_date), release_date
        )


# Sort in reverse order.
times = [(v, k) for k, v in versions_to_release_date.items()]
times = sorted(times, key=lambda x: x[0], reverse=True)

for task in data_tasks:
    created_at = task["created_at"].split("T")[0]
    for t in times:
        if t[0] < created_at:
            task["version"] = t[1]
            # print(task["version"])
            break

    if "version" not in task:
        # Assign last version.
        task["version"] = times[-1][1]


# Construct map of versions to task instances
map_v_to_t = {}
for i, t in enumerate(data_tasks):
    if t["version"] not in map_v_to_t:
        map_v_to_t[t["version"]] = []
    map_v_to_t[t["version"]].append(t)

# Save matplotlib versioned data to repository
new_file_name = Path(PATH_TASK_STATSMODELS).stem + "_versions.json"

OUTPUT_VERSION_DIR = "artifacts/2_versioning"
with open(
    os.path.join(OUTPUT_VERSION_DIR, new_file_name),
    "w",
) as f:
    json.dump(data_tasks, fp=f)
