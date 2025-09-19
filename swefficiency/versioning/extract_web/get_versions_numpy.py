import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import requests
from ghapi.core import GhApi

sys.path.append("../../harness")
from swefficiency.versioning.utils import get_instances

PATH_TASK_NUMPY = "artifacts/1_attributes/numpy-task-instances_attribute.jsonl"

# Get raw matplotlib dataset
data_tasks = get_instances(PATH_TASK_NUMPY)

# Get releases for statsmodels, which correspond to verions.
gh_api = GhApi(token=os.environ["GITHUB_TOKEN"])


PATTERN = r"^SciPy (\d+\.\d+\.\d+)$"
VERSION_TAG_PATTERN = r"^v(\d+\.\d+\.\d+)(?:rc\d+)?$"

keep_major_minor = lambda x, sep: ".".join(x.strip().split(sep)[:2])

times = dict()

i = 0
while True:
    try:
        raw_releases = gh_api.repos.list_releases(
            "numpy", "numpy", per_page=100, page=i
        )
        if len(raw_releases) == 0:
            break
        i += 1
    except:
        break

    for raw_release in raw_releases:
        release_name = raw_release["name"]
        version = raw_release["tag_name"]

        # Of the form "2019-06-07T06:11:59Z", convert to datetime.
        release_date = raw_release["created_at"].split("T")[0]

        # Find a match and extract version number if matched.
        match = re.match(PATTERN, release_name)
        version_tag_match = re.match(VERSION_TAG_PATTERN, version)
        if version_tag_match:
            version = version_tag_match.group(1)
            major_minor_version = keep_major_minor(version, ".")

            if major_minor_version not in times:
                times[major_minor_version] = release_date

            times[major_minor_version] = min(times[major_minor_version], release_date)

# Sort in reverse order.
times = sorted([(v, k) for k, v in times.items()], key=lambda x: x[0], reverse=True)

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
new_file_name = Path(PATH_TASK_NUMPY).stem + "_versions.json"

OUTPUT_VERSION_DIR = "artifacts/2_versioning"
with open(
    os.path.join(OUTPUT_VERSION_DIR, new_file_name),
    "w",
) as f:
    json.dump(data_tasks, fp=f)
