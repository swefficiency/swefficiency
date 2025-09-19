import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import requests

sys.path.append("../../harness")
from swefficiency.versioning.utils import get_instances

PATH_TASK_DASK = "artifacts/1_attributes/dask-task-instances_attribute.jsonl"

# Get raw matplotlib dataset
data_tasks = get_instances(PATH_TASK_DASK)

# Get version to date from matplotlib home page
resp = requests.get("https://docs.dask.org/en/stable/changelog.html")
# <a class="reference internal" href="v0.23.3.html">Whatâs new in 0.23.3 (July 7, 2018)</a>
pattern = r'<h2>(.*?)<a class="headerlink" href="#v(.*?)" title="Permalink to this headline">¶</a></h2>'
matches = re.findall(pattern, resp.text)
matches = list(set(matches))

# print(matches)

# print(resp.text)
# print(matches)

# Get (date, version) pairs
date_format = "%Y-%m-%d"
keep_major_minor = lambda x, sep: ".".join(x.strip().split(sep)[:2])

times = []
for match in matches:
    version_info, date_info = match[0], match[1]

    version = version_info.split("/")[0].strip()
    date_string = "-".join(date_info.split("-")[-3:-1] + ["1"])
    print(version, date_string)

    version = keep_major_minor(version, ".")

    try:
        date_obj = datetime.strptime(date_string, date_format)
        times.append((date_obj.strftime("%Y-%m-%d"), version))
    except Exception as e:
        print(e)
        continue
times = sorted(times, key=lambda x: x[0])[::-1]

print(times)


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
new_file_name = Path(PATH_TASK_DASK).stem + "_versions.json"

OUTPUT_VERSION_DIR = "artifacts/2_versioning"
with open(
    os.path.join(OUTPUT_VERSION_DIR, new_file_name),
    "w",
) as f:
    json.dump(data_tasks, fp=f)
