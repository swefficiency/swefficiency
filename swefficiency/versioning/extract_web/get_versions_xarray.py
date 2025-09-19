import json
import os
import re
import requests
import sys

from datetime import datetime

from pathlib import Path

from datetime import datetime

sys.path.append("../../harness")
from swefficiency.versioning.utils import get_instances

PATH_TASKS_XARRAY = "artifacts/1_attributes/xarray-task-instances_attribute.jsonl"

# Get raw xarray dataset
data_tasks = get_instances(PATH_TASKS_XARRAY)

# Get version to date from xarray home page
resp = requests.get("https://docs.xarray.dev/en/stable/whats-new.html")
pattern = (
    r'<a class="reference internal nav-link( active)?" href="#v(.*)">v(.*) \((.*)\)</a>'
)
matches = re.findall(pattern, resp.text)
matches = list(set(matches))
matches = [x[1:] for x in matches]

# Get (date, version) pairs
date_formats = ["%B %d %Y", "%d %B %Y"]
keep_major_minor = lambda x, sep: ".".join(x.strip().split(sep)[:2])

times = []
for match in matches:
    parts = match[0].split("-")
    version = keep_major_minor(".".join(parts[0:3]), ".")
    date_str = " ".join(parts[3:])

    for f_ in date_formats:
        try:
            date_obj = datetime.strptime(date_str, f_)
            times.append((date_obj.strftime("%Y-%m-%d"), version))
        except:
            continue
        break

times = sorted(times, key=lambda x: x[0])[::-1]

for task in data_tasks:
    created_at = task["created_at"].split("T")[0]
    found = False
    for t in times:
        if t[0] < created_at:
            task["version"] = t[1]
            found = True
            break
    if not found:
        task["version"] = None

# Save xarray versioned data to repository
new_file_name = Path(PATH_TASKS_XARRAY).stem + "_versions.json"

OUTPUT_VERSION_DIR = "artifacts/2_versioning"
with open(
    os.path.join(OUTPUT_VERSION_DIR, new_file_name),
    "w",
) as f:
    json.dump(data_tasks, fp=f)
