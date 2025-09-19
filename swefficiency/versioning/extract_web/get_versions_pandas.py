import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import requests

sys.path.append("../../harness")
from swefficiency.versioning.utils import get_instances

PATH_TASK_PANDAS = "artifacts/1_attributes/pandas-task-instances_attribute.jsonl"

# Get raw matplotlib dataset
data_tasks = get_instances(PATH_TASK_PANDAS)

# Get version to date from matplotlib home page
resp = requests.get("https://pandas.pydata.org/docs/whatsnew/index.html#release-notes")
# <a class="reference internal" href="v0.23.3.html">Whatâs new in 0.23.3 (July 7, 2018)</a>
pattern = r'<a class="reference internal" href="v(.*?).html">(.*?)</a>'
matches = re.findall(pattern, resp.text)
matches = list(set(matches))

# print(resp.text)
# print(matches)

# Get (date, version) pairs
date_format = "%b %d, %Y"
keep_major_minor = lambda x, sep: ".".join(x.strip().split(sep)[:2])

times = []
for match in matches:
    print(match)
    version, s = match[0], match[1]
    if "(" not in s:
        continue
    version = keep_major_minor(version, ".")
    date_string = s[s.find("(") + 1 : s.find(")")]
    date_string = date_string.replace("Sept", "Sep")

    month_map = {
        "January": "Jan",
        "February": "Feb",
        "March": "Mar",
        "April": "Apr",
        "May": "May",
        "June": "Jun",
        "July": "Jul",
        "August": "Aug",
        "September": "Sep",
        "October": "Oct",
        "November": "Nov",
        "December": "Dec",
    }

    # Replace full month names with abbreviated ones
    for full_month, short_month in month_map.items():
        if full_month in date_string:
            date_string = date_string.replace(full_month, short_month)
            break
    try:
        date_obj = datetime.strptime(date_string, date_format)
        times.append((date_obj.strftime("%Y-%m-%d"), version))
    except:
        continue
times = sorted(times, key=lambda x: x[0])[::-1]


for task in data_tasks:

    created_at = task["created_at"].split("T")[0]
    for t in times:
        if t[0] < created_at:
            task["version"] = t[1]
            break


# Construct map of versions to task instances
map_v_to_t = {}
for t in data_tasks:
    if t["version"] not in map_v_to_t:
        map_v_to_t[t["version"]] = []
    map_v_to_t[t["version"]].append(t)

# Save matplotlib versioned data to repository
new_file_name = Path(PATH_TASK_PANDAS).stem + "_versions.json"

OUTPUT_VERSION_DIR = "artifacts/2_versioning"
with open(
    os.path.join(OUTPUT_VERSION_DIR, new_file_name),
    "w",
) as f:
    json.dump(data_tasks, fp=f)
