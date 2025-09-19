import argparse
import docker
import pandas as pd
from pathlib import Path

from swefficiency.harness.docker_build import build_annotate_instance_images
from swefficiency.harness.test_spec import (
    PERF_WORKLOAD_SCRIPT_LOCATION,
    TestSpec,
    make_test_spec,
)
from swefficiency.harness.run_validation import (
    get_dataset_from_preds,
    get_gold_predictions,
)
from swefficiency.harness.constants import KEY_INSTANCE_ID

parser = argparse.ArgumentParser(
    description="Process some annotation instances, upload to dockerhub, and dump CSV of tasks."
)

parser.add_argument(
    "--tasks_file", type=str, help="Instances with version file to pull info from."
)
parser.add_argument("--pr_file", type=str, help="PR info.")
parser.add_argument(
    "--instance_ids", type=str, help="Instance ids specifically to build and upload."
)
parser.add_argument(
    "--force_rebuild", default=True, help="Force rebuild of images each time."
)
parser.add_argument(
    "--max_build_workers", type=int, default=1, help="Max number of build"
)
parser.add_argument(
    "--output_dir",
    type=Path,
    default=Path("scripts/annotate"),
    help="Output directory.",
)
parser.add_argument("--run_id", type=str, default="annotate", help="Run id.")

args = parser.parse_args()

instance_ids = args.instance_ids.split(",")

dataset_name = args.tasks_file
split = "gold"
run_id = "annotate"
force_rebuild = args.force_rebuild
max_build_workers = args.max_build_workers
output_dir = args.output_dir
run_id = args.run_id

predictions = get_gold_predictions(dataset_name, split)
predictions = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}
dataset = get_dataset_from_preds(dataset_name, split, instance_ids, predictions, run_id)

# Get test specs.
test_specs: list[TestSpec] = []
for instance in dataset:
    test_specs.append(make_test_spec(instance))

instance_id_to_instance_mapping = {datum["instance_id"]: datum for datum in dataset}

# Build the annotated docker containers for upload.
client = docker.from_env()
successful, failed = build_annotate_instance_images(
    client,
    test_specs,
    force_rebuild,
    max_build_workers,
    instance_id_to_instance_mapping,
)

# Get the associated PR links.
pr_info = pd.read_json(args.pr_file, lines=True)

table_rows = []

images_to_upload = []

for test_spec in successful:
    instance_id = test_spec.instance_id
    image_name = test_spec.annotate_instance_image_key

    dockerhub_tag = f"{instance_id}"

    dockerhub_image = f"sweperf/sweperf_annotate:{instance_id}"

    pull_number = int(instance_id.split("-")[-1])
    pr_info_element = pr_info[pr_info["number"] == pull_number].iloc[0]
    pull_request_link = pr_info_element["html_url"]

    sample_docker_command = f"docker run --mount type=bind,src=<REPLACE_ME>,dst={PERF_WORKLOAD_SCRIPT_LOCATION} {dockerhub_image} /bin/bash -c 'chmod +x /perf.sh && /perf.sh && git apply -q /tmp/patch.diff && /perf.sh'"
    filter_out_noise = "2>&1 | grep -v '^+' | awk '/PERF_START:/ {inblock=1; next} /PERF_END:/ {inblock=0} inblock'"
    sample_docker_command = sample_docker_command + " " + filter_out_noise

    table_rows.append(
        {
            "instance_id": instance_id,
            "image_name": image_name,
            "dockerhub_image": dockerhub_image,
            "pull_request_link": pull_request_link,
            "sample_docker_command": sample_docker_command,
        }
    )
    images_to_upload.append((instance_id, image_name))

# Convert to pandas dataframe.
df = pd.DataFrame(table_rows)
output_file = output_dir / f"annotate_images_{run_id}.csv"
df.to_csv(output_file)

print("Written to CSV...")

commands_to_run = []

for instance_id, image_name in images_to_upload:
    commands_to_run.extend(
        [
            f"docker tag {image_name} sweperf/sweperf_annotate:{instance_id} && docker push sweperf/sweperf_annotate:{instance_id}"
        ]
    )

# Print command to upload images to dockerhub.
print("Docker CMD to run:", "; ".join(commands_to_run))

# Run each of the commands in subprocess in parallel.
import subprocess
from concurrent.futures import ThreadPoolExecutor


def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Successfully ran: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {command}\nError: {e}")


with ThreadPoolExecutor(max_workers=max_build_workers) as executor:
    executor.map(run_command, commands_to_run)
