# %%
import concurrent.futures
import functools
import multiprocessing
import subprocess
import time
import traceback
from pathlib import Path
from typing import List, Optional

import docker
import pandas as pd
import requests
from tqdm import tqdm

from swefficiency.harness.constants import KEY_INSTANCE_ID
from swefficiency.harness.docker_build import build_annotate_instance_images
from swefficiency.harness.log_parsers import MAP_REPO_TO_PARSER
from swefficiency.harness.run_validation import (
    get_dataset_from_preds,
    get_gold_predictions,
)
from swefficiency.harness.test_spec import (
    PERF_WORKLOAD_SCRIPT_LOCATION,
    TestSpec,
    make_test_spec,
)

# List all files in the directory.
data_path = Path("data/")
instance_path = Path("artifacts/2_versioning")

MAX_BUILD_WORKERS = 16  # Set the max build workers as needed.

# List all immediate subdirs.
# subdirs = [d for d in data_path.iterdir() if d.is_dir()]
# print("Number of subdirs:", len(subdirs))


@functools.cache
def get_instance_file(repo_name):
    """Get the instance file for a given repo name."""
    instance_file = (
        instance_path / f"{repo_name}-task-instances_attribute_versions.non-empty.jsonl"
    )
    df = pd.read_json(instance_file, lines=True)
    return df


observed_repos = set()
observed_instance_ids = set()

BASE_DOCKER_IMAGE_TEMPLATE = "ghcr.io/swefficiency/swefficiency-images:{instance_id}"
ANNOTATE_DOCKER_IMAGE_TEMPLATE = (
    "ghcr.io/swefficency/swefficiency_annotate:{instance_id}"
)

# def process_subdir(subdir):
#     instance_id = subdir.name

#     owner = instance_id.split("__")[0]
#     repo_name, remainder = instance_id.split("__")[1].rsplit("-", 1)
#     pull_number = int(remainder)

#     observed_repos.add(repo_name)
#     observed_instance_ids.add(instance_id)

#     # Get the instance file for the repo.
#     df = get_instance_file(repo_name)
#     row = df[df["instance_id"] == instance_id].iloc[0]

#     pr_link = f"https://github.com/{owner}/{repo_name}/pull/{pull_number}"

#     base_docker_image = BASE_DOCKER_IMAGE_TEMPLATE.format(instance_id=instance_id)
#     annotate_docker_image = ANNOTATE_DOCKER_IMAGE_TEMPLATE.format(instance_id=instance_id)

#     sample_docker_command = f"docker run --mount type=bind,src=<REPLACE_ME>,dst={PERF_WORKLOAD_SCRIPT_LOCATION} {annotate_docker_image} /bin/bash -c 'chmod +x /perf.sh && /perf.sh && git apply -q /tmp/patch.diff && /perf.sh'"
#     filter_out_noise = "2>&1 | grep -v '^+' | awk '/PERF_START:/ {inblock=1; next} /PERF_END:/ {inblock=0} inblock'"
#     sample_docker_command = sample_docker_command + " " + filter_out_noise

#     # Parse covering tests and test status.
#     covering_tests = (subdir / "covering_tests.txt").read_text().splitlines()
#     test_outputs = (subdir / "raw_coverage_data").iterdir()

#     covering_tests_status = {}

#     for test_output in test_outputs:
#         if test_output.is_file() and test_output.suffix == ".txt":
#             with open(test_output, 'r') as f:
#                 raw_test_text = f.read().strip()

#                 parser = MAP_REPO_TO_PARSER[f"{owner}/{repo_name}"]
#                 parsed_data = parser(raw_test_text)

#                 first_key = next(iter(parsed_data), None)
#                 if first_key is not None:
#                     test_name = first_key.split("::")[0]
#                     if test_name in covering_tests:
#                         covering_tests_status.update(parsed_data)

#     # Get all test statuses that are PASS.
#     pass2pass = [test for test, status in covering_tests_status.items() if "PASS" in status]
#     fail2fail = [test for test, status in covering_tests_status.items() if "FAIL" in status]

#     entry = {
#         "status": "NOT_APPROVED",  # Placeholder for approval status.

#         "instance_id": instance_id,
#         "pull_request_link": pr_link,

#         "workload": None,  # Placeholder for workload, if needed.
#         "notes": None,

#         # TODO: Not uploadable to sheet.
#         # "patch": row["patch"],
#         # "test_patch": row["test_patch"],
#         # "covering_tests": "\n".join(covering_tests),
#         # "PASS_TO_PASS": "\n".join(pass2pass),
#         # "FAIL_TO_FAIL": "\n".join(fail2fail),
#         "num_covering_tests": str(len(covering_tests)),

#         "base_docker_image": base_docker_image,
#         "annotate_dockerhub_image": annotate_docker_image,
#         "annotate_sample_docker_command": sample_docker_command,

#         "repo": f"{owner}/{repo_name}",
#         "created_at": row["created_at"],
#         "base_commit": row["base_commit"],
#         "version": row["version"],
#     }

#     return repo_name, instance_id, entry

# with multiprocessing.Pool(processes=MAX_BUILD_WORKERS) as pool:
#     results = []
#     for result in tqdm(pool.imap_unordered(process_subdir, subdirs), total=len(subdirs), desc="Processing subdirs"):
#         results.append(result)

# # Collect results.
# observed_repos = set()
# observed_instance_ids = set()

# entries = []
# for repo, instance_id, entry in results:
#     observed_repos.add(repo)
#     observed_instance_ids.add(instance_id)

#     # TODO: Use the entry somehow.
#     entries.append(entry)

# # Convert entries to a DataFrame. Make sure we keep original data types as string or appropriate.
# df = pd.DataFrame(entries)

# Convert 'created_at' to string format.
# df["created_at"] = df["created_at"].astype(str)

# df.sort_values(by=["repo", "instance_id"], inplace=True)
# df.reset_index(drop=True, inplace=True)

# print(df.head())

# import gspread
# from gspread_dataframe import set_with_dataframe

# gc = gspread.service_account()
# sh = gc.open("sweperf_all_data")

# # Write the dataframe to this worksheet.
# worksheet = sh.get_worksheet(0)  # Assuming you want to write to the first worksheet.
# worksheet.clear()  # Clear existing data.

# set_with_dataframe(worksheet, df, include_index=False, include_column_header=True)


# %%

# %%

# Build docker images and upload them to Docker Hub.
print("Building and uploading docker images...")
client = docker.from_env()

run_id = "image-upload-2024-10-30"  # Set the run ID as needed.

full_dataset = []

for repo in observed_repos:
    dataset_name = str(
        instance_path / f"{repo}-task-instances_attribute_versions.non-empty.jsonl"
    )
    split = "gold"
    predictions = get_gold_predictions(dataset_name, split)
    predictions = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}

    # Filter predictions based on observed instance IDs.
    filtered_predictions = {
        k: v for k, v in predictions.items() if k in observed_instance_ids
    }
    instance_ids = list(filtered_predictions.keys())
    filtered_dataset = get_dataset_from_preds(
        dataset_name, split, instance_ids, filtered_predictions, run_id
    )
    full_dataset.extend(filtered_dataset)

print(f"Total instances in dataset: {len(full_dataset)}")

# Sort dataset by instance_id for consistency.
full_dataset.sort(key=lambda x: x["instance_id"])
# full_dataset.reverse()

import requests


# Get tags for the images.
def get_docker_hub_tags(
    namespace: str,
    repo: str,
    page_size: int = 100,
    token: Optional[str] = None,
) -> List[str]:
    # Step 1: Get the token
    token_url = "https://ghcr.io/token"
    params = {"scope": f"repository:{namespace}/{repo}:pull"}

    token_response = requests.get(token_url, params=params)
    token_response.raise_for_status()
    token = token_response.json().get("token")

    if not token:
        raise Exception("Failed to retrieve token")

    # Step 2: Use the token to get the list of tags
    page = 1
    all_tags = []

    tags_url = f"https://ghcr.io/v2/{namespace}/{repo}/tags/list?n={page_size}"

    while True:
        headers = {"Authorization": f"Bearer {token}"}

        try:
            tags_response = requests.get(tags_url, headers=headers)
            tags_response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print("Finished")
            break

        tags_response = requests.get(tags_url, headers=headers)
        tags_response.raise_for_status()
        tags_data = tags_response.json()

        for tag in tags_data.get("tags", []):
            all_tags.append(tag)

        header_next_link = tags_response.headers.get("Link")
        if not header_next_link or 'rel="next"' not in header_next_link:
            break

        url_suffix = header_next_link.split("<", 1)[1].split(">", 1)[0]
        tags_url = "https://ghcr.io" + url_suffix

    return all_tags


base_tags = set(get_docker_hub_tags("TODO", "swefficiency"))
annotate_tags = set()

print(f"Base tags on Docker Hub: {len(base_tags)}")
print(f"Annotate tags on Docker Hub: {len(annotate_tags)}")

# Filter out instances that already have images on Docker Hub.
RUN_ALL = [
    # "astropy",
    # "dask",
    # "matplotlib",
    # "numpy",
    # "pandas", # Build this later :( need more scratch space.
    # "xarray",
    # "scikit-learn",
    # "scipy"
    # "sympy",
]

import datasets

# Get all the approved rows in the gspread sheet.


def get_from_hf():
    ds = datasets.load_dataset("swefficiency/swefficiency", split="test")
    ds_list = []
    for instance in ds:
        ds_list.append(instance)

    return ds_list


# full_dataset = get_from_hf()
full_dataset = get_from_hf()

print(f"Total instances in full dataset: {len(full_dataset)}")
print(f"Length of base tags: {len(base_tags)}")

full_dataset_instance_ids = {instance["instance_id"] for instance in full_dataset}
base_tags_set = set(base_tags)
print(full_dataset_instance_ids ^ base_tags_set)

filtered_dataset = []
for instance in full_dataset:
    instance_id = instance["instance_id"]

    if instance_id in base_tags and not any(
        option in instance_id for option in RUN_ALL
    ):
        # print(f"Skipping instance {instance_id} for repo {instance['repo']} as it already exists on Docker Hub.")
        continue

    # if any(option in instance_id for option in TO_IGNORE):
    #     print(f"Skipping instance {instance_id} for repo {instance['repo']} due to TO_IGNORE list.")
    #     continue

    print(f"Processing instance {instance_id} for repo {instance['repo']}")
    filtered_dataset.append(instance)

# force_rerun = ["pandas-dev__pandas-28447"]

# if force_rerun:
#     filtered_dataset = [instance for instance in full_dataset if instance['instance_id'] in force_rerun]


full_dataset = filtered_dataset
print("Length of filtered dataset:", len(full_dataset))


def build_and_upload_images(dataset):
    # Get test specs.
    test_specs: list[TestSpec] = []
    for instance in dataset:
        try:
            test_specs.append(make_test_spec(instance))
        except NotImplementedError:
            print(
                f"Skipping instance {instance['instance_id']} due to NotImplementedError."
            )
            continue

    instance_id_to_instance_mapping = {datum["instance_id"]: datum for datum in dataset}

    # Check which images are already on dockerhub.
    print("Checking existing images on Docker Hub...")

    # If no test specs left to build, return early.
    if not test_specs:
        print("All images already exist on Docker Hub. No need to build.")
        return

    # Build the annotated docker containers for upload.
    FORCE_REBUILD = (
        False  # Set to True to force rebuild, or False to skip if already built.
    )
    successful, failed = build_annotate_instance_images(
        client,
        test_specs,
        FORCE_REBUILD,
        MAX_BUILD_WORKERS,
        instance_id_to_instance_mapping,
    )

    # Parallelize the tagging and pushing of images.
    def tag_and_push_image(test_spec: TestSpec):
        base_docker_image = test_spec.instance_image_key
        annotate_image_name = test_spec.annotate_instance_image_key

        instance_id = test_spec.instance_id

        base_dockerhub_image = BASE_DOCKER_IMAGE_TEMPLATE.format(
            instance_id=instance_id
        )
        annotate_dockerhub_image = ANNOTATE_DOCKER_IMAGE_TEMPLATE.format(
            instance_id=instance_id
        )

        # For some reason, docker client pointing to podman socket doesn't work.
        # Lets just run the docker commands directly.
        base_dommand = f"docker tag {base_docker_image} {base_dockerhub_image} && docker push {base_dockerhub_image}"
        annotate_command = f"docker tag {annotate_image_name} {annotate_dockerhub_image} && docker push {annotate_dockerhub_image}"

        for cmd in [base_dommand]:
            print(f"Running command: {cmd}")
            while True:
                try:
                    result = subprocess.run(cmd, shell=True, check=True)
                    break
                except subprocess.CalledProcessError as e:
                    print(f"Error executing command:")
                    print(f"Return code: {e.returncode}")
                    print(f"Command: {e.cmd}")
                    print(f"Error output (stderr):")
                    print(e.stderr)

        return base_dockerhub_image, annotate_dockerhub_image

        # Filter out successful instances that are already completed.

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_BUILD_WORKERS
    ) as executor:
        futures = {
            executor.submit(tag_and_push_image, test_spec): test_spec
            for test_spec in successful
        }
        tqdm_bar = tqdm(total=len(futures), desc="Pushing images")
        for future in concurrent.futures.as_completed(futures):
            tqdm_bar.update(1)
            test_spec = futures[future]
            try:
                base_dockerhub_image, annotate_dockerhub_image = future.result()
                print(
                    f"Successfully pushed {base_dockerhub_image} and {annotate_dockerhub_image}"
                )
            except Exception as e:
                print(f"Failed to push image for {test_spec.instance_id}: {e}")
                traceback.print_exc()

    for test_spec in successful:
        # Remove images locally to free up space.
        base_docker_image = test_spec.instance_image_key
        annotate_image_name = test_spec.annotate_instance_image_key

        try:
            client.images.remove(base_docker_image, force=True)
            client.images.remove(annotate_image_name, force=True)
        except Exception as e:
            print(
                f"Failed to remove images {base_docker_image} and {annotate_image_name}: {e}"
            )

    # Prune unused Docker resources to free up space.
    print("Pruning unused Docker resources...")
    client.api.prune_containers()
    client.api.prune_images()
    client.api.prune_volumes()


BATCH_SIZE = 128  # Set the batch size for processing.
batched_dataset = [
    full_dataset[i : i + BATCH_SIZE] for i in range(0, len(full_dataset), BATCH_SIZE)
]

completed_file = Path("completed_images.txt")
completed_instances = set()
if completed_file.exists():
    with open(completed_file, "r") as f:
        completed_instances = set(f.read().splitlines())

for i, batch in enumerate(batched_dataset):
    # if completed_instances:
    #     batch = [instance for instance in batch if instance['instance_id'] not in completed_instances]

    # if len(batch) == 0:
    #     print(f"Batch {i + 1} is empty, skipping...")
    #     continue

    print("Instance ids: ", [instance["instance_id"] for instance in batch])

    print(f"Processing batch {i + 1} of {len(batched_dataset)}...")
    build_and_upload_images(batch)

    # Append completed instances to the file.
    with open(completed_file, "a") as f:
        for instance in batch:
            f.write(f"{instance['instance_id']}\n")

print("All images processed.")
