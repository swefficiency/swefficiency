#!/bin/bash

# First arg is repo name, additinal args are instance ids to run
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <repo_name> [instance_ids...]"
    exit 1
fi

REPO_NAME=${1}
shift
LABELLED_INSTANCES_TO_RUN="$*"

LABELED_INSTANCE_OPTION=""
if [ -n "$LABELLED_INSTANCES_TO_RUN" ]; then
    LABELED_INSTANCE_OPTION="--instance_ids $LABELLED_INSTANCES_TO_RUN"
fi

echo "Running $REPO_NAME"

VERSION_DATA=artifacts/2_versioning/${REPO_NAME}-task-instances_attribute_versions.non-empty.jsonl

PREVIOUS_RUN_RESULTS=logs/run_evaluation/debugging_coverage_fix_import_${REPO_NAME}/gold/

# Get all the dirnames that don't contain "patch.diff"

export TMPDIR="/scratch"

echo "Running coverage validation for ${LABELLED_INSTANCES_TO_RUN}"
python swefficiency/harness/run_validation.py \
    --dataset_name $VERSION_DATA \
    --run_id debugging_coverage_fix_import_newalt_$REPO_NAME \
    --cache_level env \
    --max_build_workers 8 \
    --max_workers 4 \
    --timeout 3_600 \
    --run_coverage true \
    ${LABELED_INSTANCE_OPTION} 

