# REPO_NAMES=("scipy" "sympy" "astropy" "spaCy")
REPO_NAMES=("sympy")

# Need to run covering tests first for.
# TODO: modin, scikit-image, networkx, pytensor, scrapy, statsmodels


for REPO_NAME in "${REPO_NAMES[@]}"; do
    ARTIFACTS_DIR=artifacts

    TASKS_FILE=${ARTIFACTS_DIR}/2_versioning/${REPO_NAME}-task-instances_attribute_versions.non-empty.jsonl
    PR_FILE=${ARTIFACTS_DIR}/pull_requests/${REPO_NAME}-prs.jsonl

    python scripts/annotate/upload_annotate_docker_and_get_csv_real.py \
        --tasks_file $TASKS_FILE \
        --pr_file $PR_FILE \
        --coverage_run_dir logs/run_evaluation/debugging_coverage_$REPO_NAME/gold/ \
        --max_build_workers 8 \
        --run_id $REPO_NAME
done
