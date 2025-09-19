

# ./scripts/perf/4_run_validation_correctness.sh matplotlib numpy scikit-learn pandas xarray dask

# matplotlib done
# scikit-learn done
# xarray done
# numpy done
# pandas wip
# dask wip


for REPO_NAME in "${@}"; do
    echo "Running $REPO_NAME"

    VERSION_DATA=artifacts/2_versioning/${REPO_NAME}-task-instances_attribute_versions.non-empty.jsonl

    # # If reponame is pandas:
    # if [ "${REPO_NAME}" == "pandas" ]; then
    # Get folder names where folder does not have "perf_summary.txt"
    # LABELLED_INSTANCES=$(find logs/run_evaluation/perf_coverage_$REPO_NAME/gold/* -maxdepth 1 -type d ! -name '.' -exec bash -c '[ ! -f "$0/perf_summary.txt" ] && basename "$0"' {} \; | xargs -n1 basename | tr '\n' ' ')
    # echo "Found instances: $LABELLED_INSTANCES"

    PERF_CLEAN_RUN_DIR=logs/run_evaluation/perf_coverage_clean_run/gold

    # Keep delta's more than 1 standard deviation away from the mean.
    LABELLED_INSTANCES=$(find ${PERF_CLEAN_RUN_DIR} \
  -type f -path "*/perf_summary.txt" -exec awk '
  BEGIN { OFS = "\t" }
  /Before Mean:/ { bm = $3 }
  /Before SD:/   { bsd = $3 }
  /After Mean:/  { am = $3 }
  /After SD:/    { asd = $3 }
  /Improvement:/ {
    delta = am - bm
    maxsd = (bsd > asd ? bsd : asd)
    if (maxsd != 0 && sqrt((delta / maxsd)^2) > 1 && delta < 0) {
      print FILENAME
    }
  }' {} + | grep $REPO_NAME | xargs -r -n1 dirname | xargs -n1 basename | tr '\n' ' ')

    RUN_NAME="perf_coverage_correctness_${REPO_NAME}"

    echo "Running validation for instances: $LABELLED_INSTANCES"

    LABELLED_INSTANCES=pandas-dev__pandas-30747

    python swefficiency/harness/run_validation.py \
        --dataset_name $VERSION_DATA \
        --run_id $RUN_NAME \
        --cache_level instance \
        --max_build_workers 8 \
        --max_workers 4 \
        --timeout 3_600 \
        --run_coverage false \
        --run_perf false \
        --run_correctness true \
        --empty_patch false \
        --instance_ids $LABELLED_INSTANCES
        # --force_rebuild true # Only need to do this once.
    
    # python swefficiency/harness/run_validation.py \
    #     --dataset_name $VERSION_DATA \
    #     --run_id $RUN_NAME \
    #     --cache_level instance \
    #     --max_build_workers 4 \
    #     --max_workers 8 \
    #     --timeout 3_600 \
    #     --run_coverage false \
    #     --run_perf false \
    #     --run_correctness true \
    #     --instance_ids $LABELLED_INSTANCES \
    #     --empty_patch true \

    # docker system prune -f
done
