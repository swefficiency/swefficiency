

for REPO_NAME in "${@}"; do
    echo "Running $REPO_NAME"

    VERSION_DATA=artifacts/2_versioning/${REPO_NAME}-task-instances_attribute_versions.non-empty.jsonl

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
  }' {} + | grep $REPO_NAME | xargs -r -n1 dirname | xargs -n1 basename)

    # echo "LABELLED_INSTANCES: $LABELLED_INSTANCES"

    PREVIOUS_RESULTS_DIR=logs/run_evaluation/debugging_coverage_fix_import_${REPO_NAME}/gold/

    # For each instance in LABELLED_INSTANCES, check if the instance_id dir has a file called covering_tests.txt
    # If it doesnt, then we need to run the coverage validation again.

    LABELLED_INSTANCES_ARRAY=($LABELLED_INSTANCES)

    LABELLED_INSTANCES_TO_RUN=""
    for instance in ${LABELLED_INSTANCES_ARRAY[@]}; do
        echo "Checking instance: $instance"
        if [ ! -f ${PREVIOUS_RESULTS_DIR}/${instance}/covering_tests.txt ]; then
            LABELLED_INSTANCES_TO_RUN="${LABELLED_INSTANCES_TO_RUN} ${instance}"
        fi
    done
    
    # Convert to space separated list
    LABELLED_INSTANCES_TO_RUN=$(echo $LABELLED_INSTANCES_TO_RUN | tr '\n' ' ')

    echo "Running coverage validation for ${LABELLED_INSTANCES_TO_RUN}"
    python swefficiency/harness/run_validation.py \
        --dataset_name $VERSION_DATA \
        --run_id debugging_coverage_fix_import_$REPO_NAME \
        --cache_level env \
        --max_build_workers 32 \
        --max_workers 16 \
        --timeout 3_600 \
        --run_coverage true 
        # --instance_ids $LABELLED_INSTANCES_TO_RUN 
        # --force_rebuild true
        # --allow_test_patch \
done


# Need to run the folowing

# modin
# sympy

# statsmodels
# spaCy
# scikit-image
# networkx

# Lower priority:
# pytensor
# scrapy
