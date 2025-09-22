
RUN_ID=$1
NUM_WORKERS=$2
PREDICTIONS_PATH=$3

# If predictions_path is provided, add to args
ADDITIONAL_ARGS=""
if [ -n "$PREDICTIONS_PATH" ]; then
    ADDITIONAL_ARGS="--model_predictions $PREDICTIONS_PATH"
fi

# INSTANCE_IDS="pandas-dev__pandas-28447 dask__dask-5553 numpy__numpy-24663 numpy__numpy-25788 numpy__numpy-26599 pandas-dev__pandas-36317 pandas-dev__pandas-37064 pandas-dev__pandas-40840 pandas-dev__pandas-37130 pandas-dev__pandas-38353 pandas-dev__pandas-38379 pandas-dev__pandas-42841 pandas-dev__pandas-39332 pydata__xarray-7735 pydata__xarray-7796"
INSTANCE_IDS="pandas-dev__pandas-28447"


python swefficiency/harness/run_validation.py \
    --dataset_name swefficiency-anon/swefficiency \
    --run_id $RUN_ID \
    --cache_level env \
    --max_build_workers 16 \
    --max_workers $NUM_WORKERS \
    --timeout 7_200 \
    --run_perf true \
    --run_correctness true \
    --use_dockerhub_images true \
    --instance_ids $INSTANCE_IDS \
    $ADDITIONAL_ARGS

#     --run_perf_profiling true \

# --instance_ids pandas-dev__pandas-53731 pydata__xarray-9808 scikit-learn__scikit-learn-13290 scikit-learn__scikit-learn-13310 scikit-learn__scikit-learn-13987 scikit-learn__scikit-learn-14075 scikit-learn__scikit-learn-15049 scikit-learn__scikit-learn-15257 \
