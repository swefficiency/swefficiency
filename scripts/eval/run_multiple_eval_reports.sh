
EVAL_DIR="logs/run_evaluation"

# GOLD_RUN_NAME="$EVAL_DIR/ground_truth5/gold"
GOLD_RUN_NAME="$EVAL_DIR/ground_truth10/gold"
GOLD_RUN_NAME="$EVAL_DIR/ground_truth_latest/gold"

MODEL_NAMES=(
    # "$EVAL_DIR/ground_truth5/gold"
    # "$EVAL_DIR/ground_truth10/gold"
    # "$EVAL_DIR/ground_truth11/gold"
    # "$EVAL_DIR/ground_truth12/gold"
    # "$EVAL_DIR/ground_truth13/gold"
    # "$EVAL_DIR/ground_truth14/gold"
    # "$EVAL_DIR/ground_truth15/gold"
    # "$EVAL_DIR/ground_truth16/gold" # sparse broken
    # "$EVAL_DIR/ground_truth17/gold"
    # "$EVAL_DIR/ground_truth18/gold"
    # "$EVAL_DIR/ground_truth19/gold"
    # "$EVAL_DIR/ground_truth21/gold"
    "$EVAL_DIR/ground_truth_latest/gold"
    "$EVAL_DIR/ground_truth5/gpt-5-mini_maxiter_100_N_v0.51.1-no-hint-run_1"
    "$EVAL_DIR/ground_truth5/us.anthropic.claude-3-7-sonnet-20250219-v1_0_maxiter_100_N_v0.51.1-no-hint-run_1"
    "$EVAL_DIR/ground_truth5/gemini-2.5-flash_maxiter_100_N_v0.51.1-no-hint-run_1"
    "$EVAL_DIR/ground_truth5/deepseek-reasoner_maxiter_100_N_v0.51.1-no-hint-run_1"
    "$EVAL_DIR/ground_truth5/default_sweperf_openai__openai--gpt-5-mini__t-1.00__p-1.00__c-1.00___swefficiency_full_test"
    "$EVAL_DIR/ground_truth5/default_sweperf_claude__anthropic--claude-3-7-sonnet-20250219__t-0.00__p-1.00__c-1.00___swefficiency_full_test"
    "$EVAL_DIR/ground_truth5/default_sweperf_gemini__gemini--gemini-2.5-flash__t-0.00__p-1.00__c-1.00___swefficiency_full_test"
)

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Evaluating model: $MODEL_NAME"
    python scripts/eval/get_report.py \
        --gold_run "$GOLD_RUN_NAME" \
        --pred_run "$MODEL_NAME" \
        --num_workers 4 \
        --output_dir "eval_reports"
done
