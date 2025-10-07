
NUM_WORKERS=12
MODELS=(
    # "oh_claude37sonnet"
    "oh_gemini25flash"
    "oh_gpt5mini"
    "oh_deepseekv31"
    "oh_kimi_k2_0905"
    "oh_gemini25pro"
    "sweagent_gpt5mini"
    "sweagent_claude37sonnet"
    "sweagent_gemini25flash"
)

RUN_NAME="ground_truth_perf_isolation3"

# # # Run gold
# swefficiency eval --num_workers $NUM_WORKERS --run_id $RUN_NAME
# docker rm -f $(docker ps -aq); docker system prune -a -f;

for MODEL in "${MODELS[@]}"; do
    echo "Running evaluation for model: $MODEL"

    swefficiency eval --num_workers $NUM_WORKERS --run_id $RUN_NAME --prediction_path predictions/converted/$MODEL.jsonl
    docker rm -f $(docker ps -aq); docker system prune -a -f;

done