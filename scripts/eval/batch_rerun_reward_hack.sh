

# scripts/eval/run_eval.sh ground_truth6 12;
# docker rm -f $(docker ps -aq); docker system prune -a -f;

# scripts/eval/run_eval.sh ground_truth5 12 predictions/converted/sweagent_gpt5mini.jsonl; 
# docker rm -f $(docker ps -aq); docker system prune -a -f;

scripts/eval/run_eval.sh ground_truth5 12 predictions/converted/sweagent_claude37sonnet.jsonl rerun_sweagent_claude37sonnet.txt
docker rm -f $(docker ps -aq); docker system prune -a -f;

scripts/eval/run_eval.sh ground_truth5 12 predictions/converted/sweagent_gpt5mini.jsonl rerun_sweagent_gpt5mini.txt
docker rm -f $(docker ps -aq); docker system prune -a -f;

scripts/eval/run_eval.sh ground_truth5 12 predictions/converted/sweagent_gemini25flash.jsonl rerun_sweagent_gemini25flash.txt
docker rm -f $(docker ps -aq); docker system prune -a -f;

scripts/eval/run_eval.sh ground_truth5 12 predictions/converted/sweagent_gemini25flash.jsonl rerun_oh_gemini25flash.txt
docker rm -f $(docker ps -aq); docker system prune -a -f;

scripts/eval/run_eval.sh ground_truth5 12 predictions/converted/oh_gpt5mini.jsonl rerun_oh_gpt5mini.txt
docker rm -f $(docker ps -aq); docker system prune -a -f;

scripts/eval/run_eval.sh ground_truth5 12 predictions/converted/oh_claude37sonnet.jsonl rerun_oh_claude37sonnet.txt
docker rm -f $(docker ps -aq); docker system prune -a -f;