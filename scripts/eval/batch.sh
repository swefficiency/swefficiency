

# scripts/eval/run_eval.sh ground_truth6 12;
# docker rm -f $(docker ps -aq); docker system prune -a -f;

# scripts/eval/run_eval.sh ground_truth5 12 predictions/converted/sweagent_gpt5mini.jsonl; 
# docker rm -f $(docker ps -aq); docker system prune -a -f;

scripts/eval/run_eval.sh ground_truth5 12 predictions/converted/sweagent_claude37sonnet.jsonl;
docker rm -f $(docker ps -aq); docker system prune -a -f;

scripts/eval/run_eval.sh ground_truth5 12 predictions/converted/sweagent_gemini25flash.jsonl;
docker rm -f $(docker ps -aq); docker system prune -a -f;

scripts/eval/run_eval.sh ground_truth5 12 predictions/converted/oh_gpt5mini.jsonl;
docker rm -f $(docker ps -aq); docker system prune -a -f;

scripts/eval/run_eval.sh ground_truth5 12 predictions/converted/oh_claude37sonnet.jsonl;
docker rm -f $(docker ps -aq); docker system prune -a -f;