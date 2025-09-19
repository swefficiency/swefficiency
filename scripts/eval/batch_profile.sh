# scripts/eval/run_profiler_only.sh profile_runs 12;
# docker rm -f $(docker ps -aq); docker system prune -a -f;

# scripts/eval/run_profiler_only.sh profile_runs 12 predictions/converted/sweagent_claude37sonnet.jsonl;
# docker rm -f $(docker ps -aq); docker system prune -a -f;

# START

# scripts/eval/run_profiler_only.sh profile_runs 12 predictions/converted/oh_gemini25flash.jsonl;
# docker rm -f $(docker ps -aq); docker system prune -a -f;

# scripts/eval/run_profiler_only.sh profile_runs 12 predictions/converted/oh_gpt5mini.jsonl;
# docker rm -f $(docker ps -aq); docker system prune -a -f;

scripts/eval/run_profiler_only.sh profile_runs 12 predictions/converted/oh_claude37sonnet.jsonl;
docker rm -f $(docker ps -aq); docker system prune -a -f;

