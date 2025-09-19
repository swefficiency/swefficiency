INSTANCE_ID="pandas-dev__pandas-52054"
THRESHOLD=5

PROFILE_RUN_DIR="logs/run_evaluation/profile_runs"
OUTPUT_DIR="analysis/adhoc/speedup_attribution/flame_graphs"

echo "preedit"
flameprof $PROFILE_RUN_DIR/gold/$INSTANCE_ID/workload_preedit_cprofile.prof --threshold $THRESHOLD > $OUTPUT_DIR/requests_preedit.svg

echo "post gold"
flameprof $PROFILE_RUN_DIR/gold/$INSTANCE_ID/workload_postedit_cprofile.prof --threshold $THRESHOLD > $OUTPUT_DIR/requests_postedit.svg

echo "post LLM"
flameprof $PROFILE_RUN_DIR/default_sweperf_claude__anthropic--claude-3-7-sonnet-20250219__t-0.00__p-1.00__c-1.00___swefficiency_full_test/$INSTANCE_ID/workload_postedit_cprofile.prof --threshold $THRESHOLD > $OUTPUT_DIR/requests_postedit_llm.svg


# Get the logs as well
flameprof $PROFILE_RUN_DIR/gold/$INSTANCE_ID/workload_preedit_cprofile.prof --threshold $THRESHOLD --format log > $OUTPUT_DIR/requests_preedit.log
flameprof $PROFILE_RUN_DIR/gold/$INSTANCE_ID/workload_postedit_cprofile.prof --threshold $THRESHOLD --format log > $OUTPUT_DIR/requests_postedit.log
flameprof $PROFILE_RUN_DIR/default_sweperf_claude__anthropic--claude-3-7-sonnet-20250219__t-0.00__p-1.00__c-1.00___swefficiency_full_test/$INSTANCE_ID/workload_postedit_cprofile.prof --threshold $THRESHOLD --format log > $OUTPUT_DIR/requests_postedit_llm.log
