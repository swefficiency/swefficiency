# Copy PRs at this point to get instances that have covering tests and start labeling workloads.

find logs/run_evaluation/ -type f -name covering_tests.txt | grep "debugging_coverage_" | xargs -n1 dirname
