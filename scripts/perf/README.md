
# Data collection procedure

1. `cd swefficiency/collect`
2. `./run_get_tasks_pipeline.sh`
3. Back to root directory.
4. `scripts/perf/1_attribute_filter.sh`
5. `scripts/perf/2_run_get_versions.sh` or use specific `get_versions` python files in repo.
6. `scripts/perf/3_run_coverage_validation_multiple.sh` to identify PRs with changes covered by pytest coverage.
7. `scripts/perf/4_run_validation_correctness.sh` to get before and after correctness for testcases.
8. Go into run folder and label each with `workload.py` (`scripts/perf/get_perf_workload.py` is helpful to bring in the context).
9. `scripts/perf/4_run_validation_perf_workload.sh`
