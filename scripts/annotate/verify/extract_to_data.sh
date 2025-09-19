find logs/run_evaluation/debugging_coverage_fix_import_* \
  -type f -name 'covering_tests.txt' \
  ! -path '*_old/*' \
  | parallel --will-cite 'mkdir -p "./data/$(basename "$(dirname {})")"; cp -r "$(dirname {})" ./data/'
