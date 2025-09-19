#!/bin/bash
root=logs/run_evaluation

# Find directories containing 'run_instance.log' but do not contain 'ast_output.txt'
find "$root" -type f -name 'run_instance.log' -printf '%h\0' |
  sort -zu |
  while IFS= read -r -d '' dir; do
    # Check if 'ast_output.txt' does NOT exist in the directory
    if [ ! -e "$dir/ast_output.txt" ]; then
      printf '%s\n' "$dir/run_instance.log"
    fi
  done > investigate_failed.txt
