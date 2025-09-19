# /bin/bash
root=logs/run_evaluation

find "$root" -type f -name 'coverage_output.txt' -printf '%h\0' |
  sort -zu |
  while IFS= read -r -d '' dir; do
    if [ ! -e "$dir/covering_tests.txt" ]; then
      printf '%s\n' "$dir/coverage_output.txt"
    fi
  done > investigate.txt

# Iterate through each line in investigate.txt, read the file, and check for "timed out"
while IFS= read -r file; do
  if grep -q "Timeout" "$file"; then
    echo "Timeout found in: $file"
    # You can add additional actions here, like sending an alert or logging
  fi
done < investigate.txt

# This should print nothing if there are no timeouts.