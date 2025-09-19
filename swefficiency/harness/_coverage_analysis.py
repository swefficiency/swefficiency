import ast
import os
import re
import coverage
import argparse
from unidiff import PatchSet
from pathlib import Path
import functools
import asttokens

from unidiff.constants import (
    LINE_TYPE_ADDED,
    LINE_TYPE_CONTEXT,
    LINE_TYPE_EMPTY,
    LINE_TYPE_REMOVED,
)

# Custom exception to quickly exit nested iterations.
class SuccessException(Exception):
    pass

# Get path to diff. 
parser = argparse.ArgumentParser()
parser.add_argument(
    "--patch-file", 
    default=Path("/tmp/patch.diff"), 
    type=Path,
    help="Patch file we're performing analysis on.")
args = parser.parse_args()

patch_set = PatchSet.from_filename(str(args.patch_file), encoding="utf-8")

# Glob all data in the directory, read each coverage file.
coverage_data_dir = "/tmp/coverage_data/"
all_coverage_filepaths = os.listdir(coverage_data_dir)

covered_tests = []

@functools.lru_cache(maxsize=None)
def get_file_lines(filepath):
    return open(filepath, 'r', encoding='utf-8').read().splitlines()

def get_import_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()
    atok = asttokens.ASTTokens(code, parse=True)
    type_hints = []
    
    import_lines = []

    for node in ast.walk(atok.tree):
        if isinstance(node, ast.Import):
            span = atok.get_text_positions(node, False)
            import_lines.extend(range(span[0][0], span[1][0] + 1))
        elif isinstance(node, ast.ImportFrom):
            span = atok.get_text_positions(node, False)
            import_lines.extend(range(span[0][0], span[1][0] + 1))
        elif any(isinstance(node, t) for t in (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            # Collect all explicit targets that could be a Name
            targets = []
            if isinstance(node, ast.AugAssign):
                targets = [node.target]
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:                                # ast.Assign
                targets = node.targets
            
            # This is to ignore things that declare like "__all__" etc.
            for target in targets:
                if isinstance(target, ast.Name) and target.id.startswith("__") and target.id.endswith("__"):
                    span = atok.get_text_positions(target, False)
                    import_lines.extend(range(span[0][0], span[1][0] + 1))
        
    return set(import_lines)


for filepath in all_coverage_filepaths:
    try:
        full_filepath = os.path.join(coverage_data_dir, filepath)
        covdb = coverage.CoverageData(full_filepath)
        covdb.read()

        # Assume each file is one test file.
        test_file_name = list(covdb.measured_contexts())[0]
        measured_files = covdb.measured_files()
        
        print(test_file_name)
    except:
        continue

    try:
        for patched_file in patch_set:
            # Patched files will omit "/testbed" prefix. The coverage data will not.
            abs_path_patched_file = os.path.join("/testbed", patched_file.path)
            
            # Case 1: The files of our edits should be covered in the coverage file.
            if abs_path_patched_file not in measured_files:
                continue
            
            # Get all import related lines.
            import_lines = get_import_lines(abs_path_patched_file)

            # Case 2: If the file is in the measured set, we should next check if the
            # hunks, which include both context and added/removed lines intersect with the
            # covered lines.
            lines_executed_in_covered_file = covdb.lines(abs_path_patched_file)

            for hunk in patched_file:
                # Note that we use the diff end to, since we measure cover post-patch apply.
                hunk_start_line = hunk.target_start
                
                # First, process hunks via two rules:
                #  1) If a line is removed (pre-edit): we need to identify the surrounding lines that exist in postedit.
                #  2) If a line is added: we can simply use that line in the post-edit coverage.
                
                post_edit_lines_of_interest = []
                
                last_surviving_line = None
                remove_chunk_first_surviving_line = None
                
                for line in hunk:
                    # Track pre-edit line that survived to post-edit.                    
                    if line.line_type == LINE_TYPE_REMOVED:
                        if remove_chunk_first_surviving_line is None:
                            remove_chunk_first_surviving_line = last_surviving_line
                    elif line.line_type == LINE_TYPE_ADDED:
                        # If line is added, just immediately track the line of interest.
                        post_edit_lines_of_interest.append(line.target_line_no)
                    else:
                        last_surviving_line = line
                        
                        # If we go back to a surviving line, we need to track the preedit lines
                        # and their post edit equivalents.
                        if remove_chunk_first_surviving_line is not None:
                            start_line_range = remove_chunk_first_surviving_line.target_line_no
                            end_line_range = last_surviving_line.target_line_no
                            
                            post_edit_lines_of_interest.extend(list(range(start_line_range, end_line_range)))

                            remove_chunk_first_surviving_line = None
                
                file_lines = get_file_lines(patched_file.path)
                
                actual_lines_executed_in_covered_file = []
                
                for l in lines_executed_in_covered_file:
                    # Note that we need to do this since coverage.py will consider lines covered at import
                    # time if one imports a specific function. See
                    # https://coverage.readthedocs.io/en/7.8.0/faq.html#q-why-are-my-function-definitions-marked-as-run-when-i-haven-t-tested-them
                    if any(file_lines[l - 1].strip().startswith(kw) for kw in ['class', 'def', "#"]):
                        continue
                    
                    # Ignore import related lines that are executed: we care more specifically about actual code that runs.
                    if l in import_lines:
                        continue
                    
                    actual_lines_executed_in_covered_file.append(l)
                
                post_edit_lines_of_interest = set(post_edit_lines_of_interest)
                actual_lines_executed_in_covered_file = set(actual_lines_executed_in_covered_file)

                
                # Check for intersection.
                hunk_intersects_with_coverage = actual_lines_executed_in_covered_file.intersection(post_edit_lines_of_interest)
                if hunk_intersects_with_coverage:
                    # Print intersecting lines.
                    
                    print(hunk)
                    print(post_edit_lines_of_interest)
                    print(sorted(actual_lines_executed_in_covered_file))

                    print(hunk_intersects_with_coverage)
                    
                    # Print lines from patched_file. 
                    lines = get_file_lines(patched_file.path)
                    number = list(hunk_intersects_with_coverage)[0]
                    
                    covered_lines = [lines[i - 1] for i in hunk_intersects_with_coverage]
                    for l in covered_lines:
                        print(l)
                    
                    # print([lines[i+1] for i in range(number - 3, number + 3)])
                
                    raise SuccessException()

    except SuccessException:
        covered_tests.append(test_file_name)

print("BEGIN_VALID_TESTS")
for test_file in covered_tests:
    print(test_file)
print("END_VALID_TESTS")