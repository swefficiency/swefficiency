
import argparse
import ast
import collections
import io
import multiprocessing
import os
from pathlib import Path
import token
import tokenize
import asttokens
import coverage

from unidiff import PatchSet, PatchedFile
from unidiff.constants import (
    LINE_TYPE_ADDED,
    LINE_TYPE_CONTEXT,
    LINE_TYPE_EMPTY,
    LINE_TYPE_REMOVED,
)

"""Helper functions."""

def get_post_edit_lines_of_interest(patched_file: PatchedFile) -> list:
    """Given a patched file, return the lines of interest after the edit."""
   
    last_surviving_line = None
    remove_chunk_first_surviving_line = None
    
    per_hunk_lines_of_interest = []
    
    for hunk in patched_file:
        post_edit_lines_of_interest = []
        
        for line in hunk:
            # Track pre-edit line that survived to post-edit.                    
            if line.line_type == LINE_TYPE_REMOVED:
                if remove_chunk_first_surviving_line is None:
                    remove_chunk_first_surviving_line = last_surviving_line
                    
            else:
                if line.line_type == LINE_TYPE_ADDED:
                    # If line is added, just immediately track the line of interest.
                    post_edit_lines_of_interest.append(line.target_line_no)

                last_surviving_line = line
                
                # If we go back to a surviving line (or added line right after a removed line), we need to track the preedit lines
                # and their post edit equivalents.
                if remove_chunk_first_surviving_line is not None:
                    start_line_range = remove_chunk_first_surviving_line.target_line_no
                    end_line_range = last_surviving_line.target_line_no
                    
                    if start_line_range is None or end_line_range is None:
                        print("Warning: start_line_range or end_line_range is None, skipping this range.")
                        print(start_line_range, end_line_range)
                        continue
                    
                    post_edit_lines_of_interest.extend(list(range(start_line_range, end_line_range + 1)))
                    remove_chunk_first_surviving_line = None
        per_hunk_lines_of_interest.append(set(post_edit_lines_of_interest))
    return per_hunk_lines_of_interest


def get_def_lines_to_ignore(patched_file_text: str):
    """
    Return a sorted list of line numbers that belong to every `def` statement
    (header only – decorators are *not* included; bodies are ignored).
    """
    ignored = set()
    tokens = tokenize.generate_tokens(io.StringIO(patched_file_text).readline)

    in_header = False          # are we currently inside a def header?
    depth = 0                  # (), [], {} nesting depth
    start_row = None           # line where the current def starts

    for tok_type, tok_str, (srow, _), (_, _), _ in tokens:
        # 1. Detect the start of a function definition
        if tok_type == token.NAME and tok_str == "def":
            in_header = True
            depth = 0
            start_row = srow

        # 2. While in the header, track nesting & look for the terminating colon
        elif in_header:
            if tok_str in "([{":
                depth += 1
            elif tok_str in ")]}":
                depth -= 1
            elif tok_str == ":" and depth == 0:
                # We've reached the colon that ends the signature
                ignored.update(range(start_row, srow + 1))
                in_header = False          # reset for the next def

    return sorted(ignored)

def safe_get_text_positions(atok, node):
    if hasattr(atok, "get_text_positions"):            # asttokens 3+
        return atok.get_text_positions(node, False)

    # ---- asttokens 2.x fallback ------------------------------------------
    start_off, end_off = atok.get_text_range(node)
    # `get_position` exists in all versions and converts byte offset → (line, col)
    offset_to_line = atok._line_numbers.offset_to_line  # internal but stable
    return offset_to_line(start_off), offset_to_line(end_off)

def get_lines_to_ignore(patched_file_text):
    """Get the lines to ignore from the patched file text. This includes imports, private assignments, and function signatures."""
    atok = asttokens.ASTTokens(patched_file_text, parse=True)
    
    ignore_lines = []

    for node in ast.walk(atok.tree):
        # CASE 1: Ignore import changes, these are certainly not performance improving.
        if isinstance(node, ast.Import):
            span = safe_get_text_positions(atok, node) 
            ignore_lines.extend(range(span[0][0], span[1][0] + 1))
        elif isinstance(node, ast.ImportFrom):
            span = safe_get_text_positions(atok, node) 
            ignore_lines.extend(range(span[0][0], span[1][0] + 1))
        
        # CASE 2: Ignore changes to private assignments like "__all__" etc.
        elif any(isinstance(node, t) for t in (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            # Collect all explicit targets that could be a Name
            targets = []
            if isinstance(node, ast.AugAssign):
                targets = [node.target]
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:               # ast.Assign
                targets = node.targets
            
            # This is to ignore things that declare like "__all__" etc.
            for target in targets:
                if isinstance(target, ast.Name) and target.id.startswith("__") and target.id.endswith("__"):
                    span = safe_get_text_positions(atok, node) 
                    ignore_lines.extend(range(span[0][0], span[1][0] + 1))
                    break
        
        # CASE 3: Ignore changes to argument signature lines.
        elif isinstance(node, ast.arguments):
            span = safe_get_text_positions(atok, node) 
            ignore_lines.extend(range(span[0][0], span[1][0] + 1))
        
    # CASE 4: Get line numbers of empty lines. Note that asttokens is 1-indexed.
    # We should ignore these, since we care about actual content.
    for line_number, line in enumerate(patched_file_text.splitlines()):
        if line.strip() == "":
            ignore_lines.append(line_number + 1)
            
    # CASE 5: Function signatures are run during imports and declarations, we should also ignore.
    ignore_lines.extend(get_def_lines_to_ignore(patched_file_text))
        
    return set(ignore_lines)

def get_definition_nodes(patch_set: PatchSet):
    patched_file_to_relevant_hunks = collections.defaultdict(list)
    patched_file_to_lines_to_ignore = collections.defaultdict(set)
    original_lines_of_interest = collections.defaultdict(set)
    
    for patched_file in patch_set:
        # Get the post edit lines of interest.
        post_edit_lines_of_interest = get_post_edit_lines_of_interest(patched_file)
        
        # Patched files will omit "/testbed" prefix. The coverage data will not.
        if args.disable_testbed_prefix:
            abs_path_patched_file = os.path.join("/", patched_file.path)
        else:
            # We assume that the patched files are in a directory called "/testbed".
            abs_path_patched_file = os.path.join("/testbed", patched_file.path)
        
        if not abs_path_patched_file.endswith(".py"):
            for hunk_lines in post_edit_lines_of_interest:
                original_lines_of_interest[abs_path_patched_file].update(hunk_lines)
            continue
        
        # Specifically for python files, we want to dig in and get the body coverage.
        patched_file_text = open(abs_path_patched_file, "r", encoding="utf-8").read()

        # Get ast representation of the patched file.
        atok = asttokens.ASTTokens(patched_file_text, parse=True)
        ignore_lines = get_lines_to_ignore(patched_file_text)
        
        print("Processing file:", patched_file)
        print("Lines to ignore:", ignore_lines)
        
        patched_file_to_lines_to_ignore[abs_path_patched_file] = ignore_lines
        
        for hunk_lines_of_interest, hunk in zip(post_edit_lines_of_interest, patched_file):
            filtered_lines = [line for line in hunk_lines_of_interest if line not in ignore_lines]
            
            if len(filtered_lines) == 0:
                continue
            
            original_lines_of_interest[abs_path_patched_file].update(filtered_lines)
            min_line = min(filtered_lines)
            max_line = max(filtered_lines)
            
            # Now walk through the tree and find the smallest span that contains the min and max.
            latest_node = None
            latest_node_span_delta = None
            latest_node_span = None
            for node in ast.walk(atok.tree):
                # TODO: For now, we assume that the body coverage is only in functions and classes.
                if not any(isinstance(node, t) for t in (ast.FunctionDef, ast.ClassDef)):
                    continue

                span = safe_get_text_positions(atok, node) 
                span_delta = span[1][0] - span[0][0]

                if span[0][0] <= min_line and span[1][0] >= max_line:
                    if latest_node is None or span_delta < latest_node_span_delta:
                        latest_node = node
                        latest_node_span_delta = span_delta
                        latest_node_span = (span[0][0], span[1][0])
            
            if latest_node is not None:
                print(latest_node.name)
                patched_file_to_relevant_hunks[abs_path_patched_file].append((latest_node, latest_node_span))
    
    return patched_file_to_relevant_hunks, original_lines_of_interest, patched_file_to_lines_to_ignore


def get_lines_of_interest(definition_nodes, original_lines_of_interest, lines_to_ignore):
    """Get the lines of interest from the definition nodes."""
    lines_of_interest = collections.defaultdict(set)
    for patched_file, nodes in definition_nodes.items():
        for node, (start_line, end_line) in nodes:
            lines_of_interest[patched_file].update(range(start_line, end_line + 1))
    
    # Now add the original lines of interest.
    for patched_file, lines in original_lines_of_interest.items():
        lines_of_interest[patched_file].update(lines)
    
    # Now remove the lines to ignore (imports, private assignments, function signatures, empty lines).
    for patched_file, lines in lines_of_interest.items():
        lines_of_interest[patched_file] = lines - lines_to_ignore[patched_file]
    
    return lines_of_interest

parser = argparse.ArgumentParser()
parser.add_argument(
    "--patch-file", 
    default=Path("/tmp/patch.diff"), 
    type=Path,
    help="Patch file we're performing analysis on.")
parser.add_argument(
    "--disable-testbed-prefix",
    action="store_true",
    help="Disable the /testbed prefix for the patched files. This is useful for local testing.")
args = parser.parse_args()

patch_set = PatchSet.from_filename(str(args.patch_file), encoding="utf-8")

definition_nodes, original_lines_of_interest, lines_to_ignore = get_definition_nodes(patch_set)
lines_of_interest = get_lines_of_interest(definition_nodes, original_lines_of_interest, lines_to_ignore)

print("lines_of_interest:", lines_of_interest)

# Now for all coverage files, we want to get the lines of interest.
# Glob all data in the directory, read each coverage file.
coverage_data_dir = "/tmp/coverage_data/"
all_coverage_filepaths = os.listdir(coverage_data_dir)
if len(all_coverage_filepaths) == 0:
    print("No coverage files found.")
    exit(0)
    
def handle_coverage_file(coverage_filepath):
    print("Processing coverage file:", coverage_filepath)
    try:
        full_filepath = os.path.join(coverage_data_dir, coverage_filepath)
        covdb = coverage.CoverageData(full_filepath)
        covdb.read()
        
        print(f"Loaded coverage data from {full_filepath}")
        print(covdb.measured_files())
        
        # Assume each file is one test file.
        test_file_name = list(covdb.measured_contexts())[0]  
        
        print(f"Test file name: {test_file_name}")
        
    except Exception as e:
        print(f"Error reading coverage file {coverage_filepath}: {e}")
        return None

    for file_path, relevant_lines in lines_of_interest.items():
        try:
            lines_executed_in_covered_file = covdb.lines(file_path)
            if lines_executed_in_covered_file is None:
                continue
            
            print("File path:", file_path)
            print("Lines executed in covered file:", lines_executed_in_covered_file)
            print("Lines of interest:", relevant_lines)

            # Check intersection of lines executed and lines of interest.
            lines_executed_in_covered_file_set = set(lines_executed_in_covered_file)
            lines_of_interest_set = set(relevant_lines)
            intersection = lines_executed_in_covered_file_set.intersection(lines_of_interest_set)
            
            if len(intersection) > 0:
                return test_file_name

        except Exception as e:
            print(f"Error finding lines in {file_path}: {e}")
            continue
    
    return None
        

# Run in multiprocessing.
with multiprocessing.Pool(processes=8) as pool:
    covered_tests = pool.map(handle_coverage_file, all_coverage_filepaths)

# Filter out None results.
covered_tests = [test_file for test_file in covered_tests if test_file is not None]

print("BEGIN_VALID_TESTS")
for test_file in covered_tests:
    print(test_file)
print("END_VALID_TESTS")
