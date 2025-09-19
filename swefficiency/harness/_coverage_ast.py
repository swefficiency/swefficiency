import argparse
import ast
import collections
import fnmatch
import functools
import multiprocessing
import os
from datetime import datetime

import asttokens
import jedi
from intervaltree import Interval, IntervalTree
from unidiff import PatchSet
from unidiff.constants import LINE_TYPE_ADDED, LINE_TYPE_REMOVED

# TODO: Switch this if needed.
FILE_IGNORE_FN = lambda filepath: not filepath.startswith("/testbed")
# FILE_IGNORE_FN = lambda filepath: not 'SWE-Perf' in filepath

print("Starting coverage AST script...")

# Argument parsing.
argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--repo_directory",
    type=str,
    default="/testbed/",
    help="File to get source file imports for",
)
argparser.add_argument(
    "--test_files",
    type=str,
    default=None,
    help="Comma delimited test files of interest",
)
argparser.add_argument(
    "--source_files",
    type=str,
    default=None,
    help="Comma delimited source files of interest",
)
argparser.add_argument("--outfile", type=str, help="Outfile to write relevant tests to")
argparser.add_argument(
    "--ignore_depth", type=int, default=1, help="Depth of AST search to ignore"
)
argparser.add_argument(
    "--patch_file", type=str, default="/tmp/patch.diff", help="Patch file to use"
)
args = argparser.parse_args()


def find_test_files(repo_directory):
    test_files = []
    for root, dirs, files in os.walk(repo_directory):
        for filename in files:
            if "test" in str(filename):
                test_files.append(os.path.join(root, filename))
    return test_files


print("Finding test files...")

if args.test_files:
    test_files = args.test_files.split(",")
else:
    test_files = set(find_test_files(args.repo_directory))
source_files = set(args.source_files.split(",")) if args.source_files else None


# Helper functions for range intersection and management.
def ranges_intersect(start1, end1, start2, end2):
    return not (end1 < start2 or end2 < start1)


assert ranges_intersect(1, 2, 3, 4) == False
assert ranges_intersect(1, 3, 2, 4) == True


def coordinates_intersect(range1, range2):
    """
    range1 and range2 are tuples: ((start_line, start_col), (end_line, end_col))
    """
    start1, end1 = range1
    start2, end2 = range2

    if start1 is None or end1 is None or start2 is None or end2 is None:
        return False

    line1_start, col1_start = start1
    line1_end, col1_end = end1

    line2_start, col2_start = start2
    line2_end, col2_end = end2

    # Check if line ranges intersect
    lines_overlap = ranges_intersect(line1_start, line1_end, line2_start, line2_end)

    return lines_overlap


assert coordinates_intersect(((1, 0), (2, 0)), ((3, 0), (4, 0))) == False
assert coordinates_intersect(((1, 0), (2, 0)), ((2, 0), (3, 0))) == True


# File range tracker.
class FileLineRangeTracker:
    def __init__(self):
        self.file_trees = collections.defaultdict(IntervalTree)

    def add_range(self, filename, start_line, end_line):
        """
        Add a line range [start_line, end_line), merging with overlaps or adjacent.
        """
        tree = self.file_trees[filename]

        # Find overlapping or adjacent intervals
        overlapping = tree.overlap(start_line - 1, end_line + 1)
        to_merge = list(overlapping)

        # Remove them
        tree.difference_update(to_merge)

        # Merge into new range
        new_start = min([start_line] + [iv.begin for iv in to_merge])
        new_end = max([end_line] + [iv.end for iv in to_merge])

        # Add merged interval
        tree.add(Interval(new_start, new_end))

    def get_ranges(self, filename=None):
        """
        Return sorted list of (start, end) intervals.
        """
        if filename is None:
            return sorted(
                (iv.begin, iv.end, filename)
                for filename, tree in self.file_trees.items()
                for iv in tree
            )

        return sorted((iv.begin, iv.end, filename) for iv in self.file_trees[filename])

    def is_range_covered(self, filename, start_line, end_line):
        """
        Check if [start_line, end_line) is fully contained in any one interval.
        """
        tree = self.file_trees[filename]
        for iv in tree[start_line]:  # returns intervals that include start_line
            if iv.begin <= start_line and iv.end >= end_line:
                return True
        return False

    def has_file(self, filename):
        return filename in self.file_trees


def identify_type_hints(code):
    atok = asttokens.ASTTokens(code, parse=True)
    type_hints = []

    for node in ast.walk(atok.tree):
        if isinstance(node, ast.FunctionDef):
            for arg in node.args.args:
                if arg.annotation:
                    span = atok.get_text_positions(arg.annotation, False)
                    # print("arg:", arg.annotation)
                    type_hints.append(span)
            if node.returns:
                span = atok.get_text_positions(node.returns, False)
                # print("return:", node.returns)
                type_hints.append(span)
        elif isinstance(node, ast.AnnAssign):
            if node.annotation:
                span = atok.get_text_positions(node.annotation, False)
                # print("ann_assign:", node.annotation)
                type_hints.append(span)

    return type_hints


@functools.lru_cache(maxsize=1024)
def get_symbols_in_script(python_file):
    python_file_text = open(python_file, "r", encoding="utf-8").read()
    python_file_script = jedi.Script(python_file_text, path=python_file)

    return python_file_script.get_names(
        all_scopes=True, definitions=False, references=True
    ), identify_type_hints(python_file_text)


# For a single python file, parse out relevant definitions.
def process_file(python_file, depth=0, line_start=None, line_end=None):

    # Iterate over all symbols in test file script. Note that we want all scopes (including
    # within classes/functions/etc, can ignore file definitions, but care about refs).
    all_names, type_hints = get_symbols_in_script(python_file)

    relevant_source_files = dict()

    # Filter out names not within the line limit.
    pruned_names = []
    for name in all_names:
        name_start = name.get_definition_start_position()
        name_end = name.get_definition_end_position()

        if name_start is None or name_end is None:
            continue

        if str(name.module_path) != python_file:
            continue

        if any(
            coordinates_intersect(hint, (name_start, name_end)) for hint in type_hints
        ):
            # Ignore type hints
            continue

        if coordinates_intersect(
            (name_start, name_end), ((line_start, 0), (line_end, 0))
        ):
            pruned_names.append(name)

    for name in pruned_names:
        try:
            dependent_names = name.goto(
                follow_builtin_imports=False, follow_imports=True
            )
        except:
            continue

        for dn in dependent_names:
            module_file = str(dn.module_path)

            definition_start = dn.get_definition_start_position()
            definition_end = dn.get_definition_end_position()

            # If just a import reference and not actual usage, then start and end is none.
            if definition_start is None or definition_end is None:
                continue

            relevant_source_files[
                (module_file, definition_start[0], definition_end[0])
            ] = depth

            if module_file.endswith(".so") or module_file.endswith(".pyi"):
                # Probably a cython file, add possible files to the visited.
                module_file_before_point = module_file.split(".")[0]

                for cython_ext in ["pxd", "pyx", "pxi", "pxd.tp"]:
                    possible_cython_file = module_file_before_point + "." + cython_ext

                    # TODO: Cython handling?
                    relevant_source_files[(possible_cython_file, 0, 0)] = depth

    return python_file, relevant_source_files


def process_patch_set(range_tracker: FileLineRangeTracker, patch_set):

    for patched_file in patch_set:
        # Patched files will omit "/testbed" prefix. The coverage data will not.
        abs_path_patched_file = os.path.join("/testbed", patched_file.path)

        # Case 1: The files of our edits should be covered in the interval tree.
        if not range_tracker.has_file(abs_path_patched_file):
            continue

        # Case 2: If the file is in the measured set, we should next check if the
        # hunks, which include both context and added/removed lines intersect with the
        # covered lines.
        lines_executed_in_covered_file = [
            ln
            for start, end in range_tracker.get_ranges(abs_path_patched_file)
            for ln in range(start, end)
        ]

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
                        start_line_range = (
                            remove_chunk_first_surviving_line.target_line_no
                        )
                        end_line_range = last_surviving_line.target_line_no

                        post_edit_lines_of_interest.extend(
                            list(range(start_line_range, end_line_range))
                        )

                        remove_chunk_first_surviving_line = None

            post_edit_lines_of_interest = set(post_edit_lines_of_interest)
            lines_executed_in_covered_file = set(lines_executed_in_covered_file)

            # Check for intersection.
            print(hunk)
            print(post_edit_lines_of_interest)
            print(lines_executed_in_covered_file)
            hunk_intersects_with_coverage = lines_executed_in_covered_file.intersection(
                post_edit_lines_of_interest
            )

            if len(hunk_intersects_with_coverage) > 0:
                return True

    return False


# For a test file, visit all of its references, then references of those references etc.
def process_test_file(test_file_tuple, patch_set=None):
    index, test_file = test_file_tuple
    print("Processing test file:", test_file, index, datetime.now(), flush=True)

    visited_tree = FileLineRangeTracker()

    # Count number of lines in test_file quickly.
    with open(test_file, "rb") as f:
        test_file_length = sum(1 for line in f)

    files_to_explore = [(test_file, 0, test_file_length, 0)]

    # This works by exploring each file: (1) explore entirety of test file (2) subregions of dependent files, etc.
    while len(files_to_explore) > 0:
        current_file, line_start, line_end, depth = files_to_explore.pop(0)

        if visited_tree.is_range_covered(current_file, line_start, line_end):
            continue

        # Get stem of file and check if it has the word test in it.
        if "test" in os.path.basename(current_file) and depth > 0:
            continue

        # print(depth, current_file, len(files_to_explore), flush=True)

        visited_tree.add_range(current_file, line_start, line_end + 1)

        if current_file.endswith(".py"):
            _, additional_relevant_source_files = process_file(
                current_file, line_start=line_start, line_end=line_end, depth=depth
            )

            if depth > args.ignore_depth:
                # Exit at this point, since its a BFS queue.
                continue

            temp_tree = FileLineRangeTracker()
            for (
                dependent_file,
                dependent_file_start,
                dependent_file_end,
            ) in additional_relevant_source_files.keys():
                if FILE_IGNORE_FN(dependent_file):
                    continue

                temp_tree.add_range(
                    dependent_file, dependent_file_start, dependent_file_end + 1
                )

            for interval_begin, interval_end, filename in temp_tree.get_ranges():
                files_to_explore.append(
                    (filename, interval_begin, interval_end - 1, depth + 1)
                )

            del temp_tree

    relevant_source_files = dict()

    for interval_begin, interval_end, filename in visited_tree.get_ranges():
        relevant_source_files[filename] = 0

    # Filter out keys that don't match desired source files.
    relevant_source_files = set(
        key
        for key, value in relevant_source_files.items()
        if source_files is None or key in source_files
    )

    if patch_set is not None:
        matches = process_patch_set(visited_tree, patch_set)

        if not matches:
            return test_file, set()
        else:
            return test_file, relevant_source_files

    else:
        return test_file, relevant_source_files


# Parallel execution
tests_to_relevant_source_files = collections.defaultdict(set)

current_time = datetime.now()
print("Current time:", current_time)

print("Number of test files:", len(test_files))
print("Processing test files...")
with multiprocessing.Pool(processes=8) as pool:  # Adjust number of processes as needed
    for test_file, result in pool.imap_unordered(
        process_test_file, enumerate(test_files)
    ):
        tests_to_relevant_source_files[test_file].update(result)

time_elapsed = datetime.now() - current_time
print("Time elapsed for processing test files:", time_elapsed)

# TODO: Sort?
sorted_tests_to_relevant_source_files = tests_to_relevant_source_files

print(tests_to_relevant_source_files)

# Filter out anything that is empty.
sorted_tests_to_relevant_source_files = [
    test
    for test in sorted_tests_to_relevant_source_files
    if tests_to_relevant_source_files[test]
]

# Add any tests that obey the "test_*" or "*_test" format found.

additional_possible_test_names = []
for full_source_file in source_files:
    file_name = os.path.basename(full_source_file).split(".")[0]

    common_test_names = [
        "test_" + file_name + ".py",
        "test_" + file_name + ".py",
        file_name + "_test.py",
        file_name + "test.py",
    ]

    for common_test_name in common_test_names:
        # Check if file exists in any form in "/testbed/"
        for test_file in test_files:
            if common_test_name in test_file:
                additional_possible_test_names.append(test_file)

sorted_tests_to_relevant_source_files = list(
    set([*sorted_tests_to_relevant_source_files, *additional_possible_test_names])
)

with open(args.outfile, "w") as f:
    f.write("\n".join(sorted_tests_to_relevant_source_files))
    if len(sorted_tests_to_relevant_source_files) > 0:
        f.write("\n")
