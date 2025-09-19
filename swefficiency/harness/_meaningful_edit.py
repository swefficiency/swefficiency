
import sys
import argparse

from tree_sitter import Language, Parser
from tree_sitter_languages import get_language, get_parser

# PY_LANGUAGE = Language(tspython.language())
PY_LANGUAGE = get_language('python')
# parser = Parser(PY_LANGUAGE)
parser = get_parser('python')

# Tresitter parses triple ticks as strings and not comments. Generally, string changes are not perf improving.
NODE_IGNORE_FN = lambda n: n.type == "comment" or n.type == "string"

# Compare two parsed trees with an node ignore function.
def compare_trees(tree1, tree2, node_ignore_fn=None) -> bool:

  # Compare the root nodes.
  if tree1.type != tree2.type:
    return False

  # Filter out ignored children.
  tree1_children = [
      n
      for n in tree1.children
      if node_ignore_fn is None or not node_ignore_fn(n)
  ]
  tree2_children = [
      n
      for n in tree2.children
      if node_ignore_fn is None or not node_ignore_fn(n)
  ]

  if len(tree1_children) != len(tree2_children):
    return False

  # Iterate through the children and compare the trees recursively.
  for i, (child1, child2) in enumerate(zip(tree1_children, tree2_children)):
    if not compare_trees(child1, child2, node_ignore_fn=node_ignore_fn):
      return False

  # Otherwise, the trees are the same.
  return True

argparser = argparse.ArgumentParser()
argparser.add_argument('--preedit_file', type=str, required=True, help="Pre-edit file.")
argparser.add_argument('--postedit_file', type=str, required=True, help="Post-edit file.")

args = argparser.parse_args()

# Open the files as text, decode into UTF-8 bytes
with open(args.preedit_file, "r", encoding="utf-8") as f1:
    preedit_text = f1.read()

with open(args.postedit_file, "r", encoding="utf-8") as f2:
    postedit_text = f2.read()

# Convert the text into UTF-8 bytes
preedit_code_bytes = preedit_text.encode("utf-8")
postedit_code_bytes = postedit_text.encode("utf-8")

# Parse the UTF-8 byte sequences
preedit_tree = parser.parse(preedit_code_bytes)
postedit_tree = parser.parse(postedit_code_bytes)

trees_equal = compare_trees(
    preedit_tree.root_node,
    postedit_tree.root_node,
    NODE_IGNORE_FN
)

if trees_equal:
    print("NOOP")
else:
    print("MEANINGFUL")
