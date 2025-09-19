#!/usr/bin/env python
# ast_introspection_patchcheck_ref_py35.py
# Python 3.5+

from __future__ import print_function
import argparse, ast, os, sys, subprocess
from collections import namedtuple

# ---- policy --------------------------------------------------------------
FRAME_ATTRS = set(["f_back", "tb_frame", "gi_frame", "cr_frame", "ag_frame"])
BANNED_FUNCS = set([
    ("inspect","currentframe"), ("inspect","stack"),
    ("inspect", "getsource"), ("inspect","getsourcefile"),
    ("inspect","getouterframes"), ("inspect","getinnerframes"),
    ("inspect","trace"), ("inspect","getframeinfo"),
    ("traceback","extract_stack"), ("traceback","format_stack"),
    ("traceback","print_stack"), ("traceback","walk_stack"),
    ("sys","_getframe"), ("sys","settrace"), ("sys","setprofile"),
    ("gc","get_referrers"), ("gc","get_objects"),
])
DYN_IMPORT_MODULES = set(["inspect"])
FLAG_NATIVE_IMPORTS = True
SUPPRESS_PRAGMA = "stackguard: allow"
PY_EXTS = (".py",)

Finding = namedtuple("Finding", "path line col kind qual snippet")

# ---- unified diff parsing -------------------------------------------------
def _strip_ab_prefix(path):
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path

def parse_patch(patch_text):
    """
    Returns:
      added_lines: dict[path -> set(new_line_numbers)]
      new_files: set(paths that are brand-new in this patch)
      touched_files: set(all post-image paths we saw in +++ headers)
    """
    added_lines, new_files, touched = {}, set(), set()
    old_path, new_path, new_line = None, None, None

    for raw in patch_text.splitlines():
        if raw.startswith("--- "):
            old_path = raw[4:].strip()
        elif raw.startswith("+++ "):
            p = raw[4:].strip()
            new_path = None if p == "/dev/null" else _strip_ab_prefix(p)
            if new_path:
                touched.add(new_path)
                # brand-new if old side was /dev/null
                if old_path == "/dev/null":
                    new_files.add(new_path)
            new_line = None
        elif raw.startswith("@@ "):
            try:
                frag = raw.split(" +", 1)[1]
                n = ""
                for ch in frag:
                    if ch.isdigit(): n += ch
                    else: break
                new_line = int(n) - 1
            except Exception:
                new_line = None
        elif raw.startswith("+") and not raw.startswith("+++"):
            if new_path and os.path.splitext(new_path)[1] in PY_EXTS and new_line is not None:
                new_line += 1
                added_lines.setdefault(new_path, set()).add(new_line)
        elif raw.startswith(" ") or (raw and raw[0] not in "+-@"):
            if new_line is not None:
                new_line += 1
        # '-' lines advance only old side (we don't track)
    return added_lines, new_files, touched

# ---- AST scanning ---------------------------------------------------------
class ImportResolver(ast.NodeVisitor):
    def __init__(self):
        self.mod_alias, self.func_alias = {}, {}
    def visit_Import(self, node):
        for a in node.names:
            root = a.name.split(".")[0]
            self.mod_alias[(a.asname or root)] = root
    def visit_ImportFrom(self, node):
        if node.module is None: return
        root = node.module.split(".")[0]
        for a in node.names:
            self.func_alias[(a.asname or a.name)] = (root, a.name)

class IntrospectionScanner(ast.NodeVisitor):
    def __init__(self, src_lines, path):
        self.src, self.path, self.findings, self.imps = src_lines, path, [], ImportResolver()
        self.imported_modules = set()  # collected imports in this file (for reference graph)
    def visit_Import(self, node):
        self.imps.visit_Import(node)
        for a in node.names:
            full = a.name  # keep the full dotted name for reference graph
            self.imported_modules.add(full)
            root = a.name.split(".")[0]
            if root in ("inspect","traceback") or (FLAG_NATIVE_IMPORTS and root in ("ctypes","cffi")):
                self._add(node, "IMPORT", root, node.lineno, node.col_offset)
    def visit_ImportFrom(self, node):
        self.imps.visit_ImportFrom(node)
        if node.module:
            self.imported_modules.add(node.module)
            root = node.module.split(".")[0]
            if root in ("inspect","traceback") or (FLAG_NATIVE_IMPORTS and root in ("ctypes","cffi")):
                self._add(node, "IMPORT", root, node.lineno, node.col_offset)
    def visit_Call(self, node):
        # dynamic import
        if isinstance(node.func, ast.Name) and node.func.id == "__import__" and node.args:
            mod = self._const_str(node.args[0])
            if mod in DYN_IMPORT_MODULES:
                self._add(node,"DYNIMPORT",mod,node.lineno,node.col_offset)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == "importlib" and node.func.attr == "import_module" and node.args:
                mod = self._const_str(node.args[0])
                if mod in DYN_IMPORT_MODULES:
                    self._add(node,"DYNIMPORT",mod,node.lineno,node.col_offset)
        # normal calls
        qual = self._resolve_callee(node.func)
        if qual and qual in BANNED_FUNCS:
            self._add(node, "CALL", "%s.%s" % qual, node.lineno, node.col_offset)
        self.generic_visit(node)
    def visit_Attribute(self, node):
        if getattr(node, "attr", None) in FRAME_ATTRS:
            self._add(node, "ATTR", "."+node.attr, node.lineno, node.col_offset)
        self.generic_visit(node)
    # helpers
    def _resolve_callee(self, f):
        if isinstance(f, ast.Name):
            return self.imps.func_alias.get(f.id)
        if isinstance(f, ast.Attribute):
            left = f.value
            while isinstance(left, ast.Attribute): left = left.value
            if isinstance(left, ast.Name):
                base = left.id; mod = self.imps.mod_alias.get(base, base)
                return (mod, getattr(f, "attr", None))
        return None
    def _const_str(self, node):
        if hasattr(ast,"Constant") and isinstance(node, ast.Constant) and isinstance(node.value, str): return node.value
        if isinstance(node, ast.Str): return node.s
        return None
    def _suppressed(self, line_no):
        try: return SUPPRESS_PRAGMA in self.src[line_no-1]
        except Exception: return False
    def _add(self, node, kind, qual, lineno, col):
        if self._suppressed(lineno): return
        snippet = ""
        try: snippet = self.src[lineno-1].rstrip()
        except Exception: pass
        self.findings.append(Finding(self.path, lineno, col, kind, qual, snippet))

# ---- post-image source -----------------------------------------------------
def read_postimage(path, source):
    if source == "worktree":
        if not os.path.exists(path): return None
        try:
            with open(path, "r") as f: return f.read()
        except Exception: return None
    elif source == "index":
        try:
            return subprocess.check_output(["git", "show", ":" + path], universal_newlines=True)
        except subprocess.CalledProcessError:
            return None
    return None

# ---- module reference graph (within patch) ---------------------------------
def module_candidates_for_path(path):
    # path like foo/bar/baz.py -> ["baz", "bar.baz", "foo.bar.baz"]
    if not path.endswith(".py"): return []
    parts = path[:-3].replace("\\","/").split("/")
    cands, base = [], []
    for i in range(len(parts)-1, -1, -1):
        base.insert(0, parts[i])
        cands.append(".".join(base))
    return cands  # from shortest ("baz") to longest ("foo.bar.baz")

def collect_imports_for_files(paths, postimage_from):
    """
    Return dict[path -> set(imported module names found in that file)]
    """
    m = {}
    for rel in paths:
        src = read_postimage(rel, postimage_from)
        if not src: 
            m[rel] = set(); 
            continue
        try: tree = ast.parse(src, filename=rel)
        except SyntaxError:
            m[rel] = set(); 
            continue
        scanner = IntrospectionScanner(src.splitlines(), rel)
        # Only want imported_modules; avoid filling findings for this pass
        try: scanner.visit(tree)
        except Exception: pass
        m[rel] = set(scanner.imported_modules)
    return m

def determine_unreferenced_added(new_files, touched_files, postimage_from):
    """
    Mark added files that are NOT imported by any other touched file.
    """
    if not new_files: return set()
    imports_map = collect_imports_for_files(touched_files, postimage_from)
    # Build set of all imports from files OTHER than the candidate file
    unref = set()
    for nf in new_files:
        cands = set(module_candidates_for_path(nf))
        referenced = False
        for other in touched_files:
            if other == nf: 
                continue
            imps = imports_map.get(other, set())
            # Check exact or 'starts with' match: import foo.bar.baz or from foo.bar import baz
            for imp in imps:
                # match "foo.bar.baz" against any candidate
                if (imp in cands) or any(imp.endswith("." + c) for c in cands):
                    referenced = True
                    break
            if referenced:
                break
        if not referenced:
            unref.add(nf)
    return unref

# ---- main scan on added lines ----------------------------------------------
def scan_added_only(added_map, postimage_from):
    results = []
    perfile_imports = {}
    for rel in sorted(added_map.keys()):
        if os.path.splitext(rel)[1] not in PY_EXTS:
            continue
        src = read_postimage(rel, postimage_from)
        if not src: 
            continue
        try:
            tree = ast.parse(src, filename=rel)
        except SyntaxError:
            continue
        scanner = IntrospectionScanner(src.splitlines(), rel.replace("\\","/"))
        try: scanner.visit(tree)
        except Exception: pass
        added_lines = added_map.get(rel, set())
        results.extend([f for f in scanner.findings if f.line in added_lines])
    return results

# ---- CLI -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Report stack-introspection on ADDED lines from a patch (Py3.5+).")
    ap.add_argument("--patch-file", required=True, help="Unified diff; use '-' for stdin.")
    ap.add_argument("--postimage-from", choices=["worktree","index"], default="worktree",
                    help="Where to read the post-image (default: worktree).")
    ap.add_argument("--include-standalone-added", action="store_true",
                    help="Also flag brand-new files even if nothing else in the patch imports them.")
    ap.add_argument("--no-native-imports", action="store_true",
                    help="Do not flag 'import ctypes/cffi'.")
    args = ap.parse_args()

    if args.no_native_imports:
        global FLAG_NATIVE_IMPORTS
        FLAG_NATIVE_IMPORTS = False

    patch_text = sys.stdin.read() if args.patch_file == "-" else open(args.patch_file, "r").read()

    added_map, new_files, touched_files = parse_patch(patch_text)
    if not added_map:
        print("No Python files with added lines found in patch.")
        return 0

    # Determine brand-new files that are NOT imported by any other touched file
    unref_added = set()
    if not args.include_standalone_added:
        unref_added = determine_unreferenced_added(new_files, touched_files, args.postimage_from)

    # Run AST scan, then filter out findings that live in unreferenced new files
    findings = scan_added_only(added_map, args.postimage_from)
    filtered = [f for f in findings if f.path not in unref_added]

    if not filtered:
        print("No NEW stack-introspection usage detected on added lines.")
        if unref_added:
            print("Note: skipped {} standalone added file(s): {}".format(
                len(unref_added), ", ".join(sorted(unref_added))))
        return 0

    print("NEW stack-introspection usage detected on added lines:\n")
    for f in filtered:
        print("{path}:{line}:{col}: [{kind} {qual}]".format(
            path=f.path, line=f.line, col=f.col, kind=f.kind, qual=f.qual))
        if f.snippet:
            print("    {0}".format(f.snippet))
    if unref_added:
        print("\nNote: skipped {} standalone added file(s): {}".format(
            len(unref_added), ", ".join(sorted(unref_added))))
    print("\n(add '# stackguard: allow' at end of the ADDED line to suppress intentionally)")
    return 2

if __name__ == "__main__":
    sys.exit(main())
