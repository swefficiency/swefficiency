# rewrite_benchmark.py

import ast
import textwrap


# ---------- tiny codegen helper (py3.6+ support) ----------
def _to_source(node):
    # Prefer stdlib ast.unparse (3.9+), else astor if available.
    try:
        return ast.unparse(node)
    except AttributeError:
        try:
            import astor  # pip install astor
        except Exception as e:
            raise RuntimeError(
                "This script needs Python 3.9+ (ast.unparse) or 'astor' installed for older Pythons."
            ) from e
        return astor.to_source(node).rstrip()


# ---------- Static detector (optional but useful) ----------
SUSPECT_NAMES = ("cache", "memo", "lru", "lookup", "store")


def warn_suspect_caches(src: str):
    """
    Heuristic static checks for module-level caches.
    - module-level mutable containers (dict/list/set) with cache-y names
    - assignments like NAME[...] = ... at module scope
    - @functools.lru_cache decorators
    """
    tree = ast.parse(src)
    warnings = []

    class LruFinder(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            for dec in node.decorator_list:
                # @lru_cache or @functools.lru_cache(...)
                if isinstance(dec, ast.Name) and dec.id == "lru_cache":
                    warnings.append(f"LRU cache decorator on function '{node.name}'.")
                elif isinstance(dec, ast.Attribute) and dec.attr == "lru_cache":
                    warnings.append(f"LRU cache decorator on function '{node.name}'.")
                elif isinstance(dec, ast.Call):
                    base = dec.func
                    if (isinstance(base, ast.Name) and base.id == "lru_cache") or (
                        isinstance(base, ast.Attribute) and base.attr == "lru_cache"
                    ):
                        warnings.append(
                            f"LRU cache decorator on function '{node.name}'."
                        )
            self.generic_visit(node)

    LruFinder().visit(tree)

    # Detect module-level suspicious containers and writes
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = []
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        targets.append(t.id)
                value = node.value
            else:
                if isinstance(node.target, ast.Name):
                    targets.append(node.target.id)
                value = node.value

            def is_mut_container(v):
                return isinstance(v, (ast.Dict, ast.List, ast.Set)) or (
                    isinstance(v, ast.Call)
                    and isinstance(v.func, ast.Name)
                    and v.func.id in {"dict", "list", "set"}
                )

            for tname in targets:
                if any(k in tname.lower() for k in SUSPECT_NAMES) and is_mut_container(
                    value
                ):
                    warnings.append(
                        f"Module-level mutable '{tname}' looks like a cache."
                    )

        # Detect module-level subscript writes like _cache[n] = val
        if isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Subscript) and isinstance(
                node.target.value, ast.Name
            ):
                nm = node.target.value.id
                if any(k in nm.lower() for k in SUSPECT_NAMES):
                    warnings.append(f"Module-level subscript write into '{nm}'.")

    return sorted(set(warnings))


# ---------- AST extractor for timeit parameters ----------
class RepeatCall:
    def __init__(self, workload_name, setup_name, number, repeat):
        self.workload_name = workload_name
        self.setup_name = setup_name
        self.number = number
        self.repeat = repeat


def _is_timeit_repeat_call(call: ast.Call) -> bool:
    # Match either timeit.repeat(...) or repeat(...) when imported directly
    if isinstance(call.func, ast.Attribute):
        return (
            isinstance(call.func.value, ast.Name)
            and call.func.value.id == "timeit"
            and call.func.attr == "repeat"
        )
    if isinstance(call.func, ast.Name) and call.func.id == "repeat":
        return True
    return False


def _const_int(node, default=None):
    try:
        return int(ast.literal_eval(node))
    except Exception:
        return default


def extract_timeit_repeat(tree: ast.AST):
    """
    Look for: runtimes = timeit.repeat(workload, number=..., repeat=..., setup=setup)
    Returns RepeatCall or None.
    """
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and _is_timeit_repeat_call(node.value)
        ):
            call = node.value
            workload_name = None
            if call.args and isinstance(call.args[0], ast.Name):
                workload_name = call.args[0].id

            number = 1
            repeat = 5
            setup_name = None
            for kw in call.keywords:
                if kw.arg == "number":
                    number = _const_int(kw.value, 1)
                elif kw.arg == "repeat":
                    repeat = _const_int(kw.value, 5)
                elif kw.arg == "setup":
                    if isinstance(kw.value, ast.Name):
                        setup_name = kw.value.id

            if workload_name:
                return RepeatCall(workload_name, setup_name, number, repeat)
    return None


def has_func(tree: ast.AST, name: str) -> bool:
    return any(isinstance(n, ast.FunctionDef) and n.name == name for n in tree.body)


# ---------- NEW: detect slicing like runtimes[-10000:] ----------
def _slice_to_src(slc: ast.slice) -> str:
    """Return the textual slice part (no surrounding brackets), e.g. '-10000:' or 'a:b:c'."""
    # Python 3.8: slice is ast.Slice via ast.Index; 3.9+: directly ast.Slice
    if isinstance(slc, ast.Slice):
        parts = []
        parts.append(_to_source(slc.lower) if slc.lower is not None else "")
        parts.append(_to_source(slc.upper) if slc.upper is not None else "")
        tail = ""
        if slc.step is not None:
            tail = ":" + (_to_source(slc.step) if slc.step is not None else "")
        return f"{parts[0]}:{parts[1]}{tail}"
    # Not a slice (e.g., single index); ignore
    return ""


def find_runtimes_slice_expr(tree: ast.AST) -> str | None:
    """
    Walks the tree to find 'runtimes[<slice>]' uses.
    Returns the *last* slice expression text found (without surrounding brackets), or None.
    """
    found = []

    class SubFinder(ast.NodeVisitor):
        def visit_Subscript(self, node):
            # py3.9+: value is in node.value, slice in node.slice
            # py3.8: node.slice may be ast.Index(value=...)
            val = node.value
            if isinstance(val, ast.Name) and val.id == "runtimes":
                sl = node.slice
                # py3.8 compat
                if hasattr(ast, "Index") and isinstance(sl, ast.Index):
                    sl = sl.value
                if isinstance(sl, ast.Slice):
                    txt = _slice_to_src(sl)
                    if txt != "":
                        found.append(txt)
            self.generic_visit(node)

    SubFinder().visit(tree)
    return found[-1] if found else None


# ---------- Rewriter ----------
def transform_to_isolated_workload(src: str) -> str:
    tree = ast.parse(src)
    rep = extract_timeit_repeat(tree)
    if rep is None:
        raise SystemExit(
            "Could not find 'timeit.repeat(...)' assignment to 'runtimes' in the script."
        )

    workload = rep.workload_name
    setup = (
        rep.setup_name if rep.setup_name and has_func(tree, rep.setup_name) else None
    )
    number = rep.number if rep.number is not None else 1
    repeat = rep.repeat if rep.repeat is not None else 5

    # Detect if the original script slices the runtimes list
    slice_expr = find_runtimes_slice_expr(tree)  # e.g., '-10000:' or 'a:b' or 'a:b:c'

    # Keep original top-level code except the timeit.repeat assignment and the final prints
    kept_nodes = []
    for n in tree.body:
        skip = False
        if (
            isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and _is_timeit_repeat_call(n.value)
        ):
            skip = True
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
            cal = n.value
            if isinstance(cal.func, ast.Name) and cal.func.id == "print":
                for a in cal.args:
                    # Drop prints like: print("Mean:", ...), print("Std Dev:", ...)
                    if (
                        isinstance(a, ast.Constant)
                        and isinstance(a.value, str)
                        and ("Mean:" in a.value or "Std Dev:" in a.value)
                    ):
                        skip = True
                        break
        if not skip:
            kept_nodes.append(n)

    original_body = (
        "".join(
            _to_source(n) + ("\n" if not _to_source(n).endswith("\n") else "")
            for n in kept_nodes
        )
        + "\n"
    )

    setup_expr = setup if setup else "(lambda: None)"

    # Build a view expression to apply the same slice to our new runtimes
    # None -> full slice; else -> f"[{slice_expr}]"
    view_assignment = "runtimes_view = runtimes\n"
    if slice_expr:
        view_assignment = f"runtimes_view = runtimes[{slice_expr}]\n"

    # Isolation harness
    harness = f"""
# ---- AUTO-GENERATED ISOLATION HARNESS (timeit-in-child, spawn-safe) ----
import statistics as _statistics
import multiprocessing as _mp
import timeit as _timeit

def _child_once(number: int):
    setup_fn = {setup_expr}
    t = _timeit.Timer({workload}, setup=setup_fn)
    return t.timeit(number)

# TOP-LEVEL target: picklable under 'spawn'
def _child_target(q, number):
    try:
        dur = _child_once(number)
        q.put(dur)
    except BaseException:
        import traceback
        traceback.print_exc()
        q.put(None)

def _run_isolated(number: int, repeat: int, start_method: str = "spawn"):
    ctx = _mp.get_context(start_method)
    results = []
    for _ in range(repeat):
        q = ctx.Queue()
        p = ctx.Process(target=_child_target, args=(q, number))
        p.start()
        dur = q.get()
        p.join()
        if dur is None or p.exitcode != 0:
            raise RuntimeError("Child run failed—see traceback above.")
        results.append(dur)
    return results

if __name__ == "__main__":
    _number = {number}
    _repeat = {repeat}
    runtimes = _run_isolated(_number, _repeat, start_method="spawn")
    {view_assignment.rstrip()}
    print("Mean:", _statistics.mean(runtimes_view))
    print("Std Dev:", _statistics.stdev(runtimes_view) if len(runtimes_view) > 1 else 0.0)
"""

    return original_body + textwrap.dedent(harness)


# --------------- quick local demo ---------------
if __name__ == "__main__":
    TEST_WORKLOAD = """
import timeit
import statistics
import astropy.units as u
from astropy.units import Quantity
import numpy as np
from astropy.coordinates.angles import Longitude, Latitude, Angle

def setup():
    global values1, values2
    values1 = np.random.uniform(-180, 180, 100)
    values2 = np.random.uniform(0, 359, 100)

def workload():
    global values1, values2
    Longitude(values1, u.deg)
    Longitude(values2, u.deg)

runtimes = timeit.repeat(workload, number=10, repeat=20000, setup=setup)

print("Mean:", statistics.mean(runtimes[-10000:]))
print("Std Dev:", statistics.stdev(runtimes[-10000:]))
"""

    print("--- Original Script ---")
    print(TEST_WORKLOAD)
    print("--- Modified Script ---")
    print(transform_to_isolated_workload(TEST_WORKLOAD))
