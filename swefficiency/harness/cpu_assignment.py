import os
from collections import defaultdict, deque
from typing import Dict, List, Tuple

SYS_CPU = "/sys/devices/system/cpu"


def _parse_cpu_list(s: str) -> List[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = map(int, part.split("-"))
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return out


def _cpu_to_node(cpu: int) -> int:
    d = f"{SYS_CPU}/cpu{cpu}"
    try:
        entries = [e for e in os.listdir(d) if e.startswith("node")]
        return int(entries[0].replace("node", "")) if entries else 0
    except FileNotFoundError:
        return 0


def discover_physical_cores() -> List[Tuple[List[int], int]]:
    """
    Returns a list of (siblings, numa_node) for each physical core.
    `siblings` is a sorted list of all logical CPUs that share that core (SMT threads).
    """
    seen = set()
    cores = []
    for entry in os.listdir(SYS_CPU):
        if not entry.startswith("cpu") or not entry[3:].isdigit():
            continue
        cpu = int(entry[3:])
        if cpu in seen:
            continue
        topo = f"{SYS_CPU}/cpu{cpu}/topology/thread_siblings_list"
        try:
            with open(topo) as f:
                sibs = sorted(set(_parse_cpu_list(f.read().strip())))
        except FileNotFoundError:
            sibs = [cpu]
        for c in sibs:
            seen.add(c)
        node = _cpu_to_node(sibs[0])
        cores.append((sibs, node))
    # stable ordering by (node, first-sib)
    cores.sort(key=lambda t: (t[1], t[0][0]))
    return cores


def allocate_whole_cores(
    num_workers: int,
    vcpus_per_worker: int = 4,
    threads_per_core: int = 2,
    reserve_cores: int = 0,  # optionally keep some cores unassigned (for OS/IRQs)
):
    """
    Allocate to each worker a set of vCPUs built from WHOLE physical cores.
    - No physical core is shared across workers.
    - If threads_per_core == 2, each assigned core contributes both SMT threads (e.g., 0,32).
      So vcpus_per_worker must be divisible by threads_per_core.
    - Returns: [{worker, cpuset_cpus, cpuset_mems, nano_cpus}]
    """
    if num_workers <= 0 or vcpus_per_worker <= 0:
        raise ValueError("num_workers and vcpus_per_worker must be > 0")
    if threads_per_core not in (1, 2):
        raise ValueError("threads_per_core must be 1 or 2")
    if vcpus_per_worker % threads_per_core != 0:
        raise ValueError("vcpus_per_worker must be divisible by threads_per_core")

    cores_needed_per_worker = vcpus_per_worker // threads_per_core

    cores = discover_physical_cores()
    if reserve_cores > 0:
        # drop the first N cores (ordered by (node, cpu)) to keep for the host
        cores = cores[reserve_cores:]

    total_cores = len(cores)
    need_cores = num_workers * cores_needed_per_worker
    if total_cores < need_cores:
        raise RuntimeError(
            f"Not enough physical cores: need {need_cores}, have {total_cores}"
        )

    # Bucket cores by NUMA node
    by_node: Dict[int, deque] = defaultdict(deque)
    for sibs, node in cores:
        by_node[node].append((sibs, node))

    plans = [{"worker": i, "cores": [], "nodes": set()} for i in range(num_workers)]

    # Phase 1: pack whole workers inside a single node when possible
    w = 0
    for node in sorted(by_node.keys()):
        q = by_node[node]
        while (
            len(q) >= cores_needed_per_worker
            and w < num_workers
            and len(plans[w]["cores"]) == 0
        ):
            take = [q.popleft()[0] for _ in range(cores_needed_per_worker)]
            plans[w]["cores"].extend(take)
            plans[w]["nodes"].add(node)
            w += 1
            if w >= num_workers:
                break

    # Phase 2: fill remaining workers round-robin from whatever cores remain (may span nodes)
    remaining = []
    for node in sorted(by_node.keys()):
        remaining.extend(list(by_node[node]))
    pending = [
        i
        for i in range(num_workers)
        if len(plans[i]["cores"]) < cores_needed_per_worker
    ]
    p = 0
    for sibs, node in remaining:
        if not pending:
            break
        i = pending[p]
        plans[i]["cores"].append(sibs)
        plans[i]["nodes"].add(node)
        if len(plans[i]["cores"]) == cores_needed_per_worker:
            pending.pop(p)
            if not pending:
                break
            p %= len(pending)
        else:
            p = (p + 1) % len(pending)

    # Finalize: expand each core into the desired number of threads (1 or 2)
    out = []
    for r in plans:
        cpus = []
        for core_sibs in r["cores"]:
            chosen = core_sibs[:threads_per_core]  # pick 1 or both siblings
            cpus.extend(chosen)
        cpus = sorted(cpus)
        mems = ",".join(str(n) for n in sorted(r["nodes"])) or "0"
        out.append(
            {
                "cpuset_cpus": ",".join(map(str, cpus)),
                "cpuset_mems": mems,
                "nano_cpus": int(1e9 * len(cpus)),  # optional: hard cap to vCPU count
            }
        )
    return out


def divide_cpus_among_workers(num_workers, cpus_per_worker=4):
    cpu_groups = []

    for i in range(num_workers):
        cpu_groups.append(list(range(i * cpus_per_worker, (i + 1) * cpus_per_worker)))

    return cpu_groups


if __name__ == "__main__":
    # Example usage
    num_workers = 12
    vcpus_per_worker = 4
    threads_per_core = 2
    reserve_cores = 4

    cpu_groups = allocate_whole_cores(
        num_workers, vcpus_per_worker, threads_per_core, reserve_cores
    )

    print(len(cpu_groups), "CPU groups allocated:")

    for group in cpu_groups:
        print(group)
