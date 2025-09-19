import os
from datasets import load_dataset
from swefficiency.harness.run_validation import parse_perf_summary
import tqdm

SYNTHETIC_WORKLOADS_DIR = "logs/run_evaluation/synthetic_workloads/gold"
REAL_WORKLOADS_DIR = "logs/run_evaluation/ground_truth5/gold"

ds = load_dataset("swefficiency/swefficiency", split="test")


def worker(d):
    instance_id = d["instance_id"]
    synthetic_path = f"{SYNTHETIC_WORKLOADS_DIR}/{instance_id}/perf_summary.txt"
    real_path = f"{REAL_WORKLOADS_DIR}/{instance_id}/perf_summary.txt"

    if not (os.path.exists(synthetic_path) and os.path.exists(real_path)):
        # print(f"Skipping {instance_id} as one of the paths does not exist.")
        return False, False

    synthetic_perf = open(synthetic_path).read()
    real_perf = open(real_path).read()

    synthetic_summary = parse_perf_summary(synthetic_perf)
    real_summary = parse_perf_summary(real_perf)

    synthetic_better = synthetic_summary["improvement"] < real_summary["improvement"]

    # statistically significant if
    synthetic_improvement_mean = (
        synthetic_summary["after_mean"] - synthetic_summary["before_mean"]
    )
    synthetic_se = synthetic_summary["after_std"]

    return synthetic_better, synthetic_improvement_mean < (-2 * synthetic_se)


# Parallelize over multiprocess.
import multiprocessing

with multiprocessing.Pool(processes=4) as pool:
    results = list(
        tqdm.tqdm(
            pool.imap(worker, ds),
            total=len(ds),
            desc="Comparing synthetic vs real workloads",
        )
    )

instances_where_synthetic_better = [r[0] for r in results if r[0]]
instances_where_synthetic_is_statistically_signficant = [r[1] for r in results if r[1]]

print("count where synthetic better:", len(instances_where_synthetic_better))
print(
    "count where synthetic is statistically significant:",
    len(instances_where_synthetic_is_statistically_signficant),
)
