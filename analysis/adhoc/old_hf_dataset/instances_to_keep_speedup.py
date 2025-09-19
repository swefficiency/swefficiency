import datasets
from pathlib import Path
import tqdm

dataset = datasets.load_dataset("swefficiency/swefficiency", split="test")

eval_dir = Path("logs/run_evaluation/ground_truth2/gold")


def parse_perf_summary(perf_summary):
    perf_lines = perf_summary.strip().splitlines()

    before_mean = float(perf_lines[0].split(":")[1].strip())
    before_std = float(perf_lines[1].split(":")[1].strip())
    after_mean = float(perf_lines[2].split(":")[1].strip())
    after_std = float(perf_lines[3].split(":")[1].strip())
    improvement = (after_mean - before_mean) / before_mean * 100

    return {
        "before_mean": before_mean,
        "after_mean": after_mean,
        "before_std": before_std,
        "after_std": after_std,
        "improvement": improvement,
    }


bad_instances = set()

for d in tqdm.tqdm(dataset):
    instance_id = d["instance_id"]
    log_path = eval_dir / d["instance_id"] / "perf_summary.txt"

    if log_path.exists():
        with open(log_path) as f:
            perf_summary = f.read()
            perf_metrics = parse_perf_summary(perf_summary)

            # if perf_metrics["improvement"] > -20.0:
            # Check if "after_mean" is more than 1 "after_std" less than "before_mean"
            if (
                perf_metrics["after_mean"]
                > perf_metrics["before_mean"] - 2 * perf_metrics["after_std"]
            ):
                bad_instances.add(instance_id)
                print(
                    f"Instance {instance_id} has insufficient improvement: {perf_metrics}"
                )

            # else:
            #     continue

print(f"Total bad instances: {len(bad_instances)}")
for instance in bad_instances:
    print('"' + instance + '",')
