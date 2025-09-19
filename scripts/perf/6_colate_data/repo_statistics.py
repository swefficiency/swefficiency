# %%

# Create dataframe with details and export to sheets.

from pathlib import Path

data_dir = Path("data")

# Get list of subdirs that have "workload.py"
subdirs = [x for x in data_dir.iterdir() if x.is_dir() and (x / "workload.py").exists()]
instance_ids = [x.name for x in subdirs]


def get_instance_info(instance_id):
    namespace, remainder = instance_id.split("__")
    repo, pull_number = remainder.rsplit("-", 1)
    return namespace, repo, int(pull_number)


eval_results_dir = Path("logs/run_evaluation")


def get_perf_info(instance_id):
    namespace, repo, pull_number = get_instance_info(instance_id)

    perf_result_dir = eval_results_dir / f"perf_coverage_{repo}" / "gold" / instance_id

    debugging_result_dir = (
        eval_results_dir / f"debugging_coverage_{repo}" / "gold" / instance_id
    )

    covering_tests = data_dir / instance_id / "covering_tests.txt"
    covering_tests = covering_tests.read_text() if covering_tests.exists() else ""

    if (debugging_result_dir / "patch.diff").exists():
        patch = (debugging_result_dir / "patch.diff").read_text()
    elif (perf_result_dir / "patch.diff").exists():
        patch = (perf_result_dir / "patch.diff").read_text()
    else:
        patch = ""

    perf_workload = perf_result_dir / "workload.py"
    perf_summary = perf_result_dir / "perf_summary.txt"

    before_mean, before_std_dev, after_mean, after_std_dev, improvement = (
        None,
        None,
        None,
        None,
        None,
    )

    if perf_summary.exists():
        with open(perf_summary) as f:
            perf_summary_text_lines = f.read().strip().splitlines()

        before_mean, before_std_dev, after_mean, after_std_dev, improvement = [
            x.split()[-1] for x in perf_summary_text_lines
        ]
        improvement = improvement.replace("%", "")

        # Convert all to float.
        before_mean = float(before_mean)
        before_std_dev = float(before_std_dev)
        after_mean = float(after_mean)
        after_std_dev = float(after_std_dev)
        improvement = float(improvement)

    return {
        "before_mean": before_mean,
        "before_std_dev": before_std_dev,
        "after_mean": after_mean,
        "after_std_dev": after_std_dev,
        "improvement": improvement,
        "perf_workload": perf_workload.read_text() if perf_workload.exists() else None,
        "patch": patch,
        "len_patch": len(patch.splitlines()),
        "covering_tests": covering_tests,
        "len_covering_tests": len(covering_tests.splitlines()) if covering_tests else 0,
    }


def get_link(instance_id):
    namespace, repo, pull_number = get_instance_info(instance_id)

    return f"https://github.com/{namespace}/{repo}/pull/{pull_number}"


perf_info = [get_perf_info(instance_id) for instance_id in instance_ids]


elements = []

for instance_id in instance_ids:
    namespace, repo, pull_number = get_instance_info(instance_id)
    link = get_link(instance_id)
    perf_info = get_perf_info(instance_id)

    elements.append(
        {
            "instance_id": instance_id,
            "link": link,
            "namespace": namespace,
            "repo": repo,
            "pull_number": pull_number,
            **perf_info,
        }
    )

# Turn into a pandas dataframe.
import pandas as pd

df = pd.DataFrame(elements)

# Export to CSV
output_dir = Path("harness_data")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "analyzed_data.xlsx"

df.to_excel(output_file)


# %%
