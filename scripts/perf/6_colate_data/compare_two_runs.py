# %%

# Compare perf results.

from pathlib import Path

new_perf_run_dir = Path("logs/run_evaluation/perf_coverage_openhands/openhands")
base_perf_run_dir = Path("logs/run_evaluation/perf_coverage_openhands/gold")

instance_ids = [d.name for d in new_perf_run_dir.iterdir() if d.is_dir()]

for instance_id in instance_ids:
    new_perf_summary = new_perf_run_dir / instance_id / "perf_summary.txt"
    base_perf_summary = base_perf_run_dir / instance_id / "perf_summary.txt"

    if not new_perf_summary.exists() or not base_perf_summary.exists():
        continue

    new_perf_summary_text = new_perf_summary.read_text()
    base_perf_summary_text = base_perf_summary.read_text()

    new_improvement = new_perf_summary_text.split("Improvement: ")[-1]
    base_improvement = base_perf_summary_text.split("Improvement: ")[-1]

    print(
        f"ID: {instance_id}: OpenHands w/ Gemini 2.5 Pro {new_improvement} vs. Base {base_improvement}"
    )


# %%

# Compare correctness results.

from pathlib import Path
import json

base_correctness_run_dir = Path("logs/run_evaluation/perf_coverage_openhands")
empty_dir = base_correctness_run_dir / "empty"
openhands_dir = base_correctness_run_dir / "openhands"

instance_ids = [d.name for d in openhands_dir.iterdir() if d.is_dir()]

for instance_id in instance_ids:
    empty_subtest_status = empty_dir / instance_id / "subtest_status.json"
    openhands_subtest_status = openhands_dir / instance_id / "subtest_status.json"

    empty_subtest_status = json.loads(empty_subtest_status.read_text())
    openhands_subtest_status = json.loads(openhands_subtest_status.read_text())

    # Assert equal key by key.
    assert set(empty_subtest_status.keys()) == set(openhands_subtest_status.keys())

    print(instance_id)

    for key in empty_subtest_status:
        for subtest in empty_subtest_status[key]:
            try:
                if (
                    empty_subtest_status[key][subtest]
                    != openhands_subtest_status[key][subtest]
                ):
                    raise ValueError()
            except Exception as e:

                print(key)
                print(subtest)
