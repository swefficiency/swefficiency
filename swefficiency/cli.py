"""
Command line interface for swefficiency package.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List

from swefficiency.harness.constants import SWEfficiencyInstance
from swefficiency.harness.run_validation import main as run_validation_main
from swefficiency.harness.utils import load_swefficiency_dataset


def filter_instances_by_regex(
    instances: List[SWEfficiencyInstance], regex_pattern: str
) -> List[str]:
    """
    Filter instances by regex pattern applied to instance IDs.

    Args:
        instances: List of dataset instances
        regex_pattern: Regular expression pattern to match instance IDs

    Returns:
        List of instance IDs that match the pattern
    """
    pattern = re.compile(regex_pattern)
    return [
        instance["instance_id"]
        for instance in instances
        if pattern.search(instance["instance_id"])
    ]


def eval_command(args):
    """
    Handle the eval subcommand.
    """
    # Set defaults
    dataset_name = "swefficiency/swefficiency"
    split = "test"

    # Handle prediction path - default to None if not provided
    predictions_path = args.prediction_path if args.prediction_path else None

    # Handle instance filtering by regex
    instance_ids = None
    if args.instances_regex:
        # Load dataset to get all instances
        dataset = load_swefficiency_dataset(dataset_name, split)
        instance_ids = filter_instances_by_regex(dataset, args.instances_regex)
        if not instance_ids:
            print(
                f"Warning: No instances matched regex pattern '{args.instances_regex}'"
            )
            return
        print(
            f"Found {len(instance_ids)} instances matching regex pattern '{args.instances_regex}'"
        )

    # Generate a run ID if not provided
    run_id = args.run_id if args.run_id else f"cli_run_{int(time.time())}"

    print(f"Starting evaluation with {args.num_workers} workers...")
    if predictions_path:
        print(f"Predictions path: {predictions_path}")
    else:
        print("Using gold predictions (no prediction path provided)")
    if instance_ids:
        print(f"Running on {len(instance_ids)} filtered instances")
    print(f"Run ID: {run_id}")

    # Run the evaluation using run_validation.py's main function
    try:
        run_validation_main(
            dataset_name=dataset_name,
            split=split,
            instance_ids=instance_ids or [],
            max_workers=args.num_workers,
            max_build_workers=16,
            force_rebuild=False,
            cache_level="env",
            clean=False,
            open_file_limit=4096,
            run_id=run_id,
            timeout=7200,
            allow_test_patch=False,
            run_coverage=False,
            run_perf=True,
            run_perf_profiling=False,
            run_correctness=True,
            empty_patch=False,
            model_predictions=predictions_path or "",
            gdrive_annotation_sheet="",
            push_to_dockerhub=False,
            use_dockerhub_images=True,
            use_podman=False,
            workload_predictions="",
            force_rerun=True,
        )
        print("Evaluation completed successfully!")
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)


def report_command(args):
    """
    Handle the report subcommand.
    """
    # Set up paths
    eval_dir = Path("logs/run_evaluation")
    gold_run_name = eval_dir / args.run_id / "gold"
    pred_run_name = eval_dir / args.run_id / args.run_id

    # Check if we have the gold run
    if not gold_run_name.exists():
        print(f"Error: Gold run directory not found at {gold_run_name}")
        print("Make sure you have run evaluation with gold predictions first.")
        sys.exit(1)

    # Check if we have the prediction run
    if not pred_run_name.exists():
        print(f"Error: Prediction run directory not found at {pred_run_name}")
        print("Make sure you have run evaluation with your predictions first.")
        sys.exit(1)

    output_dir = (
        Path(args.report_output) if args.report_output else Path("eval_reports")
    )

    print("Generating report comparing gold run and prediction run...")
    print(f"Gold run: {gold_run_name}")
    print(f"Prediction run: {pred_run_name}")
    print(f"Output directory: {output_dir}")

    # Run get_report.py equivalent functionality
    try:
        # Import the script functionality
        script_path = (
            Path(__file__).parent.parent / "scripts" / "eval" / "get_report.py"
        )

        # Run the report generation script
        cmd = [
            sys.executable,
            str(script_path),
            "--gold_run",
            str(gold_run_name),
            "--pred_run",
            str(pred_run_name),
            "--num_workers",
            str(4),
            "--output_dir",
            str(output_dir),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            print("Report generated successfully!")
            print(result.stdout)
        else:
            print(f"Error generating report: {result.stderr}")
            sys.exit(1)

    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Error during report generation: {e}")
        sys.exit(1)


def main():
    """
    Main CLI entry point.
    """
    parser = argparse.ArgumentParser(
        prog="swefficiency",
        description="SWE-fficiency: A benchmark for evaluating LMs on software engineering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Eval subcommand
    eval_parser = subparsers.add_parser(
        "eval",
        help="Run evaluation on predictions",
        description="Evaluate model predictions on the SWE-fficiency benchmark",
    )
    eval_parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers to use (default: 4)",
    )
    eval_parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Run ID to identify this evaluation run",
    )
    eval_parser.add_argument(
        "--prediction_path",
        type=str,
        help="Path to predictions file (.json or .jsonl). If not provided, uses gold predictions for testing.",
    )
    eval_parser.add_argument(
        "--instances_regex",
        type=str,
        help='Regular expression pattern to filter instance IDs (e.g., "pandas.*" for Pandas instances)',
    )

    # Report subcommand
    report_parser = subparsers.add_parser(
        "report",
        help="Generate evaluation report",
        description="Generate evaluation report comparing prediction runs",
    )
    report_parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Run ID for which to generate the report",
    )
    report_parser.add_argument(
        "--report_output",
        type=str,
        default="eval_reports",
        help="Output directory for the report (default: eval_reports)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Handle subcommands
    if args.command == "eval":
        eval_command(args)
    elif args.command == "report":
        report_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
