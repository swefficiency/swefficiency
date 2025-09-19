from __future__ import annotations

import importlib.resources as ir
import json
import os
import random
import resource
import shlex
import shutil
import subprocess
import threading
import time
import traceback
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import docker
import pandas as pd
from tqdm import tqdm

import swefficiency
import swefficiency.harness
from swefficiency.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    INSTANCE_IMAGE_BUILD_DIR,
    KEY_INSTANCE_ID,
    RUN_EVALUATION_LOG_DIR,
    STACKFRAME_CHECK_EXCEPTONS,
    TestStatus,
)
from swefficiency.harness.cpu_assignment import allocate_whole_cores
from swefficiency.harness.docker_build import (
    BuildImageError,
    build_container,
    build_env_images,
    close_logger,
    create_container_from_image,
    setup_logger,
)
from swefficiency.harness.docker_utils import (
    clean_images,
    cleanup_container,
    copy_to_container,
    exec_run_with_timeout,
    list_images,
    remove_image,
    should_remove,
)
from swefficiency.harness.grading import get_logs_eval
from swefficiency.harness.log_parsers import MAP_REPO_TO_PARSER
from swefficiency.harness.test_spec import (
    COARSE_COVERAGE_AST_SCRIPT_LOCATION,
    COVERAGE_ANALYSIS_SCRIPT_LOCATION,
    DEFAULT_CORRECTNESS_TEST_OUTPUT_LOCATION,
    DEFAULT_COVERAGE_DATA_DIR,
    DEFAULT_COVERING_TESTS_LOCATION,
    DEFAULT_SINGLE_THREAD_COVERING_TESTS_LOCATION,
    INTROSPECTION_GUARD_CMD_LOCATION,
    PERF_CPROFILE_OUTPUT_LOCATION,
    PERF_WORKLOAD_SCRIPT_LOCATION,
    RAW_COVERAGE_OUTPUT_DIR,
    TestSpec,
    check_ast_result,
    make_test_spec,
    parse_coverage_report,
    parse_perf_output,
)
from swefficiency.harness.utils import load_swefficiency_dataset, str2bool


class EvaluationError(Exception):
    def __init__(self, instance_id, message, logger):
        super().__init__(message)
        self.super_str = super().__str__()
        self.instance_id = instance_id
        self.log_file = logger.log_file
        self.logger = logger

    def __str__(self):
        return (
            f"Evaluation error for {self.instance_id}: {self.super_str}\n"
            f"Check ({self.log_file}) for more information."
        )


def get_validation_report(
    test_spec: TestSpec,
    prediction: dict[str, str],
    log_path: str,
    include_tests_status: bool,
) -> dict[str, Any]:
    """
    Generate a report of model evaluation results from a prediction, task instance,
    and evaluation log.

    Args:
        test_spec (dict): test spec containing keys "instance_id", "FAIL_TO_PASS", and "PASS_TO_PASS"
        prediction (dict): prediction containing keys "instance_id", "model_name_or_path", and "model_patch"
        log_path (str): path to evaluation log
        include_tests_status (bool): whether to include the status of each test in the returned report
    Returns:
        report (dict): report of metrics
    """
    report_map = {}

    instance_id = prediction[KEY_INSTANCE_ID]
    if instance_id not in report_map:
        report_map[instance_id] = {
            "patch_is_None": False,
            "patch_exists": False,
            "patch_successfully_applied": False,
            "resolved": False,
        }

    # Check if the model patch exists
    if prediction["model_patch"] is None:
        report_map[instance_id]["none"] = True
        return report_map
    report_map[instance_id]["patch_exists"] = True

    # Get evaluation logs
    eval_sm, found = get_logs_eval(log_path)

    if not found:
        return report_map
    report_map[instance_id]["patch_successfully_applied"] = True

    report = {
        "PASS": [k for k, v in eval_sm.items() if v == TestStatus.PASSED.value],
        "FAIL": [k for k, v in eval_sm.items() if v == TestStatus.FAILED.value],
    }
    if len(report["FAIL"]) == 0 and len(report["PASS"]) > 0:
        report_map[instance_id]["resolved"] = True

    if include_tests_status:
        report_map[instance_id]["tests_status"] = report  # type: ignore

    return report_map


def try_to_apply_patch(container, instance_id, logger, base_commit=None):
    # Attempt to apply patch to container
    val = container.exec_run(
        "git apply -v /tmp/patch.diff",
        workdir="/testbed",
        user="root",
    )
    if val.exit_code != 0:
        logger.info(f"Failed to apply patch to container, trying again...")

        # # try "patch --batch --fuzz=5 -p1 -i {patch_path}" to try again
        # val = container.exec_run(
        #     "patch --batch --fuzz=5 -p1 -i /tmp/patch.diff",
        #     workdir="/testbed",
        #     user="root",
        # )
        # if val.exit_code != 0:
        logger.info(f"{APPLY_PATCH_FAIL}:\n{val.output.decode('utf-8')}")

        # Try to revert to base_commit first.
        if base_commit is not None:
            logger.info(f"Reverting to base commit {base_commit}...")
            revert_val = container.exec_run(
                f"git reset --hard {base_commit}",
                workdir="/testbed",
                user="root",
            )
            print("REVERT OUTPUT:", revert_val.output.decode("utf-8"))
            if revert_val.exit_code != 0:
                logger.info(
                    f"Failed to revert to base commit {base_commit}:\n{revert_val.output.decode('utf-8')}"
                )
                raise EvaluationError(
                    instance_id,
                    f"Failed to revert to base commit {base_commit}:\n{revert_val.output.decode('utf-8')}",
                    logger,
                )
            try_to_apply_patch(container, instance_id, logger, None)

        else:
            # Raise an error with the instance ID and the output of the command
            raise EvaluationError(
                instance_id,
                f"{APPLY_PATCH_FAIL}:\n{val.output.decode('utf-8')}",
                logger,
            )
        # else:
        #     logger.info(f"{APPLY_PATCH_PASS}:\n{val.output.decode('utf-8')}")
    else:
        logger.info(f"{APPLY_PATCH_PASS}:\n{val.output.decode('utf-8')}")


def run_instance(
    test_spec: TestSpec,
    pred: dict,
    rm_image: bool,
    force_rebuild: bool,
    client: docker.DockerClient,
    run_id: str,
    timeout: int | None = None,
    global_cpu_groups: list[list[int]] | None = None,
    run_coverage: bool = False,
    run_perf: bool = False,
    run_perf_profiling: bool = False,
    run_correctness: bool = False,
    push_to_dockerhub: bool = False,
    use_dockerhub_images: bool = False,
    use_podman: bool = False,  # Flip this to true if on SLURM or using podman without cgroups.
    force_rerun: bool = False,
):
    """
    Run a single instance with the given prediction.

    Args:
        test_spec (TestSpec): TestSpec instance
        pred (dict): Prediction w/ model_name_or_path, model_patch, instance_id
        rm_image (bool): Whether to remove the image after running
        force_rebuild (bool): Whether to force rebuild the image
        client (docker.DockerClient): Docker client
        run_id (str): Run ID
        timeout (int): Timeout for running tests
    """
    # Set up logging directory
    instance_id = test_spec.instance_id
    model_name_or_path = pred.get("model_name_or_path", "None").replace("/", "__")
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / instance_id

    if force_rerun:
        shutil.rmtree(log_dir)

    log_dir.mkdir(parents=True, exist_ok=True)

    instance_report = {
        "perf_report": None,
        "correctness_report": None,
    }

    # Link the image build dir in the log dir
    build_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key.replace(
        ":", "__"
    )
    image_build_link = log_dir / "image_build_dir"
    if not image_build_link.exists():
        try:
            # link the image build dir in the log dir
            image_build_link.symlink_to(build_dir.absolute(), target_is_directory=True)
        except:
            # some error, idk why
            pass
    log_file = log_dir / "run_instance.log"

    # Set up report file + logger
    report_path = log_dir / "report.json"
    if report_path.exists():
        return instance_id, json.loads(report_path.read_text())

    logger = setup_logger(instance_id, log_file)

    current_thread_name = threading.current_thread().name
    thread_idx = int(current_thread_name.split("_")[-1])

    # cpu_groups = ",".join([str(e) for e in global_cpu_groups[thread_idx]])
    cpu_groups = global_cpu_groups[thread_idx] if global_cpu_groups else None
    dockerhub_image_key = (
        f"ghcr.io/swefficiency/swefficiency-images:{test_spec.instance_id}"
    )

    additional_exec_args = {}
    if use_podman:
        additional_exec_args = {"taskset_cpus": cpu_groups}

    # Run the instance
    container = None
    try:
        logger.info(f"Test spec version: {test_spec.version}")

        # Build + start instance container (instance image should already be built)
        if use_dockerhub_images:
            logger.info(f"Using DockerHub image: {dockerhub_image_key}")

            # We need to pull the image first for it to show up in local registry.
            # Include fully qualified name for easier resolving.
            while True:
                try:
                    if use_podman:
                        # For some reason, podman does not support pulling images with the fully qualified name automatically.
                        subprocess.run(
                            f"podman pull {dockerhub_image_key}", shell=True, check=True
                        )
                    else:
                        client.images.pull(dockerhub_image_key)
                    break
                except Exception as e:
                    time.sleep(5)

            # Create container from image
            container = create_container_from_image(
                dockerhub_image_key,
                test_spec,
                run_id,
                client,
                logger,
                cpu_groups=cpu_groups,
            )
            container.start()
        else:
            container = build_container(
                test_spec, client, run_id, logger, rm_image, force_rebuild, cpu_groups
            )
            container.start()

        logger.info(f"Container for {instance_id} started: {container.id}")

        # Copy model prediction as patch file to container
        patch_file = Path(log_dir / "patch.diff")

        model_patch_text = pred["model_patch"]

        # Add newline if it doesn't end with one
        if model_patch_text and not model_patch_text.endswith("\n"):
            model_patch_text += "\n"

        patch_file.write_text(model_patch_text or "")
        logger.info(
            f"Intermediate patch for {instance_id} written to {patch_file}, now applying to container..."
        )
        copy_to_container(container, patch_file, Path("/tmp/patch.diff"))

        if run_coverage:
            # Copy treesitter script as patch file to container
            treesitter_file = Path(log_dir) / "treesitter_compare.py"

            with ir.as_file(
                ir.files(swefficiency.harness).joinpath("_meaningful_edit.py")
            ) as treesitter_compare_script_f:
                treesitter_compare_script = treesitter_compare_script_f.read_text(
                    encoding="utf-8"
                )

            treesitter_file.write_text(treesitter_compare_script)
            logger.info(
                f"Treesitter Python script for {instance_id} written to {treesitter_file}, now applying to container..."
            )
            copy_to_container(
                container, treesitter_file, Path("/tmp/treesitter_compare.py")
            )

            # STEP 1: Try to verify that edit is meaningful using treesitter. After this stage, the
            # repository should be reset to the preedit state.
            ast_file = Path(log_dir / "ast.sh")
            ast_file.write_text(test_spec.ast_meaningful_script)
            logger.info(
                f"Treesitter bash script for {instance_id} written to {ast_file}, now applying to container..."
            )
            copy_to_container(container, ast_file, Path("/ast.sh"))

            ast_output, timed_out, ast_runtime = exec_run_with_timeout(
                container, "/bin/bash /ast.sh", timeout, **additional_exec_args
            )
            ast_output_file = Path(log_dir / "ast_output.txt")

            ast_output_file.write_text(ast_output)

            with open(ast_output_file, "a") as ast_output_file_stream:
                if timed_out:
                    ast_output_file_stream.write(
                        f"\n\nTimeout error: {timeout} seconds exceeded."
                    )
                    raise EvaluationError(
                        instance_id,
                        f"Test timed out after {timeout} seconds.",
                        logger,
                    )

                # Check AST output between AST start and end tags that not all changes are "NON_MEANINGFUL" (at least some Warnings or meaningful edits).
                passed_ast_check = check_ast_result(ast_output)
                if not passed_ast_check:
                    ast_output_file_stream.write(
                        f"\nAST parsing failed error, non meaningful edits."
                    )
                    raise EvaluationError(
                        instance_id,
                        f"AST Parsing failed, non meaningful edits.",
                        logger,
                    )
                logger.info(f"Patch passed AST analysis check...")

            # STEP 2: Run test coverage to determine if any tests in the repo cover the edits we'd like to test.
            # Generally, we would first would like to apply the test patch without applying the source patch, then run coverage.
            # Higher odds that any added test cover the perf changes.

            # Copy treesitter script as patch file to container
            ast_coverage_script_file = Path(log_dir) / "coverage_ast.py"

            with ir.as_file(
                ir.files(swefficiency.harness).joinpath("_coverage_ast.py")
            ) as coverage_ast_script_text_f:
                script_text = coverage_ast_script_text_f.read_text(encoding="utf-8")

            ast_coverage_script_file.write_text(script_text)
            logger.info(
                f"AST coverage script for {instance_id} written to {ast_coverage_script_file}, now applying to container..."
            )
            copy_to_container(
                container,
                ast_coverage_script_file,
                Path(COARSE_COVERAGE_AST_SCRIPT_LOCATION),
            )

            coverage_file = Path(log_dir / "coverage.sh")
            coverage_file.write_text(test_spec.coverage_script)
            logger.info(
                f"Coverage script for {instance_id} written to {coverage_file}; copying to container..."
            )
            copy_to_container(container, coverage_file, Path("/coverage.sh"))

            coverage_analysis_file = Path(log_dir / "coverage_analysis.py")

            with ir.as_file(
                ir.files(swefficiency.harness).joinpath("_coverage_analysis2.py")
            ) as coverage_analysis_script_text_f:
                coverage_analysis_script_text = (
                    coverage_analysis_script_text_f.read_text(encoding="utf-8")
                )

            coverage_analysis_file.write_text(coverage_analysis_script_text)
            logger.info(
                f"Coverage analysis script for {instance_id} written to {coverage_analysis_file}; copying to container..."
            )
            copy_to_container(
                container,
                coverage_analysis_file,
                Path(COVERAGE_ANALYSIS_SCRIPT_LOCATION),
            )

            coverage_output, timed_out, coverage_runtime = exec_run_with_timeout(
                container, "/bin/bash /coverage.sh", timeout, **additional_exec_args
            )
            coverage_output_path = log_dir / "coverage_output.txt"
            logger.info(f"Coverage runtime: {coverage_runtime:_.2f} seconds")
            with open(coverage_output_path, "w") as f:
                f.write(coverage_output)
                logger.info(
                    f"Coverage output for {instance_id} written to {coverage_output_path}"
                )
                if timed_out:
                    f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                    raise EvaluationError(
                        instance_id,
                        f"Test timed out after {timeout} seconds.",
                        logger,
                    )

            # Copy coverage files from container. Both raw status and processed coverage data.
            test_status_tar_stream, _ = container.get_archive(RAW_COVERAGE_OUTPUT_DIR)
            test_status_tar_path = log_dir / "test_status.tar"
            with open(test_status_tar_path, "wb") as f:
                for chunk in test_status_tar_stream:
                    f.write(chunk)

            coverage_files_tar_stream, _ = container.get_archive(
                DEFAULT_COVERAGE_DATA_DIR
            )
            coverage_files_tar_path = log_dir / "coverage_files.tar"
            with open(coverage_files_tar_path, "wb") as f:
                for chunk in coverage_files_tar_stream:
                    f.write(chunk)

            covered_test_files = parse_coverage_report(coverage_output)

            if not covered_test_files:
                return

            print("=============================================")
            print("FOUND COVERAGE REPORT!")
            print(instance_id)
            print(len(covered_test_files))

            matching_tests = Path(log_dir / "covering_tests.txt")
            matching_tests.write_text("\n".join(covered_test_files))

        if run_perf:
            perf_summary_file = log_dir / "perf_summary.txt"
            if not perf_summary_file.exists():
                workload_text = test_spec.workload
                if not workload_text:
                    raise EvaluationError(
                        instance_id,
                        "Perf workload not found in prediction.",
                        logger,
                    )

                # Copy over the "workload.py" and the "covering_tests.txt.
                workload_file = Path(log_dir / "workload.py")
                workload_file.write_text(workload_text)

                logger.info(
                    f"Perf workload script for {instance_id} from {workload_file}; copying to container..."
                )
                copy_to_container(
                    container, workload_file, Path(PERF_WORKLOAD_SCRIPT_LOCATION)
                )

                # Write performance script to the container.
                perf_workload_file = Path(log_dir / "perf.sh")
                perf_workload_file.write_text(test_spec.performance_script)
                logger.info(
                    f"Perf script for {instance_id} written to {perf_workload_file}; copying to container..."
                )
                copy_to_container(container, perf_workload_file, Path("/perf.sh"))

                # Write performance profiling script to the container.
                perf_profiling_file = Path(log_dir / "perf_profiling.sh")
                perf_profiling_file.write_text(test_spec.performance_profiling_script)
                logger.info(
                    f"Perf profiling script for {instance_id} written to {perf_profiling_file}; copying to container..."
                )
                copy_to_container(
                    container, perf_profiling_file, Path("/perf_profiling.sh")
                )

                # Run pre-edit performance.
                pre_edit_perf_output, timed_out, pre_edit_perf_runtime = (
                    exec_run_with_timeout(
                        container, "/bin/bash /perf.sh", timeout, **additional_exec_args
                    )
                )
                pre_edit_perf_output_path = log_dir / "perf_output_preedit.txt"

                logger.info(
                    f"Pre-edit perf runtime: {pre_edit_perf_runtime:_.2f} seconds"
                )
                with open(pre_edit_perf_output_path, "w") as f:
                    f.write(pre_edit_perf_output)
                    logger.info(
                        f"Pre-edit perf output for {instance_id} written to {pre_edit_perf_output_path}"
                    )
                    if timed_out:
                        f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                        raise EvaluationError(
                            instance_id,
                            f"Perf workload timed out after {timeout} seconds.",
                            logger,
                        )
                preedit_runtime_mean, preedit_runtime_sd = parse_perf_output(
                    pre_edit_perf_output
                )

                if run_perf_profiling:
                    # Run performance profiling script.
                    perf_profiling_output, timed_out, perf_profiling_runtime = (
                        exec_run_with_timeout(
                            container,
                            "/bin/bash /perf_profiling.sh",
                            timeout,
                            **additional_exec_args,
                        )
                    )
                    perf_profiling_output_path = (
                        log_dir / "perf_profiling_output_preedit.txt"
                    )

                    logger.info(
                        f"Perf profiling runtime: {perf_profiling_runtime:_.2f} seconds"
                    )
                    with open(perf_profiling_output_path, "w") as f:
                        f.write(perf_profiling_output)
                        logger.info(
                            f"Perf profiling output for {instance_id} written to {perf_profiling_output_path}"
                        )
                        if timed_out:
                            f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                            raise EvaluationError(
                                instance_id,
                                f"Perf profiling workload timed out after {timeout} seconds.",
                                logger,
                            )

                    # Download pre-edit cprofile output from the container.
                    preedit_cprofile_output_stream, _ = container.get_archive(
                        PERF_CPROFILE_OUTPUT_LOCATION
                    )
                    preedit_profile_tar_path = log_dir / "workload_preedit_cprofile.tar"
                    with open(preedit_profile_tar_path, "wb") as f:
                        for chunk in preedit_cprofile_output_stream:
                            f.write(chunk)

                try_to_apply_patch(
                    container, instance_id, logger, test_spec.base_commit
                )

                # Run post edit.
                post_edit_perf_output, timed_out, post_edit_perf_runtime = (
                    exec_run_with_timeout(
                        container, "/bin/bash /perf.sh", timeout, **additional_exec_args
                    )
                )
                post_edit_perf_output_path = log_dir / "perf_output_postedit.txt"

                # Check post diff runtime performance.
                logger.info(
                    f"Post-edit perf runtime: {post_edit_perf_runtime:_.2f} seconds"
                )
                with open(post_edit_perf_output_path, "w") as f:
                    f.write(post_edit_perf_output)
                    logger.info(
                        f"Post-edit perf output for {instance_id} written to {post_edit_perf_output_path}"
                    )
                    if timed_out:
                        f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                        raise EvaluationError(
                            instance_id,
                            f"Post-edit perf workload timed out after {timeout} seconds.",
                            logger,
                        )

                if run_perf_profiling:
                    # Run performance profiling script.
                    (
                        post_edit_perf_profiling_output,
                        timed_out,
                        post_edit_perf_profiling_runtime,
                    ) = exec_run_with_timeout(
                        container,
                        "/bin/bash /perf_profiling.sh",
                        timeout,
                        **additional_exec_args,
                    )
                    post_edit_perf_profiling_output_path = (
                        log_dir / "perf_profiling_output_postedit.txt"
                    )

                    logger.info(
                        f"Post-edit perf profiling runtime: {post_edit_perf_profiling_runtime:_.2f} seconds"
                    )
                    with open(post_edit_perf_profiling_output_path, "w") as f:
                        f.write(post_edit_perf_profiling_output)
                        logger.info(
                            f"Post-edit perf profiling output for {instance_id} written to {post_edit_perf_profiling_output_path}"
                        )
                        if timed_out:
                            f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                            raise EvaluationError(
                                instance_id,
                                f"Post-edit perf profiling workload timed out after {timeout} seconds.",
                                logger,
                            )

                    # Download post-edit cprofile output from the container.
                    postedit_cprofile_output_stream, _ = container.get_archive(
                        PERF_CPROFILE_OUTPUT_LOCATION
                    )
                    postedit_profile_tar_path = (
                        log_dir / "workload_postedit_cprofile.tar"
                    )
                    with open(postedit_profile_tar_path, "wb") as f:
                        for chunk in postedit_cprofile_output_stream:
                            f.write(chunk)

                postedit_runtime_mean, postedit_runtime_sd = parse_perf_output(
                    post_edit_perf_output
                )
                improvement = (
                    postedit_runtime_mean - preedit_runtime_mean
                ) / preedit_runtime_mean

                perf_summary_file.write_text(
                    "\n".join(
                        [
                            f"Before Mean: {preedit_runtime_mean}",
                            f"Before SD: {preedit_runtime_sd}",
                            f"After Mean: {postedit_runtime_mean}",
                            f"After SD: {postedit_runtime_sd}",
                            f"Improvement: {improvement * 100:.2f}%",
                        ]
                    )
                )

                if improvement >= 0:
                    flag_bad_workload_file = Path(log_dir / "flag_bad_workload.txt")
                    flag_bad_workload_file.write_text(
                        f"Improvement is {improvement * 100:.2f}%, which is not a performance improvement. "
                        f"Please check the workload for {instance_id}."
                    )

                if run_perf_profiling:
                    # Untar the cprofile output file.
                    subprocess.run(
                        [
                            "tar",
                            "-xf",
                            str(preedit_profile_tar_path),
                            "-C",
                            str(log_dir),
                        ],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    # Rename the extracted file "workload_cprofile.prof" to "workload_preedit_cprofile.prof".
                    preedit_profile_path = log_dir / "workload_cprofile.prof"
                    if preedit_profile_path.exists():
                        preedit_profile_new_path = (
                            log_dir / "workload_preedit_cprofile.prof"
                        )
                        preedit_profile_path.rename(preedit_profile_new_path)
                        logger.info(
                            f"Renamed {preedit_profile_path} to {preedit_profile_new_path}"
                        )

                    os.remove(
                        preedit_profile_tar_path
                    )  # Remove the tar file after extracting

                    # Untar the cprofile output file and name it "workload_cprofile_postedit.tar".
                    subprocess.run(
                        [
                            "tar",
                            "-xf",
                            str(postedit_profile_tar_path),
                            "-C",
                            str(log_dir),
                        ],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

                    # Rename the extracted file "workload_cprofile.prof" to "workload_postedit_cprofile.prof".
                    postedit_profile_path = log_dir / "workload_cprofile.prof"
                    if postedit_profile_path.exists():
                        postedit_profile_new_path = (
                            log_dir / "workload_postedit_cprofile.prof"
                        )
                        postedit_profile_path.rename(postedit_profile_new_path)
                        logger.info(
                            f"Renamed {postedit_profile_path} to {postedit_profile_new_path}"
                        )
                    os.remove(
                        postedit_profile_tar_path
                    )  # Remove the tar file after extracting

                instance_report["perf_report"] = {
                    "before_mean": preedit_runtime_mean,
                    "before_sd": preedit_runtime_sd,
                    "after_mean": postedit_runtime_mean,
                    "after_sd": postedit_runtime_sd,
                    "improvement": improvement,
                }

        if run_correctness:
            # Write prettified JSON.
            subtest_status_output = Path(log_dir / "covering_test_status.json")

            # Skip if the subtest status output already exists.
            if not subtest_status_output.exists():
                # At a high level, we should apply patch, run tests, download the test status, then parse.
                correctness_tests = test_spec.covering_tests
                if not correctness_tests:
                    raise EvaluationError(
                        instance_id,
                        "Correctness tests not found in test spec.",
                        logger,
                    )

                # Write correctness tests to the container.
                covering_tests_file = Path(log_dir / "covering_tests.txt")
                covering_tests_file.write_text("\n".join(correctness_tests))
                logger.info(
                    f"Covering tests {instance_id} written to {covering_tests_file}; copying to container..."
                )
                copy_to_container(
                    container,
                    covering_tests_file,
                    Path(DEFAULT_COVERING_TESTS_LOCATION),
                )

                single_thread_tests = test_spec.single_thread_tests or []

                single_thread_tests_file = Path(log_dir / "single_thread_tests.txt")
                single_thread_tests_file.write_text("\n".join(single_thread_tests))
                logger.info(
                    f"Single threaded tests {instance_id} written to {single_thread_tests_file}; copying to container..."
                )
                copy_to_container(
                    container,
                    single_thread_tests_file,
                    Path(DEFAULT_SINGLE_THREAD_COVERING_TESTS_LOCATION),
                )

                if not run_perf:
                    # Assume that patch is applied already if we run perf.
                    try_to_apply_patch(container, instance_id, logger)

                # # For correctness, we first revert the test files to the base commit. This ensures that patches cannot
                # # reward hack by modifying tests.
                # revert_val = container.exec_run(
                #     f"git checkout {test_spec.base_commit} -- {' '.join(correctness_tests)}",
                #     workdir="/testbed",
                # )

                paths = correctness_tests
                base = test_spec.base_commit
                revert_test_cmd = (
                    "/bin/bash -lc 'commit={c}; for p in {paths}; do "
                    'if [ -e "$p" ]; then '
                    '  git checkout "$commit" -- "$p" || {{ echo "Failed to revert $p" >&2; exit 1; }}; '
                    "fi; "
                    "done'".format(
                        c=shlex.quote(base),
                        paths=" ".join(shlex.quote(p) for p in paths)
                        or "''",  # safe if empty
                    )
                )
                revert_val = container.exec_run(revert_test_cmd, workdir="/testbed")

                logger.info(
                    f"Revert command output:\n{revert_val.output.decode('utf-8')}"
                )

                if revert_val.exit_code != 0:
                    logger.info(
                        f"Failed to revert correctness tests to base commit:\n{revert_val.output.decode('utf-8')}"
                    )
                    raise EvaluationError(
                        instance_id,
                        f"Failed to revert correctness tests to base commit:\n{revert_val.output.decode('utf-8')}",
                        logger,
                    )

                # Write introspection guard script to the container
                if instance_id not in STACKFRAME_CHECK_EXCEPTONS:
                    introspection_guard_helper_file = Path(
                        log_dir / "introspection_patch_check.py"
                    )

                    with ir.as_file(
                        ir.files(swefficiency.harness).joinpath(
                            "_introspection_patch_check.py"
                        )
                    ) as f:
                        introspection_guard_helper_text = f.read_text(encoding="utf-8")

                    introspection_guard_helper_file.write_text(
                        introspection_guard_helper_text
                    )
                    logger.info(
                        f"Introspection guard script for {instance_id} written to {introspection_guard_helper_file}; copying to container..."
                    )
                    copy_to_container(
                        container,
                        introspection_guard_helper_file,
                        Path(INTROSPECTION_GUARD_CMD_LOCATION),
                    )

                    introspection_guard_script = Path(
                        log_dir / "introspection_guard.sh"
                    )
                    introspection_guard_script.write_text(
                        test_spec.introspection_guard_script
                    )
                    logger.info(
                        f"Introspection guard bash script for {instance_id} written to {introspection_guard_script}; copying to container..."
                    )
                    copy_to_container(
                        container,
                        introspection_guard_script,
                        Path("/introspection_guard.sh"),
                    )

                    # Run introspection guard to ensure no forbidden modules are imported in the tests.
                    introspection_result = container.exec_run(
                        "/bin/bash /introspection_guard.sh",
                        workdir="/testbed",
                    )
                    if introspection_result.exit_code != 0:
                        logger.info(
                            f"Failed to run introspection guard:\n{introspection_result.output.decode('utf-8')[-1000:]}"
                        )
                        raise EvaluationError(
                            instance_id,
                            f"Failed to run introspection guard:\n{introspection_result.output.decode('utf-8')[-1000:]}",  # Last 1000 chars of output
                            logger,
                        )

                logger.info(f"Introspection guard passed for {instance_id}.")

                # Write correctness script to the container.
                correctness_file = Path(log_dir / "correctness.sh")
                correctness_file.write_text(test_spec.correctness_script)
                logger.info(
                    f"Correctness script for {instance_id} written to {correctness_file}; copying to container..."
                )
                copy_to_container(container, correctness_file, Path("/correctness.sh"))

                correctness_output, timed_out, correctness_runtime = (
                    exec_run_with_timeout(
                        container,
                        "/bin/bash /correctness.sh",
                        timeout,
                        **additional_exec_args,
                    )
                )
                correctness_output_path = log_dir / "correctness_output.txt"

                logger.info(f"Correctness runtime: {correctness_runtime:_.2f} seconds")
                with open(correctness_output_path, "w") as f:
                    f.write(correctness_output)
                    logger.info(
                        f"Correctness output for {instance_id} written to {correctness_output_path}"
                    )
                    if timed_out:
                        f.write(f"\n\nTimeout error: {timeout} seconds exceeded.")
                        raise EvaluationError(
                            instance_id,
                            f"Correctness workload timed out after {timeout} seconds.",
                            logger,
                        )

                # Copy correctness tests from container.
                test_status_tar_stream, _ = container.get_archive(
                    DEFAULT_CORRECTNESS_TEST_OUTPUT_LOCATION
                )
                test_status_tar_path = log_dir / "test_status.tar"
                with open(test_status_tar_path, "wb") as f:
                    for chunk in test_status_tar_stream:
                        f.write(chunk)

                # Untar the test status tar file.
                subprocess.run(
                    ["tar", "-xf", str(test_status_tar_path), "-C", str(log_dir)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                test_results = {}
                for test_output in (log_dir / "raw_correctness_output").glob("*.txt"):
                    with open(test_output, "r") as f:
                        content = f.read()
                        file_test_results = MAP_REPO_TO_PARSER[test_spec.repo](content)
                        test_results.update(file_test_results)

                with open(subtest_status_output, "w") as f:
                    json.dump(test_results, f, indent=4)

                instance_report["correctness_report"] = {
                    "test_results": test_results,
                    "PASS_TO_PASS": test_spec.PASS_TO_PASS,
                }

    except EvaluationError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
    except BuildImageError as e:
        error_msg = traceback.format_exc()
        logger.info(error_msg)
        print(e)
    except Exception as e:
        error_msg = (
            f"Error in evaluating model for {instance_id}: {e}\n"
            f"{traceback.format_exc()}\n"
            f"Check ({logger.log_file}) for more information."
        )
        logger.error(error_msg)
    finally:
        if push_to_dockerhub and not use_dockerhub_images:
            logger.info(
                f"Pushing image {test_spec.instance_image_key} to DockerHub since we rebuilt and marked it..."
            )

            # Run subprocess to push the image to DockerHub.
            base_docker_image = test_spec.instance_image_key
            base_dockerhub_image = f"ghcr.io/{dockerhub_image_key}"

            base_command = f"docker tag {base_docker_image} {base_dockerhub_image} && docker push {base_dockerhub_image}"
            try:
                subprocess.run(base_command, shell=True, check=True)
                logger.info(f"Successfully pushed {base_dockerhub_image} to DockerHub.")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to push {base_dockerhub_image} to DockerHub: {e}")
                print(f"Failed to push {base_dockerhub_image} to DockerHub: {e}")

        # Remove instance container + image, close logger
        cleanup_container(client, container, logger)
        if rm_image:
            image_key = test_spec.instance_image_key
            if use_dockerhub_images:
                image_key = dockerhub_image_key

            remove_image(client, image_key, logger)
        close_logger(logger)
    return instance_id, instance_report


def run_instances(
    predictions: dict,
    instances: list,
    cache_level: str,
    clean: bool,
    force_rebuild: bool,
    max_workers: int,
    run_id: str,
    timeout: int,
    run_coverage=False,
    run_perf=False,
    run_perf_profiling=False,
    run_correctness=False,
    push_to_dockerhub=False,
    use_dockerhub_images=False,
    one_per_version_debug=False,
    use_podman=False,  # Set to True if running on SLURM or using podman without cgroups.
    force_rerun=True,
):
    """
    Run all instances for the given predictions in parallel.

    Args:
        predictions (dict): Predictions dict generated by the model
        instances (list): List of instances
        cache_level (str): Cache level
        clean (bool): Clean images above cache level
        force_rebuild (bool): Force rebuild images
        max_workers (int): Maximum number of workers
        run_id (str): Run ID
        timeout (int): Timeout for running tests
    """
    client = docker.from_env(timeout=3600)

    test_specs = []
    working_instances = []

    observed_versions = None
    if one_per_version_debug:
        observed_versions = set()

    for instance in instances:
        try:
            test_specs.append(make_test_spec(instance, observed_versions))
            working_instances.append(instance)
        except NotImplementedError:
            continue
        except RuntimeError:
            print("runtime error")
            continue

        except Exception as e:
            print(f"Error making test spec for {instance[KEY_INSTANCE_ID]}: {e}")
            traceback.print_exc()
            continue

    instances = working_instances

    # print number of existing instance images
    instance_image_ids = {x.instance_image_key for x in test_specs}
    existing_images = {
        tag
        for i in client.images.list(all=True)
        for tag in i.tags
        if tag in instance_image_ids
    }
    if not force_rebuild and len(existing_images):
        print(
            f"Found {len(existing_images)} existing instance images. Will reuse them."
        )

    # # Get number of CPUs and divy into groups.
    # global_cpu_groups = divide_cpus_among_workers(max_workers)

    # HACK: If we are running on SLURM, we need to allocate whole cores, should use something else in non VM setting.
    # For n2-standard-64, we have 64 vCPUs (2 threads per core), so we can allocate 16 workers with 4 vCPUs each.
    global_cpu_groups = allocate_whole_cores(
        max_workers,
        vcpus_per_worker=4,
        threads_per_core=2,
        reserve_cores=4,
    )

    # run instances in parallel
    print(f"Running {len(instances)} instances...")
    per_instance_results = {}
    with tqdm(total=len(instances), smoothing=0) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a future for running each instance
            futures = {
                executor.submit(
                    run_instance,
                    test_spec,
                    predictions[test_spec.instance_id],
                    should_remove(
                        test_spec.instance_image_key,
                        cache_level,
                        clean,
                        existing_images,
                    ),
                    force_rebuild,
                    client,
                    run_id,
                    timeout,
                    global_cpu_groups,
                    run_coverage,
                    run_perf,
                    run_perf_profiling,
                    run_correctness,
                    push_to_dockerhub,
                    use_dockerhub_images,
                    use_podman,
                    force_rerun,
                ): None
                for test_spec in test_specs
            }
            # Wait for each future to complete
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    # Update progress bar, check if instance ran successfully
                    result = future.result()
                    if result is not None:
                        print(f"Instance {result[0]} completed successfully.")
                        instance_id, instance_report = result
                        per_instance_results[instance_id] = instance_report
                except Exception as e:
                    traceback.print_exc()
                    continue
    print("All instances run.")

    # Assume first prediction model name.
    first_prediction = list(predictions.values())[0]
    model_name_or_path = first_prediction.get("model_name_or_path", "None").replace(
        "/", "__"
    )
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path

    filename = f"validation_report_{run_id}.json"
    report_path = Path(log_dir) / filename
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(per_instance_results, f, indent=4)
    print(f"Results saved to {report_path}")


def get_dataset_from_preds(
    dataset_name: str,
    split: str,
    instance_ids: list,
    predictions: dict,
    run_id: str,
    exclude_completed: bool = True,
):
    """
    Return only instances that have predictions and are in the dataset.
    If instance_ids is provided, only return instances with those IDs.
    If exclude_completed is True, only return instances that have not been run yet.
    Note: in this version, we will still keep the data point even if the patch is empty.
    """
    # load dataset
    dataset = load_swefficiency_dataset(dataset_name, split)
    dataset_ids = {i[KEY_INSTANCE_ID] for i in dataset}

    if instance_ids:
        # check that all instance IDs are in the dataset
        instance_ids = set(instance_ids)
        if instance_ids - dataset_ids:
            raise ValueError(
                (
                    "Some instance IDs not found in dataset!"
                    f"\nMissing IDs:\n{' '.join(instance_ids - dataset_ids)}"
                )
            )
        # check that all instance IDs have predictions
        missing_preds = instance_ids - set(predictions.keys())
        if missing_preds:
            print(
                f"Warning: Missing predictions for {len(missing_preds)} instance IDs."
            )

    # check that all prediction IDs are in the dataset
    prediction_ids = set(predictions.keys())
    if prediction_ids - dataset_ids:
        raise ValueError(
            (
                "Some prediction IDs not found in dataset!"
                f"\nMissing IDs:\n{' '.join(prediction_ids - dataset_ids)}"
            )
        )

    if instance_ids:
        # filter dataset to just the instance IDs
        dataset = [i for i in dataset if i[KEY_INSTANCE_ID] in instance_ids]

    # check which instance IDs have already been run
    completed_ids = set()
    for instance in dataset:
        if instance[KEY_INSTANCE_ID] not in prediction_ids:
            # skip instances without predictions
            continue
        prediction = predictions[instance[KEY_INSTANCE_ID]]
        report_dir = (
            RUN_EVALUATION_LOG_DIR
            / run_id
            / prediction["model_name_or_path"].replace("/", "__")
            / prediction[KEY_INSTANCE_ID]
        )
        covering_tests_file = report_dir / "covering_test_status.json"
        perf_summary_file = report_dir / "perf_summary.txt"

        if covering_tests_file.exists() and perf_summary_file.exists():
            completed_ids.add(instance[KEY_INSTANCE_ID])

    if completed_ids and exclude_completed:
        # filter dataset to only instances that have not been run
        print(f"{len(completed_ids)} instances already run, skipping...")
        dataset = [i for i in dataset if i[KEY_INSTANCE_ID] not in completed_ids]
    return dataset


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


def get_gold_predictions(dataset_name: str, split: str):
    """
    Get gold predictions for the given dataset and split.
    """
    dataset = load_swefficiency_dataset(dataset_name, split)
    return [
        {
            KEY_INSTANCE_ID: datum[KEY_INSTANCE_ID],
            "model_patch": datum["patch"],
            "model_name_or_path": "gold",
        }
        for datum in dataset
    ]


def get_empty_predictions(dataset_name: str, split: str):
    """
    Get empty predictions for the given dataset and split.
    """
    dataset = load_swefficiency_dataset(dataset_name, split)
    return [
        {
            KEY_INSTANCE_ID: datum[KEY_INSTANCE_ID],
            "model_patch": "",
            "model_name_or_path": "empty",
        }
        for datum in dataset
    ]


def get_model_predictions(dataset_path: str):
    """
    Get model predictions from a JSONL file.
    """
    dataset = []
    with open(dataset_path, "r") as f:
        for line in f:
            pred = json.loads(line.strip())
            if KEY_INSTANCE_ID not in pred:
                raise ValueError(f"Missing {KEY_INSTANCE_ID} in prediction: {pred}")
            if "model_patch" not in pred:
                raise ValueError(f"Missing model_patch in prediction: {pred}")
            if "model_name_or_path" not in pred:
                raise ValueError(f"Missing model_name_or_path in prediction: {pred}")
            dataset.append(pred)
    return dataset


def get_model_predictions_sweagent(dataset_path: str):
    """
    Get model predictions from a JSONL file.
    """
    dataset = []
    with open(dataset_path, "r") as f:
        full_text = f.read().strip()
        full_dataset = json.loads(full_text)
        for pred in full_dataset.values():
            if KEY_INSTANCE_ID not in pred:
                raise ValueError(f"Missing {KEY_INSTANCE_ID} in prediction: {pred}")
            if "model_patch" not in pred:
                raise ValueError(f"Missing model_patch in prediction: {pred}")
            if "model_name_or_path" not in pred:
                raise ValueError(f"Missing model_name_or_path in prediction: {pred}")
            dataset.append(pred)
    return dataset


def delete_instance_container(client, dataset):
    instance_ids = [dp["instance_id"] for dp in dataset]
    container_names = set(
        [f"sweb.eval.{instance_id}.test" for instance_id in instance_ids]
    )
    container_list = client.containers.list(all=True)
    for container in container_list:
        if container.name in container_names:
            container.remove(force=True)


def main(
    dataset_name: str,
    split: str,
    instance_ids: list,
    max_workers: int,
    max_build_workers: int,
    force_rebuild: bool,
    cache_level: str,
    clean: bool,
    open_file_limit: int,
    run_id: str,
    timeout: int,
    allow_test_patch: bool,
    run_coverage: bool,
    run_perf: bool,
    run_perf_profiling: bool,
    run_correctness: bool,
    empty_patch: bool,
    model_predictions: str,
    gdrive_annotation_sheet: str,
    push_to_dockerhub: bool,
    use_dockerhub_images: bool,
    use_podman: bool,
    workload_predictions: str,
    force_rerun: bool,
):
    """
    Run evaluation harness for the given dataset and predictions.
    """
    # set open file limit
    assert len(run_id) > 0, "Run ID must be provided"
    resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))
    client = docker.from_env(timeout=3600)

    # load predictions as map of instance_id to prediction
    print("Using gold predictions - ignoring predictions_path")
    if empty_patch:
        predictions = get_empty_predictions(dataset_name, split)
    elif model_predictions:
        predictions = get_model_predictions(model_predictions)
        # predictions = get_model_predictions_sweagent(model_predictions)
    else:
        predictions = get_gold_predictions(dataset_name, split)
    predictions = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}

    # get dataset from predictions
    exclude_completed = True
    if force_rerun:
        exclude_completed = False

    dataset = get_dataset_from_preds(
        dataset_name,
        split,
        instance_ids,
        predictions,
        run_id,
        exclude_completed=exclude_completed,
    )

    if workload_predictions:
        # Load workload predictions from JSONL file.
        with open(workload_predictions, "r") as f:
            workload_preds = {}
            for line in f:
                pred = json.loads(line.strip())
                if KEY_INSTANCE_ID not in pred:
                    raise ValueError(
                        f"Missing {KEY_INSTANCE_ID} in workload prediction: {pred}"
                    )
                if "workload" not in pred:
                    raise ValueError(f"Missing workload in workload prediction: {pred}")
                workload_preds[pred[KEY_INSTANCE_ID]] = pred["workload"]

        # Replace the workload field in the dataset with the predictions.
        counter = 0
        for datum in dataset:
            instance_id = datum[KEY_INSTANCE_ID]
            if instance_id in workload_preds:
                workload_text = workload_preds[instance_id]
                if not workload_text or workload_text.strip() == "nan":
                    print(
                        f"Skipping {instance_id} as it has a workload that is just 'nan'."
                    )
                    datum["workload"] = ""
                else:
                    datum["workload"] = workload_text
                    counter += 1
            else:
                # Replace workload with empty string if not found.
                datum["workload"] = ""

        print(f"Loaded {counter} workload predictions from {workload_predictions}.")

    if gdrive_annotation_sheet:
        # Populate the workload field in the dataset.
        import gspread
        from gspread_dataframe import get_as_dataframe

        # Assumes you're already authenticated (https://docs.gspread.org/en/v6.1.4/).
        gc = gspread.service_account()
        spreadsheet = gc.open(gdrive_annotation_sheet)
        worksheet = spreadsheet.get_worksheet(0)

        df = get_as_dataframe(worksheet, header=0, index_col=None)

        print(f"Loaded {len(df)} rows from Google Sheet {gdrive_annotation_sheet}.")

        filtered_dataset = []
        for datum in dataset:
            instance_id = datum[KEY_INSTANCE_ID]
            if instance_id in df["instance_id"].values:
                row = df.loc[df["instance_id"] == instance_id].iloc[0]
                workload_text = str(row["workload"]).strip()
                status = str(row["status"]).strip()

                if workload_text and status in ["APPROVED", "NEEDS_ANNOTATE"]:
                    datum["workload"] = workload_text
                    datum["notes"] = str(row["notes"]).strip()
                    filtered_dataset.append(datum)
                    continue

            print(
                f"Instance {instance_id} not in sheet or doesn't have workload, skipping..."
            )

        dataset = filtered_dataset
        # predictions = {d[KEY_INSTANCE_ID]: predictions[d[KEY_INSTANCE_ID]] for d in dataset}

        for datum in dataset:
            # HACK: Add covering test information from the "data" folder.\
            instance_id = datum[KEY_INSTANCE_ID]
            instance_data_folder = Path(REPO_FOLDER / "data" / instance_id)
            if (instance_data_folder / "covering_tests.txt").exists():
                datum["covering_tests"] = (
                    (instance_data_folder / "covering_tests.txt")
                    .read_text()
                    .splitlines()
                )

    # Avoid rerunning instances that have already been run. This is a hack for SLURM tbh.
    if run_coverage:
        filtered_dataset = []
        filtered_predictions = {}
        for data in dataset:
            instance_id = data[KEY_INSTANCE_ID]
            pred = predictions.get(instance_id, None)
            if not pred:
                print(f"Skipping {instance_id} as it has no prediction.")
                continue

            ignore_with_test_patch = not allow_test_patch
            if ignore_with_test_patch and data.get("test_patch", False):
                print(f"Skipping {instance_id} as it has a test patch.")
                continue

            filtered_dataset.append(data)
            filtered_predictions[instance_id] = pred

        print("Original dataset size:", len(dataset))
        print("Filtered dataset size:", len(filtered_dataset))

        dataset = filtered_dataset
        predictions = filtered_predictions
        print(f"Filtered dataset to {len(dataset)} instances with predictions.")

        dataset.reverse()

    if run_perf:
        filtered_dataset = []
        filtered_predictions = {}
        for data in dataset:
            instance_id = data[KEY_INSTANCE_ID]
            pred = predictions.get(instance_id, None)
            if pred is None:
                print(f"Skipping {instance_id} as we don't have a prediction for it.")
                continue

            model_name_or_path = pred.get("model_name_or_path", "None").replace(
                "/", "__"
            )

            log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / instance_id
            workload_text = data["workload"]

            if workload_text.strip() == "nan":
                print(
                    f"Skipping {instance_id} as it has a workload that is just 'nan'."
                )
                continue
            # else:
            #     if log_dir.exists() and (log_dir / "perf.sh").exists():
            #         # Rerun only if notes column says re-run.
            #         if 'rerun' not in data.get("notes", ""):
            #             # Check if workload just contains "nan".:
            #             print(f"Skipping performance run for {instance_id} as it has a valid workload and has already been run and we did not indicate rerun.")
            #             continue

            filtered_dataset.append(data)
            filtered_predictions[instance_id] = pred

        print("Original dataset size:", len(dataset))
        print("Filtered dataset size:", len(filtered_dataset))
        print("Instances:", [d[KEY_INSTANCE_ID] for d in filtered_dataset])

        dataset = filtered_dataset
        predictions = filtered_predictions
        print(f"Filtered dataset to {len(dataset)} instances with predictions.")

    # Sort dataset by instance ID for reproducibility.
    dataset.sort(key=lambda x: x[KEY_INSTANCE_ID])
    random.seed(42)
    random.shuffle(dataset)  # Shuffle with fixed seed for load balancing.

    existing_images = list_images(client)
    delete_instance_container(client, dataset)
    print(f"Running {len(dataset)} unevaluated instances...")
    if not dataset:
        print("No instances to run.")
    elif use_dockerhub_images:
        # run instances using dockerhub images
        run_instances(
            predictions,
            dataset,
            cache_level,
            clean,
            force_rebuild,
            max_workers,
            run_id,
            timeout,
            run_coverage,
            run_perf,
            run_perf_profiling,
            run_correctness,
            use_dockerhub_images=use_dockerhub_images,
            use_podman=use_podman,
            force_rerun=force_rerun,
        )
    else:
        # build environment images + run instances
        build_env_images(client, dataset, force_rebuild, max_build_workers)
        # this time w/ golden predictions (patch)
        run_instances(
            predictions,
            dataset,
            cache_level,
            clean,
            force_rebuild,
            max_workers,
            run_id,
            timeout,
            run_coverage,
            run_perf,
            run_perf_profiling,
            run_correctness,
            push_to_dockerhub=push_to_dockerhub,
            use_podman=use_podman,
            force_rerun=force_rerun,
        )

    # this will remove the container in the gloden run
    delete_instance_container(client, dataset)

    # clean images + make final report
    clean_images(client, existing_images, cache_level, clean)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        default="sweperf/sweperf",
        type=str,
        help="Name of dataset or path to JSON file.",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Split of the dataset"
    )
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        help="Instance IDs to run (space separated)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of workers (should be <= 75%% of CPU cores)",
    )
    parser.add_argument(
        "--max_build_workers",
        type=int,
        default=4,
        help="Maximum number of workers for building images",
    )
    parser.add_argument(
        "--open_file_limit", type=int, default=4096, help="Open file limit"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1_800,
        help="Timeout (in seconds) for running tests for each instance",
    )
    parser.add_argument(
        "--force_rebuild",
        type=str2bool,
        default=False,
        help="Force rebuild of all images",
    )
    parser.add_argument(
        "--cache_level",
        type=str,
        choices=["none", "base", "env", "instance"],
        help="Cache level - remove images above this level",
        default="env",
    )
    # if clean is true then we remove all images that are above the cache level
    # if clean is false, we only remove images above the cache level if they don't already exist
    parser.add_argument(
        "--clean", type=str2bool, default=False, help="Clean images above cache level"
    )
    parser.add_argument(
        "--run_id", type=str, required=True, help="Run ID - identifies the run"
    )

    parser.add_argument(
        "--empty_patch", type=str2bool, default=False, help="Use empty patch."
    )
    parser.add_argument(
        "--model_predictions", type=str, help="Path to JSONL with model predictions."
    )
    parser.add_argument(
        "--gdrive_annotation_sheet",
        type=str,
        default=None,
        help="Google Drive annotation sheet.",
    )
    parser.add_argument(
        "--push_to_dockerhub",
        type=str2bool,
        default=False,
        help="Push images to DockerHub.",
    )
    parser.add_argument(
        "--use_dockerhub_images",
        type=str2bool,
        default=False,
        help="Use DockerHub images instead of building them.",
    )
    parser.add_argument(
        "--use_podman",
        type=str2bool,
        default=False,
        help="Use podman options instead of docker (uses taskset to run commands).",
    )
    parser.add_argument(
        "--workload_predictions",
        type=str,
        help="Path to JSONL with workload predictions.",
    )

    # 3 options: Coverage runs the coverage identification. Perf is to run workloads to measure before and after.
    parser.add_argument(
        "--allow_test_patch",
        action="store_true",
        help="Whether to keep PRs with no test patch",
    )
    parser.add_argument(
        "--run_coverage", type=str2bool, default=False, help="Run coverage stage."
    )
    parser.add_argument(
        "--run_perf", type=str2bool, default=False, help="Run perf stage."
    )
    parser.add_argument(
        "--run_perf_profiling",
        type=str2bool,
        default=False,
        help="Run perf profiling stage.",
    )
    parser.add_argument(
        "--run_correctness", type=str2bool, default=False, help="Run correctness stage."
    )

    parser.add_argument(
        "--force_rerun",
        type=str2bool,
        default=False,
        help="Force rerun even if results already exist.",
    )

    args = parser.parse_args()
    main(**vars(args))
