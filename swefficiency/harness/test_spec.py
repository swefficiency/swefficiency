from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import textwrap
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union, cast

from swefficiency.harness.constants import (
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    MAP_REPO_TO_INSTALL,
    MAP_REPO_VERSION_TO_SPECS,
    PASS_TO_PASS,
    USE_X86,
    SWEfficiencyInstance,
)
from swefficiency.harness.dockerfiles import (
    get_dockerfile_annotate_instance,
    get_dockerfile_base,
    get_dockerfile_env,
    get_dockerfile_instance,
)
from swefficiency.harness.utils import (
    get_environment_yml,
    get_requirements,
    get_test_directives,
)
from swefficiency.perf_filter import utils as perf_utils

DIFF_MODIFIED_FILE_REGEX = r"--- a/(.*)"


@dataclass
class TestSpec:
    """
    A dataclass that represents a test specification for a single instance of SWE-bench.
    """

    instance_id: str
    repo: str
    version: str
    repo_script_list: list[str]
    eval_script_list: list[str]
    env_script_list: list[str]
    arch: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    base_commit: str

    coverage_script_list: list[str]
    meaningful_edit_script_list: list[str]
    performance_script_list: list[str]
    performance_profiling_script_list: list[str]
    correctness_script_list: list[str]
    introspection_guard_script_list: list[str]

    workload: Optional[str]
    covering_tests: Optional[list[str]]
    single_thread_tests: Optional[list[str]]

    build_timeout: Optional[int]

    @property
    def setup_env_script(self):
        return (
            "\n".join(["#!/bin/bash", "set -exo pipefail"] + self.env_script_list)
            + "\n"
        )

    @property
    def eval_script(self):
        return (
            "\n".join(["#!/bin/bash", "set -xo pipefail"] + self.eval_script_list)
            + "\n"
        )
        # Don't exit early because we need to revert tests at the end

    @property
    def install_repo_script(self):
        return (
            "\n".join(["#!/bin/bash", "set -exo pipefail"] + self.repo_script_list)
            + "\n"
        )

    @property
    def ast_meaningful_script(self):
        return (
            "\n".join(
                ["#!/bin/bash", "set -xo pipefail"] + self.meaningful_edit_script_list
            )
            + "\n"
        )

    @property
    def coverage_script(self):
        return (
            "\n".join(["#!/bin/bash", "set -xo pipefail"] + self.coverage_script_list)
            + "\n"
        )

    @property
    def performance_script(self):
        return (
            "\n".join(
                ["#!/bin/bash", "set -exo pipefail"] + self.performance_script_list
            )
            + "\n"
        )

    @property
    def performance_profiling_script(self):
        return (
            "\n".join(
                ["#!/bin/bash", "set -exo pipefail"]
                + self.performance_profiling_script_list
            )
            + "\n"
        )

    @property
    def correctness_script(self):
        # TODO: Check whether we should add -e here?
        return (
            "\n".join(
                ["#!/bin/bash", "set -xo pipefail"] + self.correctness_script_list
            )
            + "\n"
        )

    @property
    def introspection_guard_script(self):
        return (
            "\n".join(
                ["#!/bin/bash", "set -exo pipefail"]
                + self.introspection_guard_script_list
            )
            + "\n"
        )

    @property
    def base_image_key(self):
        return f"sweb.base.{self.arch}:latest"

    @property
    def env_image_key(self):
        """
        The key for the environment image is based on the hash of the environment script list.
        If the environment script list changes, the image will be rebuilt automatically.

        Note that old images are not automatically deleted, so consider cleaning up old images periodically.
        """
        hash_object = hashlib.sha256()
        hash_object.update(str(self.env_script_list).encode("utf-8"))
        hash_value = hash_object.hexdigest()
        val = hash_value[:22]  # 22 characters is still very likely to be unique
        return f"sweb.env.{self.arch}.{val}:latest"

    @property
    def instance_image_key(self):
        return f"sweb.eval.{self.arch}.{self.instance_id}:latest"

    @property
    def annotate_instance_image_key(self):
        return f"sweb.eval.{self.arch}.{self.instance_id}.annotate:latest"

    def get_instance_container_name(self, run_id=None):
        if not run_id:
            return f"sweb.eval.{self.instance_id}"
        return f"sweb.eval.{self.instance_id}.{run_id}"

    @property
    def base_dockerfile(self):
        return get_dockerfile_base(self.platform, self.arch)

    @property
    def env_dockerfile(self):
        return get_dockerfile_env(self.platform, self.arch)

    @property
    def instance_dockerfile(self):
        return get_dockerfile_instance(self.platform, self.env_image_key)

    @property
    def annotate_instance_dockerfile(self):
        return get_dockerfile_annotate_instance(self.platform, self.instance_image_key)

    @property
    def platform(self):
        if self.arch == "x86_64":
            return "linux/x86_64"
        elif self.arch == "arm64":
            return "linux/arm64/v8"
        else:
            raise ValueError(f"Invalid architecture: {self.arch}")


def get_test_specs_from_dataset(
    dataset: Union[list[SWEfficiencyInstance], list[TestSpec]],
) -> list[TestSpec]:
    """
    Idempotent function that converts a list of SWEfficiencyInstance objects to a list of TestSpec objects.
    """
    if isinstance(dataset[0], TestSpec):
        return cast(list[TestSpec], dataset)

    test_specs = []
    for instance in dataset:
        try:
            test_specs.append(make_test_spec(instance))
        except NotImplementedError as e:
            continue
        except Exception as e:
            print(
                f"Error creating test spec for instance {instance[KEY_INSTANCE_ID]} for version {instance['version']}: {e}"
            )
            traceback.print_exc()

    return test_specs


def make_repo_script_list(
    specs, repo, repo_directory, base_commit, env_name, treesitter_env_name=None
):
    """
    Create a list of bash commands to set up the repository for testing.
    This is the setup script for the instance image.
    """
    setup_commands = [
        f"git clone -o origin https://github.com/{repo} {repo_directory}",
        f"chmod -R 777 {repo_directory}",  # So nonroot user can run tests
        f"cd {repo_directory}",
        f"git reset --hard {base_commit}",
        # Remove the remote so the agent won't see newer commits.
        "git remote remove origin",
        "git tag -d $(git tag -l)",
        "git reflog expire --expire=now --all",
        "git gc --prune=now --aggressive",
        # Verify future logs aren't available
        f"TARGET_TIMESTAMP=$(git show -s --format=%ci {base_commit})",
        "AFTER_TIMESTAMP=$(date -d \"$TARGET_TIMESTAMP + 1 second\" '+%Y-%m-%d %H:%M:%S')",
        'COMMIT_COUNT=$(git log --oneline --all --since="$AFTER_TIMESTAMP" | wc -l)',
        '[ "$COMMIT_COUNT" -eq 0 ] || exit 1',
        # Make sure conda is available for later use
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        'echo "Current environment: $CONDA_DEFAULT_ENV"',
    ]

    if repo in MAP_REPO_TO_INSTALL:
        setup_commands.append(MAP_REPO_TO_INSTALL[repo])

    # Run pre-install set up if provided
    if "pre_install" in specs:
        assert isinstance(
            specs["pre_install"], list
        ), "pre_install should be a list of commands"
        for pre_install in specs["pre_install"]:
            setup_commands.append(pre_install)

    if "install" in specs:
        setup_commands.append(specs["install"])

    # SWE-fficiency: Add coverage, pytest-profiling, pytest-memray.
    setup_commands.append("python -m pip install coverage unidiff")

    # TODO: Add pytest-profiling, pytest-memray later.
    # Note: only sklearn requires interval tree, so we don't install it by default.
    setup_commands.append("python -m pip install jedi asttokens")

    # Tree-sitter is backwards compatible,
    if treesitter_env_name:
        # TODO: Hack to get treesitter working, just make a seperate env with a diff python version (3.6).
        setup_commands.extend(
            [
                f"conda activate {treesitter_env_name}",
                "python -m pip install tree-sitter tree_sitter_languages",
                f"conda activate {env_name}",
            ]
        )

    # There may be untracked changes, just try to commit all of them (ok if it fails).
    setup_commands.append(
        "git -c user.name='Automated Test' -c user.email='automated@test.com' commit -a -m 'Fix environment' || true"
    )
    return setup_commands


def replace_uninstallable_packages_requirements_txt(requirement_str: str) -> str:
    """Replaces certain packages in a requirements.txt-like string.
    For example, some packages have been yanked and we need to replace them with compatible alternatives.
    """
    replacements = {
        # See https://github.com/princeton-nlp/SWE-bench/issues/199
        # This package was sinced yanked, so we need to force pip
        # to install it.
        # "types-pkg_resources": "types-pkg-resources==0.1.3",
    }
    requirements = [req.strip() for req in requirement_str.split("\n") if req.strip()]
    requirements_replaced = []
    for requirement in requirements:
        if requirement in replacements:
            print(
                f"Replaced {requirement!r} with {replacements[requirement]!r} (replace_uninstallable_packages)"
            )
            requirements_replaced.append(replacements[requirement])
        else:
            requirements_replaced.append(requirement)
    return "\n".join(requirements_replaced) + "\n"


def make_env_script_list(
    instance: SWEfficiencyInstance, specs: dict, env_name: str, treesitter_env_name=None
) -> list[str]:
    """
    Creates the list of commands to set up the conda environment for testing.
    This is the setup script for the environment image.

    Returns:
        list[str]: List of commands to set up the conda environment
    """
    HEREDOC_DELIMITER = "EOF_59812759871"
    reqs_commands = [
        "source /opt/miniconda3/bin/activate",
    ]

    reqs_commands.extend(
        [
            # TODO: For safety, remove defaults?
            # "conda config --remove channels defaults",
            # "conda config --add channels conda-forge",
            # "conda config --set channel_priority strict",
            # "conda install -n base conda-libmamba-solver",
            # "conda config --set solver libmamba",
            # "conda config --set safety_checks disabled",
        ]
    )

    env_command = "conda"  # "conda"

    # Create conda environment according to install instructinos
    pkgs = specs.get("packages", "")
    if pkgs == "requirements.txt":
        # Create environment
        cmd = f"{env_command} create -n {env_name} python={specs['python']} -y"
        reqs_commands.append(cmd)

        # Install dependencies
        reqs = replace_uninstallable_packages_requirements_txt(
            get_requirements(instance)
        )
        path_to_reqs = f"$HOME/{pkgs}"
        reqs_commands.append(
            f"cat <<'{HEREDOC_DELIMITER}' > {path_to_reqs}\n{reqs}\n{HEREDOC_DELIMITER}"
        )
        if "env_patches" in specs:
            reqs_commands += specs["env_patches"]
        cmd = f"conda activate {env_name} && python -m pip install -r {path_to_reqs}"
        reqs_commands.append(cmd)
        reqs_commands.append(f"rm {path_to_reqs}")
    elif pkgs == "environment.yml":
        # Create environment from yml
        reqs = get_environment_yml(instance, env_name)
        path_to_reqs = "environment.yml"

        if "- defaults" not in reqs:
            reqs_commands.append("conda config --remove channels defaults")

        reqs_commands.append(
            f"cat <<'{HEREDOC_DELIMITER}' > {path_to_reqs}\n{reqs}\n{HEREDOC_DELIMITER}"
        )
        if "env_patches" in specs:
            reqs_commands += specs["env_patches"]
        if "no_use_env" in specs and specs["no_use_env"]:
            # `conda create` based installation
            cmd = f"{env_command} create -c conda-forge -n {env_name} python={specs['python']} -y"
            reqs_commands.append(cmd)

            # Install dependencies
            cmd = f"{env_command} env update -f {path_to_reqs}"
            reqs_commands.append(cmd)
        else:
            # `conda env create` based installation
            cmd = f"{env_command} env create -n {env_name} --file {path_to_reqs} -v"
            reqs_commands.append(cmd)

            if "python" in specs:
                cmd = f"conda activate {env_name} && conda install python={specs['python']} -y"
            else:
                cmd = f"conda activate {env_name}"
            reqs_commands.append(cmd)

        # Remove environment.yml
        reqs_commands.append(f"rm {path_to_reqs}")
    else:
        # Create environment + install dependencies
        if "env_patches" in specs:
            reqs_commands += specs["env_patches"]
        cmd = (
            f"{env_command} create -n {env_name} python={specs['python']} {pkgs} -y -v"
        )
        reqs_commands.append(cmd)

    reqs_commands.append("conda clean --all -y")  # Clean up conda cache to save space
    reqs_commands.append(f"conda activate {env_name}")

    # Install additional packages if specified
    if "pip_packages" in specs:
        pip_packages = " ".join(specs["pip_packages"])
        cmd = f"python -m pip install {pip_packages}"
        reqs_commands.append(cmd)

    if treesitter_env_name:
        # TODO: Just hardcode treesitter python version.
        reqs_commands.append(
            f"{env_command} create -n {treesitter_env_name} python=3.6 -y"
        )

    # Some additional cleaning in case.
    # reqs_commands.append("pip cache purge")
    return reqs_commands


def make_test_command(instance, without_directives=False, prefer_distributed=False):
    if instance["repo"] == "python/mypy":
        pattern = r"\[case ([^\]]+)\]"
        test_keys = re.findall(pattern, instance["test_patch"])
        test_keys_or = " or ".join(test_keys)
        test_command = (
            MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]["test_cmd"]
            + " "
            + f'"{test_keys_or}"'
        )
        return test_command
    else:
        test_directives = [] if without_directives else get_test_directives(instance)
        specs = MAP_REPO_VERSION_TO_SPECS[instance["repo"].lower()][instance["version"]]

        # NOTE: Prefer distributed test command in the eval only setting, since tests can take a while.
        test_cmd = (
            specs.get("distributed_test_cmd", specs["test_cmd"])
            if prefer_distributed
            else specs["test_cmd"]
        )
        test_command = " ".join(
            [
                test_cmd,
                *test_directives,
            ]
        )
        return test_command


def _get_basic_eval_components(
    instance, specs, env_name, repo_directory, base_commit
) -> list[str]:
    eval_commands = [
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
    ]
    if "eval_commands" in specs:
        eval_commands += specs["eval_commands"]

    eval_commands += [
        f"git config --global --add safe.directory {repo_directory}",  # for nonroot user
        f"cd {repo_directory}",
        # This is just informational, so we have a record
        "git status",
        "git show",
    ]
    if "install" in specs:
        eval_commands.append(specs["install"])

    return eval_commands


def make_eval_script_list(
    instance, specs, env_name, repo_directory, base_commit, test_patch
):
    """
    Assume that patch is already applied.
    """
    HEREDOC_DELIMITER = "EOF_114329324912"
    test_files = re.findall(DIFF_MODIFIED_FILE_REGEX, test_patch)
    # Reset test files to the state they should be in before the patch.
    reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}"
    apply_test_patch_command = (
        f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
    )

    test_command = make_test_command(
        instance, without_directives=True, prefer_distributed=True
    )
    eval_commands = _get_basic_eval_components(
        instance, specs, env_name, repo_directory, base_commit
    )
    eval_commands += [
        reset_tests_command,
        apply_test_patch_command,  # This should be empty, but let's just keep this.
    ]

    if "covering_tests" in instance:
        # For actual dataset, we know covering tests, so pull this from the instance.
        eval_commands += [test_command + " " + " ".join(instance["covering_tests"])]
    else:
        # This should probably never run on actual eval.
        eval_commands += [make_test_command(instance)]

    eval_commands += [
        reset_tests_command,  # Revert tests after done, leave the repo in the same state as before
    ]
    return eval_commands


### BEGIN SWE-FFICIENCY EDITS

AST_START_TAG = "SWEPERF_AST_START"
AST_END_TAG = "SWEPERF_AST_END"


def make_ast_copy_preedit_commands(
    source_files: list[str], preedit_base_dir="/tmp/preedit/"
) -> list[str]:
    """Copy preedit state to a temp directory."""
    commands = [f"mkdir -p {preedit_base_dir}"]

    for file in source_files:
        # Assume file is in relative path form already.
        target_dir = os.path.join(preedit_base_dir, os.path.dirname(file))
        target_path = os.path.join(preedit_base_dir, file)

        commands.append(f"mkdir -p {target_dir}")
        commands.append(f"cp {file} {target_path}")

    return commands


ACCEPTABLE_EXTENIONS = [
    ".py",
    ".pyx",
    ".pyx.tp",
    ".pyi",
    ".pxi",
    ".pxd",
    ".pxd.tp",
]


def make_ast_run_tree_comparison_commands(
    source_files: list[str],
    preedit_base_dir="/tmp/preedit/",
    treesitter_file="/tresitter_compare.py",
) -> str:
    """For each source file, check if it exists in the `preedit_base_dir` using bash commands:
    if it exists, run the `treesitter_file`.
    """
    commands = []

    for file in source_files:
        # Construct the relative path in the preedit directory
        preedit_file = os.path.join(preedit_base_dir, file)
        postedit_file = file

        # Ignore anything that doesn't
        if all(ext not in file for ext in ACCEPTABLE_EXTENIONS):
            continue

        elif not file.endswith(".py"):
            commands.append(f"echo 'Warning: {file} is a non Python change.'")
            continue

        # Create the bash command to check if the file exists
        check_exists_command = f"if [ -f {postedit_file} ]; then python {treesitter_file} --preedit_file {preedit_file} --postedit_file {postedit_file}; else echo 'Warning: {file} not found in preedit directory'; fi"

        # Add the check-existence and run-comparison command to the list
        commands.append(check_exists_command)

    return commands


def make_meaningful_edit_script_list(
    instance, specs, env_name, repo_directory, base_commit, test_patch
):
    """
    This script list checks that the AST for pre and post edit files (ignoring comments) are different.
    We assume (safely) that performance related edits must be different.

    We also assume that patch (on source files is already)
    """

    HEREDOC_DELIMITER = "EOF_114329324912"
    test_files = re.findall(DIFF_MODIFIED_FILE_REGEX, test_patch)
    source_files = re.findall(DIFF_MODIFIED_FILE_REGEX, instance["patch"])
    all_files = test_files + source_files
    # Reset entire repo state
    reset_command = f"git checkout {base_commit} {' '.join(all_files)}"

    filtered_source_files = [
        file
        for file in source_files
        if not perf_utils.is_doc_file(file)
        and not perf_utils.has_lock_file_change(file)
    ]

    # 1. Make a copy of all the source files to a temp directory.
    preedit_base_dir = "/tmp/preedit/"
    treesitter_file = "/tmp/treesitter_compare.py"

    copy_preedit_commands = make_ast_copy_preedit_commands(
        filtered_source_files, preedit_base_dir=preedit_base_dir
    )

    # 2. Apply patch diff (assuming it is copied at "/tmp/patch.diff"). Old versions don't have --allow-empty.
    apply_patch_command = "git apply -v /tmp/patch.diff"
    run_tree_comparison_commands = make_ast_run_tree_comparison_commands(
        filtered_source_files,
        preedit_base_dir=preedit_base_dir,
        treesitter_file=treesitter_file,
    )

    ast_commands = [
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
        f"git config --global --add safe.directory {repo_directory}",  # for nonroot user
        f"cd {repo_directory}",
        # This is just informational, so we have a record
        "git status",
        "git show",
        f"git diff {base_commit}",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        reset_command,
    ]

    ast_commands.extend(copy_preedit_commands)
    ast_commands.append(apply_patch_command)
    ast_commands.append(f"echo {AST_START_TAG}")  # For easier parsing later.
    ast_commands.extend(run_tree_comparison_commands)

    ast_commands += [f"echo {AST_END_TAG}", reset_command]  # For easier parsing later.
    return ast_commands


def check_ast_result(raw_output):
    """Checks that AST parser says edit is reasonably meaningful."""
    relevant_lines = raw_output.splitlines()

    # Remove lines that start with "+"
    std_out_lines = [line for line in relevant_lines if not line.startswith("+")]

    # Get all the elements between "AST_START_TAG" and "AST_END_TAG".
    start_tag_index = std_out_lines.index(AST_START_TAG)
    end_tag_index = std_out_lines.index(AST_END_TAG)
    relevant_lines = std_out_lines[start_tag_index + 1 : end_tag_index]

    content = "\n".join(relevant_lines)

    warning_count = content.count("Warning")
    non_meaningful_count = content.count("NOOP")
    meaningful_count = content.count("MEANINGFUL")

    # Meaningful edits are Python edits that change the AST.
    has_meaningful = meaningful_count > 0

    # Warning just means that there are edits that we prob don't know.
    has_warning = warning_count > 0

    return has_meaningful or has_warning


COVERAGE_START_TAG = "SWEPERF_COVERAGE_START"
COVERAGE_END_TAG = "SWEPERF_COVERAGE_END"
PER_TEST_TAG = "SWEPERF_COVERAGE_REPORT"

COARSE_COVERAGE_AST_SCRIPT_LOCATION = "/tmp/coverage_ast.py"

CYTHON_COVERAGE_RC_FILE = """\
[run]
plugins = Cython.Coverage
"""
COVERAGE_RC_FILE_MAPPING = {
    "scikit-learn/scikit-learn": CYTHON_COVERAGE_RC_FILE,
    "pandas-dev/pandas": CYTHON_COVERAGE_RC_FILE,
    "numpy/numpy": CYTHON_COVERAGE_RC_FILE,
    "scipy/scipy": CYTHON_COVERAGE_RC_FILE,
    "explosion/spacy": CYTHON_COVERAGE_RC_FILE,
}

DEFAULT_COVERAGE_DATA_DIR = "/tmp/coverage_data/"
RAW_COVERAGE_OUTPUT_DIR = "/tmp/raw_coverage_data/"
DEFAULT_OUTFILE = "/tmp/all_tests.txt"

# Sklearn and dask are quite slow so use coarse AST methods.
REPOS_TO_IGNORE_COARSE_AST_COVERAGE = {
    "pandas-dev/pandas",  # Dependencies hard to track, lets just run full repo level coverage
    # "django/django", # Django"
    "numpy/numpy",  # Numpy tests run fast anyway.
    "pydata/xarray",  # Runs fast anyway.
    "matplotlib/matplotlib",  # Runs fast as well anyway.
    "explosion/spacy",
    "modin-project/modin",
    "pandas-dev/pandas",
}

# This identifies any tests in parent directories: many data science libraries obey this type of ruling.
identify_parent_dir_tests = 'while read file; do dir=$(dirname "$file"); prev=""; while [ "$dir" != "$prev" ]; do testdir="$dir/tests"; [ -d "$testdir" ] && find "$testdir" -type f; prev="$dir"; dir=$(dirname "$dir"); done; done | sort -u'

REPOS_TO_SPECIAL_TEST_COVERAGE = {
    "scipy/scipy": "echo '{source_files}' | tr ',' '\\n'"
    + " | "
    + identify_parent_dir_tests
    + " | grep '\\.py$'",
    "sympy/sympy": "echo '{source_files}' | tr ',' '\\n'"
    + " | "
    + identify_parent_dir_tests
    + " | grep '\\.py$'",
    "astropy/astropy": "echo '{source_files}' | tr ',' '\\n'"
    + " | "
    + identify_parent_dir_tests
    + " | grep '\\.py$'",
    "dask/dask": "echo '{source_files}' | tr ',' '\\n'"
    + " | "
    + identify_parent_dir_tests
    + " | grep '\\.py$'",
    "scikit-learn/scikit-learn": "echo '{source_files}' | tr ',' '\\n'"
    + " | "
    + identify_parent_dir_tests
    + " | grep '\\.py$'",
}

REPOS_TO_TEST_SEARCH_DEPTH = {
    # "pandas-dev/pandas": 1,
    # "modin-project/modin": 1,
    "scikit-learn/scikit-learn": 3,  # Sklearn tests are expensive
}

check_exit_code_pytest = 'status=$?; if [[ $status -ne 0 && $status -ne 1 ]]; then echo "Workload failed with exit code $status" >&2; exit $status; fi'


def make_coverage_commands_old(
    instance,
    repo_directory,
    source_files: list[str],
    parallel_coverage_workers: int = 4,
) -> list[str]:
    """
    This command enumerates all test files, and runs pytest coverage over all tests, printing the coverage report afterwards.
    """

    # TODO: Coverage run does not play well with pytest distributed, so we run it in a single process.
    test_cmd = make_test_command(
        instance, without_directives=True, prefer_distributed=False
    )
    testbed_source_files = [f"/testbed/{f}" for f in source_files]

    env_portion = ""

    if "pytest" in test_cmd or "tox" in test_cmd:
        module_cmd = f" --module {test_cmd}"
    else:
        if instance["repo"] == "sympy/sympy":
            env_portion = test_cmd.split("bin/test")[0].strip()
            module_cmd = test_cmd[len(env_portion) :]
        else:
            # Assume that we're just running a Python entrypoint.
            module_cmd = f" {test_cmd}"

    # First step is coarse pruning, using AST to identify collisions between test files and files edited in diff.
    # find_and_get_tests_as_single_line = f"find {repo_directory} -type f -name \"test_*.py\" -o -name \"*_test.py\" | xargs readlink -f | tr '\\n' ',' | sed '$s/,$//'"
    if instance["repo"] in REPOS_TO_IGNORE_COARSE_AST_COVERAGE:
        find_test_cmd = f'find {repo_directory} -type f -name "*test*.py"'
        run_ast_coverage_cmd_and_get_tests = find_test_cmd + f" > {DEFAULT_OUTFILE}"

    elif instance["repo"] in REPOS_TO_SPECIAL_TEST_COVERAGE:
        # assert False
        try:
            template = REPOS_TO_SPECIAL_TEST_COVERAGE[instance["repo"]]
            source_files_commas = ",".join(source_files)

            run_ast_coverage_cmd_and_get_tests = (
                template.format(source_files=source_files_commas)
                + f" > {DEFAULT_OUTFILE}"
            )
        except Exception as e:
            print(e)
            traceback.print_exc()

    else:
        test_recursion_depth = REPOS_TO_TEST_SEARCH_DEPTH.get(instance["repo"], 3)
        run_ast_coverage_cmd_and_get_tests = f"python {COARSE_COVERAGE_AST_SCRIPT_LOCATION} --source_files {','.join(testbed_source_files)} --outfile {DEFAULT_OUTFILE} --ignore_depth {test_recursion_depth} 2> /dev/null || exit 1"

    coverage_file_content = COVERAGE_RC_FILE_MAPPING.get(instance["repo"])
    coverage_write_script = ""
    coverage_rc_file_addition = ""
    if coverage_file_content:
        coverage_file = "/tmp/.coveragerc"
        coverage_write_script = f"""/bin/bash -c \'mkdir -p "$(dirname {coverage_file})" && cat <<EOF > "{coverage_file}"
{coverage_file_content}
EOF\'"""
        coverage_rc_file_addition = f"COVERAGE_RCFILE={coverage_file} "

    # remove_existing_coveragerc = "rm /testbed/.coveragerc"

    test_directive_symbol = "$line"
    if instance["repo"] == "django/django":
        # Django requires as modeule.
        test_directive_symbol = (
            '$(echo $line | sed "s|.py||; s|/testbed/tests/||; s|/|.|g")'
        )
    elif instance["repo"] == "scipy/scipy" and instance["version"] in [
        "0.16",
        "0.17",
        "0.18",
        "0.19",
    ]:
        # Scipy early versions require you to provide as module file.
        test_directive_symbol = (
            '$(echo $line | sed "s|.py||; s|/testbed/tests/||; s|/|.|g")'
        )

    # Next, up for matched tests, we then feed this into actual pytest with coverage to verify that the diff intersects with the actual edit.

    # coverage_command = (
    #     f"cat {DEFAULT_OUTFILE} |"
    #     f"xargs -n 1 -P {parallel_coverage_workers} -d\\\\n --verbose --no-run-if-empty -I % "
    #     f"/bin/bash -c 'tmpfile=$(mktemp); echo \"Processing: %\" >&2; {coverage_rc_file_addition} "
    #     f"COVERAGE_FILE={DEFAULT_COVERAGE_DATA_DIR}$(echo -n \"%\" | sha256sum | cut -d\" \" -f1) {env_portion} "
    #     f"coverage run -p --include={','.join(testbed_source_files)} --context=% {module_cmd} {test_directive_symbol}"
    # )

    # f"coverage run -p --include={','.join(testbed_source_files)} --context=\"$line\" {module_cmd} {test_directive_symbol} | tee \"$outfile\" > /dev/null'"
    coverage_command = (
        f"cat {DEFAULT_OUTFILE} | "
        f"shuf | "  # Shuffle to avoid bias.
        "awk -F/ '{if (tolower($NF) ~ /^(test.*\\.py|.*_test\\.py)$/) print $0}' | "
        f"awk '{{ print NR-1 \" \" $0 }}' | "
        f"xargs -P {parallel_coverage_workers} -d\\\\n --verbose --no-run-if-empty -I % "
        f'/bin/bash -c \'idx=$(echo % | cut -d" " -f1); '
        f'line=$(echo % | cut -d" " -f2-); '
        f'outfile="{RAW_COVERAGE_OUTPUT_DIR}$idx.txt"; '
        f"mkdir -p {RAW_COVERAGE_OUTPUT_DIR}; "
        f"currenttime=$(date); "
        f'echo "[$currenttime] Processing [$idx]: $line" >&2; '
        f"{coverage_rc_file_addition} "
        f'COVERAGE_FILE={DEFAULT_COVERAGE_DATA_DIR}$(echo -n "$line" | sha256sum | cut -d" " -f1) {env_portion} '
        f"coverage run -p --include={','.join(testbed_source_files)} --context=\"$line\" {module_cmd} {test_directive_symbol} | tee \"$outfile\" > /dev/null; "
        f'echo "[$currenttime] Finished processing [$idx]: $line" >&2;\''
    )

    # Print all outputs, then delete them.
    # print_output_command = f"cat /tmp/coverage_raw_output/*.txt"

    echo_nproc = 'echo "Number of parallel workers: $(nproc)"'

    return [
        # remove_existing_coveragerc,
        echo_nproc,
        coverage_write_script,
        run_ast_coverage_cmd_and_get_tests,
        f"cat {DEFAULT_OUTFILE}",
        coverage_command,
        # print_output_command, # This makes container take literally forever to run, so we don't do this.
    ]


def make_coverage_commands(
    instance,
    repo_directory,
    source_files: list[str],
    parallel_coverage_workers: int = 4,
) -> list[str]:
    """
    Build shell snippets that:
      • compute the candidate-test list (AST coarse coverage or repo-specific)
      • create a helper script to run one test with coverage
      • launch an xargs pipeline that feeds tests to that helper
    """

    # ------------------------------------------------------------------ basics
    test_cmd = make_test_command(
        instance, without_directives=True, prefer_distributed=False
    )
    testbed_source_files = [f"/testbed/{f}" for f in source_files]

    # --- translate test_cmd into the form "coverage run … --module pytest …"
    env_portion = ""
    if "pytest" in test_cmd or "tox" in test_cmd:
        module_cmd = f" --module {test_cmd}"
    else:
        if instance["repo"] == "sympy/sympy":
            env_portion = test_cmd.split("bin/test")[0].strip()
            module_cmd = test_cmd[len(env_portion) :]
        else:
            module_cmd = f" {test_cmd}"

    # ------------------------------------------------------------------ coarse AST pruning
    if instance["repo"] in REPOS_TO_IGNORE_COARSE_AST_COVERAGE:
        find_tests = f'find {repo_directory} -type f -name "*test*.py"'
        run_ast_coverage_cmd_and_get_tests = f"{find_tests} > {DEFAULT_OUTFILE}"
    elif instance["repo"] in REPOS_TO_SPECIAL_TEST_COVERAGE:
        tmpl = REPOS_TO_SPECIAL_TEST_COVERAGE[instance["repo"]]
        run_ast_coverage_cmd_and_get_tests = (
            tmpl.format(source_files=",".join(source_files)) + f" > {DEFAULT_OUTFILE}"
        )
    else:
        depth = REPOS_TO_TEST_SEARCH_DEPTH.get(instance["repo"], 3)
        run_ast_coverage_cmd_and_get_tests = (
            f"python {COARSE_COVERAGE_AST_SCRIPT_LOCATION} "
            f"--source_files {','.join(testbed_source_files)} "
            f"--outfile {DEFAULT_OUTFILE} --ignore_depth {depth} 2>/dev/null || exit 1"
        )

    # ------------------------------------------------------------------ repo-specific test-directive symbol
    if instance["repo"] == "django/django":
        tdir = "$(echo \"$line\" | sed 's|.py||; s|/testbed/tests/||; s|/|.|g')"
    elif instance["repo"] == "scipy/scipy" and instance["version"] in [
        "0.16",
        "0.17",
        "0.18",
        "0.19",
    ]:
        tdir = "$(echo \"$line\" | sed 's|.py||; s|/testbed/tests/||; s|/|.|g')"
    else:
        tdir = "$line"

    # ------------------------------------------------------------------ optional .coveragerc support
    coverage_file_content = COVERAGE_RC_FILE_MAPPING.get(instance["repo"])
    coverage_write_script = ""
    coverage_rc_file_addition = ""
    if coverage_file_content:
        coverage_file = "/tmp/.coveragerc"
        coverage_write_script = f"""\
/bin/bash -c 'mkdir -p "$(dirname {coverage_file})" && cat <<"EOF" > {coverage_file}
{coverage_file_content}
EOF'"""
        coverage_rc_file_addition = f"COVERAGE_RCFILE={coverage_file} "

    # ------------------------------------------------------------------ helper script (written by the driver itself)
    helper_name = "./run_one_coverage_test.sh"
    helper_body = textwrap.dedent(
        f"""\
#!/usr/bin/env bash
set -euo pipefail

input=$1
read -r idx line <<<"$input"

outfile="{RAW_COVERAGE_OUTPUT_DIR}/${{idx}}.txt"
mkdir -p "{RAW_COVERAGE_OUTPUT_DIR}"

echo "[ $(date) ] Processing [${{idx}}]: $line" >&2

{coverage_rc_file_addition} \\
COVERAGE_FILE={DEFAULT_COVERAGE_DATA_DIR}$(printf "%s" "$line" | sha256sum | cut -d" " -f1) \\
{env_portion} coverage run -p \\
    --include={','.join(testbed_source_files)} \\
    --context="$line" {module_cmd} {tdir} \\
| tee "$outfile" > /dev/null"""
    ).rstrip()

    create_helper_cmd = f"""\
/bin/bash -c 'cat <<"EOF" > {helper_name}
{helper_body}
EOF'"""
    make_helper_executable = f"chmod +x {helper_name}"

    # ------------------------------------------------------------------ xargs pipeline
    coverage_command = (
        f"cat {DEFAULT_OUTFILE} | "
        f"shuf | "  # Shuffle to avoid bias.
        "awk -F/ '{if (tolower($NF) ~ /^(test.*\\.py|.*_test\\.py)$/) print $0}' | "
        f"awk '{{ print NR-1 \" \" $0 }}' | "
        f"xargs -P {parallel_coverage_workers} -d\\\\n --verbose --no-run-if-empty -I % "
        f"{helper_name} %"
    )

    # ------------------------------------------------------------------ extras / bookkeeping
    echo_nproc = 'echo "Number of parallel workers: $(nproc)"'

    # ------------------------------------------------------------------ final command list
    return [
        echo_nproc,
        coverage_write_script,  # may be empty string
        run_ast_coverage_cmd_and_get_tests,
        f"cat {DEFAULT_OUTFILE}",  # optional diagnostic
        create_helper_cmd,
        make_helper_executable,
        f"cat {helper_name}",  # optional diagnostic
        coverage_command,
        f"rm {helper_name}",  # cleanup helper script
    ]


COVERAGE_ANALYSIS_SCRIPT_LOCATION = "/tmp/coverage_analysis.py"


def make_coverage_script_list(
    instance, specs, env_name, repo_directory, base_commit, test_patch
):
    """
    Applies the test patch and runs each test one by one to determine coverage. Assumes
    coverage is alreayd installed in the environment.
    """

    HEREDOC_DELIMITER = "EOF_114329324912"
    test_files = re.findall(DIFF_MODIFIED_FILE_REGEX, test_patch)
    # Reset test files to the state they should be in before the patch.
    reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}"
    apply_test_patch_command = (
        (f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}")
        if test_patch
        else ""
    )

    apply_patch_command = "git apply -v /tmp/patch.diff"
    reset_command = f"git checkout {base_commit}"

    source_files = re.findall(DIFF_MODIFIED_FILE_REGEX, instance["patch"])

    num_parallel_workers = 8

    REPOS_TO_SLOW_COVERAGE = {
        "scikit-learn/scikit-learn",
        # "sympy/sympy",
    }

    if instance["repo"] in REPOS_TO_SLOW_COVERAGE:
        # Tests are expensive, so we run sequentially.
        num_parallel_workers = 1

    coverage_commands = make_coverage_commands(
        instance,
        repo_directory,
        source_files,
        parallel_coverage_workers=num_parallel_workers,
    )

    # Get base components first.
    eval_commands = _get_basic_eval_components(
        instance, specs, env_name, repo_directory, base_commit
    )

    # coverage does not have a fun time when you don't run it from package root dir.
    cd_into_module_dir_cmd = "cd $(python -c 'import setuptools; packages = setuptools.find_packages(); print(next((p for p in packages if \".\" not in p), None))')"
    cd_back_to_repo = f"cd {repo_directory}"

    # Parse coverage analysis command: assumes that patch is at /tmp/patch.diff.
    parse_coverage_file_cmd = (
        f"python {COVERAGE_ANALYSIS_SCRIPT_LOCATION} --patch-file /tmp/patch.diff"
    )

    eval_commands += [
        reset_tests_command,
        apply_test_patch_command,
        apply_patch_command,
    ]

    if "install" in specs:
        eval_commands.append(specs["install"])

    eval_commands += [
        # cd_into_module_dir_cmd, # At this point, we are in the module source directory.
        f"coverage --version",
        f"mkdir -p {DEFAULT_COVERAGE_DATA_DIR}",
        *coverage_commands,
        f"echo {COVERAGE_START_TAG}",
        parse_coverage_file_cmd,
        f"echo {COVERAGE_END_TAG}",
        # cd_back_to_repo,
        reset_tests_command,  # Revert tests after done, leave the repo in the same state as before
        reset_command,
    ]
    return eval_commands


def parse_coverage_report(raw_output):
    # Coverage output is sandwitched between COVERAGE_START_TAG and COVERAGE_END_TAGS.
    raw_stdout_lines = [l for l in raw_output.splitlines() if not l.startswith("+")]
    line_start_index = raw_stdout_lines.index("BEGIN_VALID_TESTS")
    line_end_index = raw_stdout_lines.index("END_VALID_TESTS")

    return raw_stdout_lines[line_start_index + 1 : line_end_index]


PERF_WORKLOAD_SCRIPT_LOCATION = "/tmp/workload.py"
PERF_CPROFILE_OUTPUT_LOCATION = "/tmp/workload_cprofile.prof"

perf_start_tag = "PERF_START:"
perf_end_tag = "PERF_END:"


def make_performance_script_list(
    instance, specs, env_name, repo_directory, base_commit, test_patch
):
    """
    Assumes we have already copied results to the directories specified above.
    """
    # Run with a few iterations of warmup first.
    eval_commands = _get_basic_eval_components(
        instance, specs, env_name, repo_directory, base_commit
    )

    # Compute the before and after
    eval_commands.extend(
        [
            f"echo '{perf_start_tag}'",
            f"python {PERF_WORKLOAD_SCRIPT_LOCATION}",
            f"echo '{perf_end_tag}'",
        ]
    )

    return eval_commands


def make_performance_profiling_script_list(
    instance, specs, env_name, repo_directory, base_commit, test_patch
):
    """
    Assumes we have already copied results to the directories specified above.
    """
    # Run with a few iterations of warmup first.
    eval_commands = _get_basic_eval_components(
        instance, specs, env_name, repo_directory, base_commit
    )

    # Compute the before and after
    eval_commands.extend(
        [
            # Run the cProfile command to profile the workload script.
            f"python -m cProfile -o {PERF_CPROFILE_OUTPUT_LOCATION} {PERF_WORKLOAD_SCRIPT_LOCATION}",
        ]
    )

    return eval_commands


def parse_perf_output(raw_output):
    cleaned_perf_output = [l for l in raw_output.splitlines() if not l.startswith("+")]
    cleaned_per_output = "\n".join(cleaned_perf_output) + "\n"

    # Extract whats in between "START_AFTER_CHANGE:" and "END_AFTER_CHANGE:".
    start_index = cleaned_per_output.find(perf_start_tag)
    end_index = cleaned_per_output.find(perf_end_tag)

    if start_index == -1 or end_index == -1:
        raise ValueError("Perf tags not found in output.")

    # Get only text between the tags.
    perf_text = cleaned_per_output[start_index + len(perf_start_tag) : end_index]
    perf_text = cleaned_per_output

    pattern = r"(?:Mean|Std\s*Dev):\s*([\S]+)"
    matches = re.findall(pattern, perf_text)

    mean, std_dev = matches[0], matches[1]
    return float(mean), float(std_dev)


DEFAULT_COVERING_TESTS_LOCATION = "/tmp/covering_tests.txt"
DEFAULT_SINGLE_THREAD_COVERING_TESTS_LOCATION = "/tmp/single_thread_tests.txt"
DEFAULT_CORRECTNESS_TEST_OUTPUT_LOCATION = "/tmp/raw_correctness_output"


def get_correctness_script_list(
    instance,
    specs,
    env_name,
    repo_directory,
    base_commit,
    test_patch,
    parallel_workers=4,
):
    """
    Assumes we have already copied results to the directories specified above.
    """
    eval_commands = _get_basic_eval_components(
        instance, specs, env_name, repo_directory, base_commit
    )

    test_command = make_test_command(
        instance, without_directives=True, prefer_distributed=False
    )

    test_directive_symbol = "$line"
    if instance["repo"] == "django/django":
        # Django requires as modeule.
        test_directive_symbol = (
            '$(echo $line | sed "s|.py||; s|/testbed/tests/||; s|/|.|g")'
        )
    elif instance["repo"] == "scipy/scipy" and instance["version"] in [
        "0.16",
        "0.17",
        "0.18",
        "0.19",
    ]:
        # Scipy early versions require you to provide as module file.
        test_directive_symbol = (
            '$(echo $line | sed "s|.py||; s|/testbed/tests/||; s|/|.|g")'
        )

    # if instance['repo'] == "scikit-learn/scikit-learn":
    #     # Sklearn tests are quite expensive, so we run them single threaded.
    #     parallel_workers = 1

    NUM_RETRY = 2

    correctness_command = (
        f"cat {DEFAULT_COVERING_TESTS_LOCATION} | "
        f"grep -vFx -f {DEFAULT_SINGLE_THREAD_COVERING_TESTS_LOCATION} | "
        f"shuf | "
        "awk -F/ '{if (tolower($NF) ~ /^(test.*\\.py|.*_test\\.py)$/) print $0}' | "
        f"awk '{{ print NR-1 \" \" $0 }}' | "
        f"xargs -P {parallel_workers} -d\\\\n --verbose --no-run-if-empty -I % "
        f"/bin/bash -c '"
        f"set -o pipefail; "  # make PIPESTATUS meaningful
        f'idx=$(echo % | cut -d" " -f1); '
        f'line=$(echo % | cut -d" " -f2-); '
        f'outfile="{DEFAULT_CORRECTNESS_TEST_OUTPUT_LOCATION}/$idx.txt"; '
        f"mkdir -p {DEFAULT_CORRECTNESS_TEST_OUTPUT_LOCATION}; "
        f'echo "[$(date)] Processing [$idx]: $line" >&2; '
        f"retries=0; rc=0; "
        f"while true; do "
        # capture both stdout+stderr; write to file; print nothing to terminal
        f'{test_command} {test_directive_symbol} | tee "$outfile" > /dev/null; '
        f"rc=${{PIPESTATUS[0]}}; "
        f'if [ "$rc" -le 1 ]; then break; fi; '
        f'if [ "$retries" -ge {NUM_RETRY} ]; then break; fi; '
        f"retries=$((retries+1)); "
        f'echo "[$(date)] Exit $rc (>1). Retrying [$idx] attempt $retries/{NUM_RETRY}: $line" >&2; '
        f"done; "
        f'echo "[$(date)] Finished processing [$idx] with exit $rc: $line" >&2; '
        f"exit $rc;"
        f"'"
    )

    # # We need to double quote test_command since
    # correctness_command = (
    #     f"cat {DEFAULT_COVERING_TESTS_LOCATION} | "
    #     f"grep -vFx -f {DEFAULT_SINGLE_THREAD_COVERING_TESTS_LOCATION} | " # Remove single-threaded tests.
    #     f"shuf | " # Shuffle to avoid bias.
    #     "awk -F/ '{if (tolower($NF) ~ /^(test.*\\.py|.*_test\\.py)$/) print $0}' | "
    #     f"awk '{{ print NR-1 \" \" $0 }}' | "
    #     f"xargs -P {parallel_workers} -d\\\\n --verbose --no-run-if-empty -I % "
    #     f"/bin/bash -c 'idx=$(echo % | cut -d\" \" -f1); "
    #     f"line=$(echo % | cut -d\" \" -f2-); "
    #     f"outfile=\"{DEFAULT_CORRECTNESS_TEST_OUTPUT_LOCATION}/$idx.txt\"; "
    #     f"mkdir -p {DEFAULT_CORRECTNESS_TEST_OUTPUT_LOCATION}; "
    #     f"currenttime=$(date); "
    #     f"echo \"[$currenttime] Processing [$idx]: $line\" >&2; "
    #     f"{test_command} {test_directive_symbol} | tee \"$outfile\" > /dev/null; "
    #     f"echo \"[$currenttime] Finished processing [$idx]: $line\" >&2;'"
    # )

    single_threaded_correctness_cmd = (
        f"if [[ -s {DEFAULT_SINGLE_THREAD_COVERING_TESTS_LOCATION} ]]; then "
        f"cat {DEFAULT_SINGLE_THREAD_COVERING_TESTS_LOCATION} | "
        f"shuf | "  # Shuffle to avoid bias.
        "awk -F/ '{if (tolower($NF) ~ /^(test.*\\.py|.*_test\\.py)$/) print $0}' | "
        f"awk '{{ print NR-1 \" \" $0 }}' | "
        f"xargs -P 1 -d\\\\n --verbose --no-run-if-empty -I % "
        f"/bin/bash -c '"
        f'idx=$(echo % | cut -d" " -f1); '
        f"NUM_PARALLEL_TESTS=$(cat {DEFAULT_COVERING_TESTS_LOCATION} | grep -vFx -f {DEFAULT_SINGLE_THREAD_COVERING_TESTS_LOCATION} | wc -l); "
        f'line=$(echo % | cut -d" " -f2-); '
        f'outfile="{DEFAULT_CORRECTNESS_TEST_OUTPUT_LOCATION}/$((idx + NUM_PARALLEL_TESTS)).txt"; '
        f"mkdir -p {DEFAULT_CORRECTNESS_TEST_OUTPUT_LOCATION}; "
        f'echo "[$(date)] Processing [$((idx + NUM_PARALLEL_TESTS))]: $line" >&2; '
        f"retries=0; rc=0; "
        f"while true; do "
        f'PYTHONHASHSEED=0 {test_command} {test_directive_symbol} | tee "$outfile" > /dev/null; '
        f"rc=${{PIPESTATUS[0]}}; "
        f'if [ "$rc" -le 1 ]; then break; fi; '
        f'if [ "$retries" -ge {NUM_RETRY} ]; then break; fi; '
        f"retries=$((retries+1)); "
        f'echo "[$(date)] Exit $rc (>1). Retrying [$((idx + NUM_PARALLEL_TESTS))] attempt $retries/{NUM_RETRY}: $line" >&2; '
        f"done; "
        f'echo "[$(date)] Finished processing [$((idx + NUM_PARALLEL_TESTS))] with exit $rc: $line" >&2; '
        f"exit $rc;' "
        f"else echo 'Input file is empty, skipping.' >&2; fi"
    )

    # single_threaded_correctness_cmd = (
    #     f"if [[ -s {DEFAULT_SINGLE_THREAD_COVERING_TESTS_LOCATION} ]]; then "
    #     f"cat {DEFAULT_SINGLE_THREAD_COVERING_TESTS_LOCATION} | "
    #     f"shuf | " # Shuffle to avoid bias.
    #     "awk -F/ '{if (tolower($NF) ~ /^(test.*\\.py|.*_test\\.py)$/) print $0}' | "
    #     f"awk '{{ print NR-1 \" \" $0 }}' | "
    #     f"xargs -P 1 -d\\\\n --verbose --no-run-if-empty -I % "
    #     f"/bin/bash -c 'idx=$(echo % | cut -d\" \" -f1); "
    #     f"NUM_PARALLEL_TESTS=$(cat {DEFAULT_COVERING_TESTS_LOCATION} | grep -vFx -f {DEFAULT_SINGLE_THREAD_COVERING_TESTS_LOCATION} | wc -l); "
    #     f"line=$(echo % | cut -d\" \" -f2-); "
    #     f"outfile=\"{DEFAULT_CORRECTNESS_TEST_OUTPUT_LOCATION}/$((idx + NUM_PARALLEL_TESTS)).txt\"; "
    #     f"mkdir -p {DEFAULT_CORRECTNESS_TEST_OUTPUT_LOCATION}; "
    #     f"currenttime=$(date); "
    #     f"echo \"[$currenttime] Processing [$((idx + NUM_PARALLEL_TESTS))]: $line\" >&2; "
    #     f"{test_command} {test_directive_symbol} | tee \"$outfile\" > /dev/null'; "
    #     f"else echo 'Input file is empty, skipping.' >&2; fi"
    # )

    eval_commands.extend([correctness_command, single_threaded_correctness_cmd])

    return eval_commands


INTROSPECTION_GUARD_CMD_LOCATION = "/tmp/introspection_patch_check.py"


def get_introspection_guard_cmds(
    instance, specs, env_name, repo_directory, base_commit, test_patch
):
    """
    Get commands to run introspection guard. Assumes that patch is already applied.
    """
    eval_commands = _get_basic_eval_components(
        instance, specs, env_name, repo_directory, base_commit
    )
    check_introspection_cmd = (
        f"python {INTROSPECTION_GUARD_CMD_LOCATION} --patch-file /tmp/patch.diff"
    )
    eval_commands.append(check_introspection_cmd)

    return eval_commands


### END SWE-FFICIENCY EDITS

BUILD_TIMEOUT_CONFIGS_OVERRIDES = {
    "scikit-learn/scikit-learn": 600,
    "pandas-dev/pandas": None,
    "pydata/xarray": 3600,
}


def make_test_spec(instance: SWEfficiencyInstance, observed_versions=None) -> TestSpec:
    if isinstance(instance, TestSpec):
        return instance
    instance_id = instance[KEY_INSTANCE_ID]
    # if there's capital letters in the repo name, convert to lowercase
    if instance_id != instance_id.lower():
        print(
            f"Instance ID {instance_id} contains capital letters. Converting to lowercase."
        )
        instance_id = instance_id.lower()
    repo = instance["repo"].lower()
    version = instance["version"]
    base_commit = instance["base_commit"]
    problem_statement = instance["problem_statement"]
    hints_text = instance["hints_text"]  # Unused
    test_patch = instance["test_patch"]

    build_timeout = BUILD_TIMEOUT_CONFIGS_OVERRIDES.get(repo)

    def _from_json_or_obj(key: str) -> Any:
        """If key points to string, load with json"""
        if isinstance(instance[key], str):
            return json.loads(instance[key])
        return instance[key]

    try:
        pass_to_pass = _from_json_or_obj(PASS_TO_PASS)
    except Exception as e:
        # print(f"Error parsing PASS_TO_PASS for instance {instance_id}: {e}. PASS_TO_PASS: {instance[PASS_TO_PASS]}")
        pass_to_pass = []

    try:
        fail_to_pass = _from_json_or_obj(FAIL_TO_PASS)
    except Exception as e:
        # print(f"Error parsing FAIL_TO_PASS for instance {instance_id}: {e}. FAIL_TO_PASS: {instance[FAIL_TO_PASS]}")
        fail_to_pass = []

    try:
        covering_tests = _from_json_or_obj("covering_tests")
    except Exception as e:
        # print(f"Error parsing covering_tests for instance {instance_id}: {e}. covering_tests: {instance['covering_tests']}")
        covering_tests = []

    env_name = "testbed"
    treesitter_env_name = "treesitter"
    repo_directory = f"/{env_name}"

    if version not in MAP_REPO_VERSION_TO_SPECS[repo]:
        raise NotImplementedError(f"Version {version} not implemented for repo {repo}")

    if observed_versions is not None:
        if version in observed_versions:
            raise RuntimeError(f"Version has already been observed: {version}")
        observed_versions.add(version)

    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]

    repo_script_list = make_repo_script_list(
        specs,
        repo,
        repo_directory,
        base_commit,
        env_name,
        treesitter_env_name=treesitter_env_name,
    )
    env_script_list = make_env_script_list(
        instance, specs, env_name, treesitter_env_name=treesitter_env_name
    )
    eval_script_list = make_eval_script_list(
        instance, specs, env_name, repo_directory, base_commit, test_patch
    )
    coverage_script_list = make_coverage_script_list(
        instance, specs, env_name, repo_directory, base_commit, test_patch
    )
    meaningful_edit_script_list = make_meaningful_edit_script_list(
        instance, specs, treesitter_env_name, repo_directory, base_commit, test_patch
    )

    # Note that this needs to be a function
    performance_script_list = make_performance_script_list(
        instance, specs, env_name, repo_directory, base_commit, test_patch
    )
    correctness_script_list = get_correctness_script_list(
        instance, specs, env_name, repo_directory, base_commit, test_patch
    )

    if platform.machine() in {"aarch64", "arm64"}:
        # use arm64 unless explicitly specified
        arch = "arm64" if instance_id not in USE_X86 else "x86_64"
    else:
        arch = "x86_64"

    workload_text = instance.get("workload", "")
    if workload_text.strip() == "nan" or workload_text.strip() == "":
        workload_text = None

    return TestSpec(
        instance_id=instance_id,
        repo=repo,
        env_script_list=env_script_list,
        repo_script_list=repo_script_list,
        eval_script_list=eval_script_list,
        version=version,
        arch=arch,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        # SWE-fficiency changes.
        coverage_script_list=coverage_script_list,
        meaningful_edit_script_list=meaningful_edit_script_list,
        workload=workload_text,
        covering_tests=covering_tests,
        performance_script_list=performance_script_list,
        performance_profiling_script_list=make_performance_profiling_script_list(
            instance, specs, env_name, repo_directory, base_commit, test_patch
        ),
        correctness_script_list=correctness_script_list,
        introspection_guard_script_list=get_introspection_guard_cmds(
            instance, specs, env_name, repo_directory, base_commit, test_patch
        ),
        build_timeout=build_timeout,
        single_thread_tests=instance.get("single_thread_tests", []),
        base_commit=base_commit,
    )
