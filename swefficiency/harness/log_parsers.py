import re
from enum import Enum

from swefficiency.harness.constants import TestStatus


def pytest_drop_trailing_info(test_name_with_trailing):
    """
    Given a pytest test name with trailing info, drop the trailing info.

    Args:
        test_name_with_trailing (str): The test name with trailing info.
    Returns:
        str: The test name without trailing info.
    """
    # Check if line has ".py::"
    if not "[" in test_name_with_trailing:
        return test_name_with_trailing.split()[0]
    else:
        # Only drop the trailing info if the prefix is unique.
        prefix = test_name_with_trailing.split("[", 1)[0]

        # For the remainder, parse until brackets are all closed.
        bracket_counter = 0
        char_index = len(prefix)

        chars = []

        for i in range(char_index, len(test_name_with_trailing)):
            char = test_name_with_trailing[i]
            chars.append(char)
            if char == "[":
                bracket_counter += 1
            elif char == "]":
                bracket_counter -= 1
                if bracket_counter == 0:
                    break

        return prefix + "".join(chars)


def parse_log_pytest(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.splitlines():
        if any([line.startswith(x.value) for x in TestStatus]):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split(maxsplit=1)
            if len(test_case) <= 1:
                continue
            test_case_name = test_case[1]

            test_case_name = pytest_drop_trailing_info(test_case_name)

            test_status_map[test_case_name] = test_case[0]
    return test_status_map


def parse_log_pytest_options(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework with options

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    option_pattern = re.compile(r"(.*?)\[(.*)\]")
    test_status_map = {}
    for line in log.split("\n"):
        if any([line.startswith(x.value) for x in TestStatus]):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            has_option = option_pattern.search(test_case[1])
            if has_option:
                main, option = has_option.groups()
                if (
                    option.startswith("/")
                    and not option.startswith("//")
                    and "*" not in option
                ):
                    option = "/" + option.split("/")[-1]
                test_name = f"{main}[{option}]"
            else:
                test_name = test_case[1]
            test_status_map[test_name] = test_case[0]
    return test_status_map


def parse_log_django(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with Django tester framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    lines = log.split("\n")

    prev_test = None
    for line in lines:
        line = line.strip()

        # This isn't ideal but the test output spans multiple lines
        if "--version is equivalent to version" in line:
            test_status_map["--version is equivalent to version"] = (
                TestStatus.PASSED.value
            )

        # Log it in case of error
        if " ... " in line:
            prev_test = line.split(" ... ")[0]

        pass_suffixes = (" ... ok", " ... OK", " ...  OK")
        for suffix in pass_suffixes:
            if line.endswith(suffix):
                # TODO: Temporary, exclusive fix for django__django-7188
                # The proper fix should involve somehow getting the test results to
                # print on a separate line, rather than the same line
                if line.strip().startswith(
                    "Applying sites.0002_alter_domain_unique...test_no_migrations"
                ):
                    line = line.split("...", 1)[-1].strip()
                test = line.rsplit(suffix, 1)[0]
                test_status_map[test] = TestStatus.PASSED.value
                break
        if " ... skipped" in line:
            test = line.split(" ... skipped")[0]
            test_status_map[test] = TestStatus.SKIPPED.value
        if line.endswith(" ... FAIL"):
            test = line.split(" ... FAIL")[0]
            test_status_map[test] = TestStatus.FAILED.value
        if line.startswith("FAIL:"):
            test = line.split()[1].strip()
            test_status_map[test] = TestStatus.FAILED.value
        if line.endswith(" ... ERROR"):
            test = line.split(" ... ERROR")[0]
            test_status_map[test] = TestStatus.ERROR.value
        if line.startswith("ERROR:"):
            test = line.split()[1].strip()
            test_status_map[test] = TestStatus.ERROR.value

        if line.lstrip().startswith("ok") and prev_test is not None:
            # It means the test passed, but there's some additional output (including new lines)
            # between "..." and "ok" message
            test = prev_test
            test_status_map[test] = TestStatus.PASSED.value

    # TODO: This is very brittle, we should do better
    # There's a bug in the django logger, such that sometimes a test output near the end gets
    # interrupted by a particular long multiline print statement.
    # We have observed this in one of 3 forms:
    # - "{test_name} ... Testing against Django installed in {*} silenced.\nok"
    # - "{test_name} ... Internal Server Error: \/(.*)\/\nok"
    # - "{test_name} ... System check identified no issues (0 silenced).\nok"
    patterns = [
        r"^(.*?)\s\.\.\.\sTesting\ against\ Django\ installed\ in\ ((?s:.*?))\ silenced\)\.\nok$",
        r"^(.*?)\s\.\.\.\sInternal\ Server\ Error:\ \/(.*)\/\nok$",
        r"^(.*?)\s\.\.\.\sSystem check identified no issues \(0 silenced\)\nok$",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, log, re.MULTILINE):
            test_name = match.group(1)
            test_status_map[test_name] = TestStatus.PASSED.value
    return test_status_map


def parse_log_pytest_v2(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework (Later Version)

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    escapes = "".join([chr(char) for char in range(1, 32)])
    for i, line in enumerate(log.split("\n")):
        # line = re.sub(r"\[(\d+)m", "", line)
        # translator = str.maketrans("", "", escapes)
        # line = line.translate(translator)

        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        line = ansi_escape.sub("", line)

        if any([line.startswith(x.value) for x in TestStatus]):
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split(maxsplit=1)

            if len(test_case) < 2:
                continue

            test_case_name = test_case[1]
            test_case_name = pytest_drop_trailing_info(test_case_name)

            test_status_map[test_case[1]] = test_case[0]
        # Support older pytest versions by checking if the line ends with the test status
        elif any([line.endswith(x.value) for x in TestStatus]):
            test_case = line.rsplit(maxsplit=1)

            if len(test_case) < 2:
                continue

            test_status_map[test_case[0]] = test_case[1]
    return test_status_map


def parse_log_seaborn(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with seaborn testing framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        if line.startswith(TestStatus.FAILED.value):
            test_case = line.split()[1]
            test_status_map[test_case] = TestStatus.FAILED.value
        elif f" {TestStatus.PASSED.value} " in line:
            parts = line.split()
            if parts[1] == TestStatus.PASSED.value:
                test_case = parts[0]
                test_status_map[test_case] = TestStatus.PASSED.value
        elif line.startswith(TestStatus.PASSED.value):
            parts = line.split()
            test_case = parts[1]
            test_status_map[test_case] = TestStatus.PASSED.value
    return test_status_map


def parse_log_sympy(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with Sympy framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    if "pytest" in log:
        # Split into lines.
        lines = log.splitlines()
        sympy_lines = [line for line in lines if line.strip().startswith("sympy/")]
        for line in sympy_lines:
            try:
                test_name, test_status, _ = line.split(None, 2)
                test_status_map[test_name] = test_status
            except ValueError:
                continue
        return test_status_map

    pattern = r"(_*) (.*)\.py:(.*) (_*)"
    matches = re.findall(pattern, log)
    for match in matches:
        test_case = f"{match[1]}.py:{match[2]}"
        test_status_map[test_case] = TestStatus.FAILED.value
    for line in log.split("\n"):
        line = line.strip()
        if line.startswith("test_"):
            if line.endswith("[FAIL]") or line.endswith("[OK]"):
                line = line[: line.rfind("[")]
                line = line.strip()
            if line.endswith(" E"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.ERROR.value
            if line.endswith(" F"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.FAILED.value
            if line.endswith(" ok"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.PASSED.value
    return test_status_map


def parse_log_matplotlib(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        line = line.replace("MouseButton.LEFT", "1")
        line = line.replace("MouseButton.RIGHT", "3")
        if any([line.startswith(x.value) for x in TestStatus]):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


def parse_log_pytest_pydantic(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework (Later Version)

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    escapes = "".join([chr(char) for char in range(1, 32)])
    for line in log.split("\n"):
        line = re.sub(r"\[(\d+)m", "", line)
        translator = str.maketrans("", "", escapes)
        line = line.translate(translator)
        # additionally to pytest v2 we remove the [...] from FAILED
        line = re.sub(r"FAILED\s*\[.*?\]", "FAILED", line)
        if "tests/test_main.py::test_model_post_init_supertype_private_attr" in line:
            print(line)

        if any([line.startswith(x.value) for x in TestStatus]):
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            test_status_map[test_case[1]] = test_case[0]
        # Support older pytest versions by checking if the line ends with the test status
        elif any([line.endswith(x.value) for x in TestStatus]):
            test_case = line.split()
            test_status_map[test_case[0]] = test_case[1]
    return test_status_map


parse_log_astroid = parse_log_pytest
parse_log_flask = parse_log_pytest
parse_log_marshmallow = parse_log_pytest
parse_log_pvlib = parse_log_pytest
parse_log_pyvista = parse_log_pytest
parse_log_sqlfluff = parse_log_pytest
parse_log_xarray = parse_log_pytest

parse_log_pydicom = parse_log_pytest_options
parse_log_requests = parse_log_pytest_options
parse_log_pylint = parse_log_pytest_options

parse_log_astropy = parse_log_pytest_v2
parse_log_scikit = parse_log_pytest_v2
parse_log_sphinx = parse_log_pytest_v2


MAP_REPO_TO_PARSER = {
    "astropy/astropy": parse_log_astropy,
    "django/django": parse_log_django,
    "marshmallow-code/marshmallow": parse_log_marshmallow,
    "matplotlib/matplotlib": parse_log_matplotlib,
    "mwaskom/seaborn": parse_log_seaborn,
    "pallets/flask": parse_log_flask,
    "psf/requests": parse_log_requests,
    "pvlib/pvlib-python": parse_log_pvlib,
    "pydata/xarray": parse_log_xarray,
    "pydicom/pydicom": parse_log_pydicom,
    "pylint-dev/astroid": parse_log_astroid,
    "pylint-dev/pylint": parse_log_pylint,
    "pytest-dev/pytest": parse_log_pytest,
    "pyvista/pyvista": parse_log_pyvista,
    "scikit-learn/scikit-learn": parse_log_scikit,
    "sqlfluff/sqlfluff": parse_log_sqlfluff,
    "sphinx-doc/sphinx": parse_log_sphinx,
    "sympy/sympy": parse_log_sympy,
}

parse_log_mypy = parse_log_pytest
parse_log_moto = parse_log_pytest
parse_log_conan = parse_log_pytest
MAP_REPO_TO_PARSER.update(
    {
        "python/mypy": parse_log_mypy,
        "getmoto/moto": parse_log_moto,
        "conan-io/conan": parse_log_conan,
    }
)


parse_log_modin = parse_log_pytest
MAP_REPO_TO_PARSER.update(
    {
        "modin-project/modin": parse_log_modin,
    }
)

parse_log_monai = parse_log_pytest
MAP_REPO_TO_PARSER.update(
    {
        "Project-MONAI/MONAI": parse_log_monai,
    }
)

parse_log_dvc = parse_log_pytest
MAP_REPO_TO_PARSER.update(
    {
        "iterative/dvc": parse_log_dvc,
    }
)

parse_log_dask = parse_log_pytest
MAP_REPO_TO_PARSER.update(
    {
        "dask/dask": parse_log_dask,
    }
)

parse_log_bokeh = parse_log_pytest
MAP_REPO_TO_PARSER.update(
    {
        "bokeh/bokeh": parse_log_bokeh,
    }
)

parse_log_mne = parse_log_pytest
MAP_REPO_TO_PARSER.update(
    {
        "mne-tools/mne-python": parse_log_mne,
    }
)

parse_log_hypothesis = parse_log_pytest
MAP_REPO_TO_PARSER.update(
    {
        "HypothesisWorks/hypothesis": parse_log_hypothesis,
    }
)

parse_log_pydantic = parse_log_pytest_pydantic
MAP_REPO_TO_PARSER.update(
    {
        "pydantic/pydantic": parse_log_pydantic,
    }
)

parse_log_pandas = parse_log_pytest_v2
MAP_REPO_TO_PARSER.update({"pandas-dev/pandas": parse_log_pandas})

parse_log_hydra = parse_log_pytest
MAP_REPO_TO_PARSER.update({"facebookresearch/hydra": parse_log_hydra})


def parse_log_numpy(log) -> dict[str, str]:
    raw_result = parse_log_pytest(log)

    # Do some reprocessing: drop the [...] part only when the prefix is unique.
    test_status_map = {}
    prefix_count = {}
    for test_case in raw_result.keys():
        prefix = test_case.split("[", 1)[0]
        if prefix not in prefix_count:
            prefix_count[prefix] = 0
        prefix_count[prefix] += 1

    for test_case, status in raw_result.items():
        # Keep only prefix if unique
        prefix = test_case.split("[", 1)[0]
        if prefix_count[prefix] == 1:
            test_status_map[prefix] = status

    return test_status_map


MAP_REPO_TO_PARSER.update({"numpy/numpy": parse_log_numpy})

parse_log_scipy = parse_log_numpy
MAP_REPO_TO_PARSER.update({"scipy/scipy": parse_log_scipy})


# All keys should be in lower case
LOWER_MAP_REPO_TO_PARSER = {k.lower(): v for k, v in MAP_REPO_TO_PARSER.items()}
MAP_REPO_TO_PARSER = LOWER_MAP_REPO_TO_PARSER
