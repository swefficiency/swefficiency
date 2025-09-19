from enum import Enum
from pathlib import Path
from typing import TypedDict

from typing_extensions import NotRequired

# Constants - Evaluation Log Directories
BASE_IMAGE_BUILD_DIR = Path("logs/build_images/base")
ENV_IMAGE_BUILD_DIR = Path("logs/build_images/env")
INSTANCE_IMAGE_BUILD_DIR = Path("logs/build_images/instances")
ANNOTATE_IMAGE_BUILD_DIR = Path("logs/build_images/annotate")
RUN_EVALUATION_LOG_DIR = Path("logs/run_evaluation")


# Constants - Task Instance Class
class SWEfficiencyInstance(TypedDict):
    repo: str
    instance_id: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: str
    created_at: str
    version: str
    FAIL_TO_PASS: str
    PASS_TO_PASS: str
    environment_setup_commit: str

    # SWE-Perf Additions
    workload: NotRequired[str]
    speedup: NotRequired[float]
    covering_tests: NotRequired[list[str]]
    notes: NotRequired[str]
    single_thread_tests: NotRequired[list[str]]


# Constants - Test Types, Statuses, Commands
FAIL_TO_PASS = "FAIL_TO_PASS"
FAIL_TO_FAIL = "FAIL_TO_FAIL"
PASS_TO_PASS = "PASS_TO_PASS"
PASS_TO_FAIL = "PASS_TO_FAIL"


class ResolvedStatus(Enum):
    NO = "RESOLVED_NO"
    PARTIAL = "RESOLVED_PARTIAL"
    FULL = "RESOLVED_FULL"


class TestStatus(Enum):
    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
    XFAIL = "XFAIL"


TEST_PYTEST = (
    "pytest --no-header -rA --tb=no -p no:cacheprovider --continue-on-collection-errors"
)
TEST_PYTEST_VERBOSE = (
    "pytest -v -rA --tb=long -p no:cacheprovider --continue-on-collection-errors"
)
TEST_PYTEST_DISTRIBUTED = "pytest --no-header -rA --tb=no -p no:cacheprovider --continue-on-collection-errors -n 4"

TEST_ASTROPY_PYTEST = "pytest -rA -vv -o console_output_style=classic --tb=no"
TEST_DJANGO = "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1"
TEST_DJANGO_NO_PARALLEL = "./tests/runtests.py --verbosity 2"
TEST_SEABORN = "pytest --no-header -rA"
TEST_SEABORN_VERBOSE = "pytest -rA --tb=long"
# TEST_PYTEST = "pytest -rA"
# TEST_PYTEST_VERBOSE = "pytest -rA --tb=long"
TEST_SPHINX = "tox --current-env -epy39 -v --"

TEST_SYMPY = "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test --no-subprocess --verbose"
TEST_SYMPY_VERBOSE = "bin/test -C --verbose"
TEST_SYMPY_DISTRIBUTED = "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test --no-subprocess --verbose"

"""
SPECS = {
    # commands to run before doing anything
    "pre_install": [
        "apt-get update && apt-get install -y locales",
        "echo 'en_US UTF-8' > /etc/locale.gen",
        "locale-gen en_US.UTF-8",
    ],

    #python version, can be skipped if using environment.yml in "packages"
    "python": "3.9", 

        "env_patches": [
            # "some commands to modify the env"
            # environment.yml or requirements.txt is at current dir
        ],
    # pacakges type:
    # either environment.yml or requirements.txt 
    # or specifying the packages naively
    "packages": "numpy scipy pytest" or "environment.yml" or "requirements.txt"

    # installation command
    "install": "pip install -v --no-use-pep517 --no-build-isolation -e .",

    # prefix for running the test command
    "test_cmd": "pytest -n0 -rA"
}
"""

# Constants - Installation Specifications

SPECS_SKLEARN = {}


# Note: Can't really do anything here?
# SPECS_SKLEARN.update({
#     k: {
#         "python": "3.4",
#         "packages": "'numpy==1.6.1' 'scipy==0.9' 'cython==0.25.2' pytest",
#         "install": "pip install -v --no-use-pep517 --no-build-isolation -e .",
#         "pip_packages": [
#             "'cython==0.25.2'",
#             "numpy==1.6.1",
#             "setuptools",
#             "scipy==0.9",
#         ],
#         "test_cmd": TEST_PYTEST,
#     }
#     for k in ['0.14', '0.15', '0.16', '0.17']
# })


# These use nosetest, so maybe ignore?
# SPECS_SKLEARN.update({
#     k: {
#         "python": "3.6",
#         "packages": "numpy scipy 'cython==0.25.2' pytest pandas matplotlib",
#         "install": "pip install -v --no-use-pep517 --no-build-isolation -e .",
#         "pip_packages": [
#             "'cython==0.25.2'",
#             "numpy==1.8.2",
#             "setuptools",
#             "scipy==1.0.0",
#         ],
#         "test_cmd": "",
#     }
#     for k in ['0.18', '0.19']
# })


SPECS_SKLEARN.update(
    {
        k: {
            "python": "3.6",
            "packages": "numpy scipy 'cython<0.28' 'pytest<7.0' pandas matplotlib pyamg matplotlib",
            "install": "pip install --no-use-pep517 --no-build-isolation -e .",
            "pip_packages": [
                "'cython<0.28'",
                "'numpy==1.17.3'",
                "setuptools",
                "scipy==1.5.2",
            ],
            "test_cmd": TEST_PYTEST,
        }
        for k in ["0.2", "0.20"]
    }
)

SPECS_SKLEARN.update(
    {
        k: {
            "python": "3.6",
            "packages": "numpy scipy 'cython==0.28.5' 'pytest<7.0' pandas matplotlib pyamg matplotlib",
            "install": "pip install --no-use-pep517 --no-build-isolation -e .",
            "pip_packages": [
                "'cython==0.28.5'",
                "'numpy==1.17.3'",
                "setuptools",
                "scipy==1.5.2",
            ],
            "test_cmd": TEST_PYTEST,
        }
        for k in ["0.201", "0.21"]
    }
)

SPECS_SKLEARN.update(
    {
        k: {
            "python": "3.6",
            "packages": "numpy scipy 'cython<3' 'pytest<7.0' pandas matplotlib pyamg matplotlib",
            "install": "pip install --no-use-pep517 --no-build-isolation -e .",
            "pip_packages": [
                "'cython<3'",
                "numpy==1.19.2",
                "setuptools",
                "scipy==1.5.2",
            ],
            "test_cmd": TEST_PYTEST,
        }
        for k in ["0.22", "0.222", "0.23", "0.24"]
    }
)

SPECS_SKLEARN.update(
    {
        k: {
            "python": "3.9",
            "packages": "'numpy==1.19.5' 'scipy==1.6.0' 'cython<3' 'pytest<7.0' 'pandas<2.0.0' 'matplotlib<3.9.0' 'setuptools<61' pytest joblib threadpoolctl pyamg matplotlib",
            "install": "pip install --no-use-pep517 --no-build-isolation -e .",
            "pip_packages": ["'cython<3'", "'setuptools<61'", "numpy", "scipy"],
            "test_cmd": TEST_PYTEST,
        }
        for k in ["0.9", "1.0", "1.1"]
    }
)

SPECS_SKLEARN.update(
    {
        k: {
            "python": "3.9",
            "packages": "'numpy==1.19.5' 'scipy==1.6.0' 'cython<3' 'pytest<7.0' 'pandas<2.0.0' 'matplotlib<3.9.0' 'setuptools<65' pytest joblib threadpoolctl pyamg matplotlib",
            "install": "pip install --no-use-pep517 --no-build-isolation -e .",
            "pip_packages": ["'cython<3'", "'setuptools<65'", "numpy", "scipy"],
            "test_cmd": TEST_PYTEST,
        }
        for k in ["1.2"]
    }
)

SPECS_SKLEARN.update(
    {
        k: {
            "python": "3.9",
            "packages": "'numpy==1.19.5' 'scipy==1.6.0' 'cython<3' pytest 'pandas<2.0.0' 'matplotlib<3.9.0' 'setuptools<65' pytest 'joblib>=1.2.0' 'threadpoolctl>=3.1.0' pyamg matplotlib",
            "install": "pip install --no-use-pep517 --no-build-isolation -e .",
            "pip_packages": ["cython", "setuptools", "numpy", "scipy"],
            "test_cmd": TEST_PYTEST,
        }
        for k in ["1.3", "1.4"]
    }
)


SPECS_SKLEARN.update(
    {
        k: {
            "python": "3.9",
            "packages": "'numpy==1.19.5' 'scipy==1.6.0' 'cython' pytest 'pandas<2.0.0' 'matplotlib<3.9.0' 'setuptools<65' pytest 'joblib>=1.2.0' 'threadpoolctl>=3.1.0' meson-python pyamg matplotlib",
            "install": "pip install --no-build-isolation -e .",
            "pip_packages": ["cython", "setuptools", "numpy", "scipy", "meson-python"],
            "test_cmd": TEST_PYTEST,
        }
        for k in ["1.5", "1.6", "1.7"]
    }
)

for k in SPECS_SKLEARN:
    # HACK: This is needed because we use an AST pruning technique to identify tests.
    SPECS_SKLEARN[k]["pip_packages"].extend(["intervaltree"])

SPECS_FLASK = {
    "2.0": {
        "python": "3.9",
        "packages": "requirements.txt",
        "install": "pip install -e .",
        "pip_packages": [
            "setuptools==70.0.0",
            "Werkzeug==2.3.7",
            "Jinja2==3.0.1",
            "itsdangerous==2.1.2",
            "click==8.0.1",
            "MarkupSafe==2.1.3",
        ],
        "test_cmd": TEST_PYTEST,
    },
    "2.1": {
        "python": "3.10",
        "packages": "requirements.txt",
        "install": "pip install -e .",
        "pip_packages": [
            "click==8.1.3",
            "itsdangerous==2.1.2",
            "Jinja2==3.1.2",
            "MarkupSafe==2.1.1",
            "Werkzeug==2.3.7",
        ],
        "test_cmd": TEST_PYTEST,
    },
}
SPECS_FLASK.update(
    {
        k: {
            "python": "3.11",
            "packages": "requirements.txt",
            "install": "pip install -e .",
            "pip_packages": [
                "click==8.1.3",
                "itsdangerous==2.1.2",
                "Jinja2==3.1.2",
                "MarkupSafe==2.1.1",
                "Werkzeug==2.3.7",
            ],
            "test_cmd": TEST_PYTEST,
        }
        for k in ["2.2", "2.3"]
    }
)

SPECS_FLASK.update(
    {
        k: {
            "python": "3.11",
            "packages": "requirements.txt",
            "install": "pip install -e .",
            "test_cmd": TEST_PYTEST,
        }
        for k in ["3.0", "3.1"]
    }
)

SPECS_DJANGO = {
    k: {
        "python": "3.5",
        "packages": "requirements.txt",
        "pre_install": [
            "apt-get update && apt-get install -y locales",
            "echo 'en_US UTF-8' > /etc/locale.gen",
            "locale-gen en_US.UTF-8",
        ],
        "install": "python setup.py install",
        "pip_packages": ["setuptools"],
        "eval_commands": [
            "export LANG=en_US.UTF-8",
            "export LC_ALL=en_US.UTF-8",
            "export PYTHONIOENCODING=utf8",
            "export LANGUAGE=en_US:en",
        ],
        "test_cmd": TEST_DJANGO,
    }
    for k in ["1.7", "1.8", "1.9", "1.10", "1.11", "2.0", "2.1", "2.2"]
}
SPECS_DJANGO.update(
    {
        k: {
            "python": "3.5",
            "install": "python setup.py install",
            "test_cmd": TEST_DJANGO,
        }
        for k in ["1.4", "1.5", "1.6"]
    }
)
SPECS_DJANGO.update(
    {
        k: {
            "python": "3.6",
            "packages": "requirements.txt",
            "install": "pip install -e .",
            "eval_commands": [
                "sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen",
                "export LANG=en_US.UTF-8",
                "export LANGUAGE=en_US:en",
                "export LC_ALL=en_US.UTF-8",
            ],
            "test_cmd": TEST_DJANGO,
        }
        for k in ["3.0", "3.1", "3.2"]
    }
)
SPECS_DJANGO.update(
    {
        k: {
            "python": "3.8",
            "packages": "requirements.txt",
            "install": "pip install -e .",
            "test_cmd": TEST_DJANGO,
        }
        for k in ["4.0"]
    }
)
SPECS_DJANGO.update(
    {
        k: {
            "python": "3.9",
            "packages": "requirements.txt",
            "install": "pip install -e .",
            "test_cmd": TEST_DJANGO,
        }
        for k in ["4.1", "4.2"]
    }
)
SPECS_DJANGO.update(
    {
        k: {
            "python": "3.11",
            "packages": "requirements.txt",
            "install": "pip install -e .",
            "test_cmd": TEST_DJANGO,
        }
        for k in ["5.0", "5.1", "5.2", "6.0"]
    }
)
SPECS_DJANGO["1.9"]["test_cmd"] = TEST_DJANGO_NO_PARALLEL

SPECS_REQUESTS = {
    k: {
        "python": "3.9",
        "packages": "pytest",
        "install": "pip install .",
        "test_cmd": TEST_PYTEST,
    }
    for k in ["0.7", "0.8", "0.9", "0.11", "0.13", "0.14", "1.1", "1.2", "2.0", "2.2"]
    + ["2.3", "2.4", "2.5", "2.7", "2.8", "2.9", "2.10", "2.11", "2.12", "2.17"]
    + ["2.18", "2.19", "2.22", "2.26", "2.25", "2.27", "3.0"]
    + ["2.31"]
}

SPECS_SEABORN = {
    k: {
        "python": "3.9",
        "install": "pip install -e .",
        "pip_packages": [
            "contourpy==1.1.0",
            "cycler==0.11.0",
            "fonttools==4.42.1",
            "importlib-resources==6.0.1",
            "kiwisolver==1.4.5",
            "matplotlib==3.7.2",
            "numpy==1.25.2",
            "packaging==23.1",
            "pandas==1.3.5",  # 2.0.3
            "pillow==10.0.0",
            "pyparsing==3.0.9",
            "pytest",
            "python-dateutil==2.8.2",
            "pytz==2023.3.post1",
            "scipy==1.11.2",
            "six==1.16.0",
            "tzdata==2023.1",
            "zipp==3.16.2",
        ],
        "test_cmd": TEST_SEABORN,
    }
    for k in ["0.11"]
}
SPECS_SEABORN.update(
    {
        k: {
            "python": "3.9",
            "install": "pip install -e .[dev]",
            "pip_packages": [
                "contourpy==1.1.0",
                "cycler==0.11.0",
                "fonttools==4.42.1",
                "importlib-resources==6.0.1",
                "kiwisolver==1.4.5",
                "matplotlib==3.7.2",
                "numpy==1.25.2",
                "packaging==23.1",
                "pandas==2.0.0",
                "pillow==10.0.0",
                "pyparsing==3.0.9",
                "pytest",
                "python-dateutil==2.8.2",
                "pytz==2023.3.post1",
                "scipy==1.11.2",
                "six==1.16.0",
                "tzdata==2023.1",
                "zipp==3.16.2",
            ],
            "test_cmd": TEST_SEABORN,
        }
        for k in ["0.12", "0.13", "0.14"]
    }
)

SPECS_PYTEST = {
    k: {
        "python": "3.9",
        "install": "pip install -e .",
        "test_cmd": TEST_PYTEST,
    }
    for k in [
        "4.4",
        "4.5",
        "4.6",
        "5.0",
        "5.1",
        "5.2",
        "5.3",
        "5.4",
        "6.0",
        "6.2",
        "6.3",
        "7.0",
        "7.1",
        "7.2",
        "7.4",
        "8.0",
        "8.1",
        "8.2",
        "8.3",
        "8.4",
    ]
}
SPECS_PYTEST["4.4"]["pip_packages"] = [
    "atomicwrites==1.4.1",
    "attrs==23.1.0",
    "more-itertools==10.1.0",
    "pluggy==0.13.1",
    "py==1.11.0",
    "setuptools==68.0.0",
    "six==1.16.0",
]
SPECS_PYTEST["4.5"]["pip_packages"] = [
    "atomicwrites==1.4.1",
    "attrs==23.1.0",
    "more-itertools==10.1.0",
    "pluggy==0.11.0",
    "py==1.11.0",
    "setuptools==68.0.0",
    "six==1.16.0",
    "wcwidth==0.2.6",
]
SPECS_PYTEST["4.6"]["pip_packages"] = [
    "atomicwrites==1.4.1",
    "attrs==23.1.0",
    "more-itertools==10.1.0",
    "packaging==23.1",
    "pluggy==0.13.1",
    "py==1.11.0",
    "six==1.16.0",
    "wcwidth==0.2.6",
]
for k in ["5.0", "5.1", "5.2"]:
    SPECS_PYTEST[k]["pip_packages"] = [
        "atomicwrites==1.4.1",
        "attrs==23.1.0",
        "more-itertools==10.1.0",
        "packaging==23.1",
        "pluggy==0.13.1",
        "py==1.11.0",
        "wcwidth==0.2.6",
    ]
SPECS_PYTEST["5.3"]["pip_packages"] = [
    "attrs==23.1.0",
    "more-itertools==10.1.0",
    "packaging==23.1",
    "pluggy==0.13.1",
    "py==1.11.0",
    "wcwidth==0.2.6",
]
SPECS_PYTEST["5.4"]["pip_packages"] = [
    "py==1.11.0",
    "packaging==23.1",
    "attrs==23.1.0",
    "more-itertools==10.1.0",
    "pluggy==0.13.1",
]
SPECS_PYTEST["6.0"]["pip_packages"] = [
    "attrs==23.1.0",
    "iniconfig==2.0.0",
    "more-itertools==10.1.0",
    "packaging==23.1",
    "pluggy==0.13.1",
    "py==1.11.0",
    "toml==0.10.2",
]
for k in ["6.2", "6.3"]:
    SPECS_PYTEST[k]["pip_packages"] = [
        "attrs==23.1.0",
        "iniconfig==2.0.0",
        "packaging==23.1",
        "pluggy==0.13.1",
        "py==1.11.0",
        "toml==0.10.2",
    ]
SPECS_PYTEST["7.0"]["pip_packages"] = [
    "attrs==23.1.0",
    "iniconfig==2.0.0",
    "packaging==23.1",
    "pluggy==0.13.1",
    "py==1.11.0",
]
for k in ["7.1", "7.2"]:
    SPECS_PYTEST[k]["pip_packages"] = [
        "attrs==23.1.0",
        "iniconfig==2.0.0",
        "packaging==23.1",
        "pluggy==0.13.1",
        "py==1.11.0",
        "tomli==2.0.1",
    ]
SPECS_PYTEST["7.4"]["pip_packages"] = [
    "iniconfig==2.0.0",
    "packaging==23.1",
    "pluggy==1.3.0",
    "exceptiongroup==1.1.3",
    "tomli==2.0.1",
]
SPECS_PYTEST["8.0"]["pip_packages"] = [
    "iniconfig==2.0.0",
    "packaging==23.1",
    "pluggy==1.3.0",
    "exceptiongroup==1.1.3",
    "tomli==2.0.1",
]

for k in ["8.0", "8.1", "8.2", "8.3", "8.4"]:
    SPECS_PYTEST[k]["pip_packages"] = [
        "decorator",
        "attrs==23.1.0",
    ]


TEST_MATPLOTLIB = TEST_PYTEST
TEST_MATPLOTLIB_DISTRIBUTED = TEST_PYTEST_DISTRIBUTED
SPECS_MATPLOTLIB = {
    k: {
        "python": "3.11",
        "packages": "environment.yml",
        "env_patches": [
            "apt-get -y update && apt-get -y upgrade && DEBIAN_FRONTEND=noninteractive apt-get install -y imagemagick ffmpeg texlive texlive-latex-extra texlive-fonts-recommended texlive-xetex texlive-luatex cm-super dvipng libfreetype6 libfreetype6-dev libpng-dev libfontconfig1-dev xclip libcairo2-dev pkg-config libgirepository1.0-dev",
            "sed -i 's|git+git://|git+https://|g' environment.yml",
            # Install old GS.
            "wget -q https://github.com/ArtifexSoftware/ghostpdl-downloads/releases/download/gs927/ghostscript-9.27.tar.gz -O /tmp/ghostscript-9.27.tar.gz",
            "tar -xzf /tmp/ghostscript-9.27.tar.gz -C /tmp",
            "cd /tmp/ghostscript-9.27 && ./configure && make && make install",
            "export GS_OPTIONS='-dNOSAFER'",
            "cd -",
        ],
        "install": "pip install -e . --no-build-isolation",
        "pip_packages": [
            "contourpy==1.1.0",
            "cycler==0.11.0",
            "fonttools==4.42.1",
            "ghostscript",
            "kiwisolver==1.4.5",
            "numpy==1.25.2",
            "'packaging>=25'",
            "pillow==10.0.0",
            "pikepdf",
            "pyparsing==3.0.9",
            "python-dateutil==2.8.2",
            "six==1.16.0",
            "'setuptools>70'",
            "setuptools-scm==7.1.0",
            "typing-extensions==4.7.1",
            "latex",
            "pytest-xdist",
            "cppy",
            "'meson-python>=0.17'",
            "pycairo",
            "PyGObject",
        ],
        "test_cmd": TEST_MATPLOTLIB,
        "distributed_test_cmd": TEST_MATPLOTLIB_DISTRIBUTED,
    }
    for k in ["3.10"]
}


SPECS_MATPLOTLIB.update(
    {
        k: {
            "python": "3.11",
            "packages": "environment.yml",
            "install": "pip install -e . --no-build-isolation",
            "env_patches": [
                "apt-get -y update && apt-get -y upgrade && DEBIAN_FRONTEND=noninteractive apt-get install -y imagemagick ffmpeg texlive texlive-latex-extra texlive-fonts-recommended texlive-xetex texlive-luatex cm-super dvipng libfreetype6 libfreetype6-dev libpng-dev libfontconfig1-dev xclip libcairo2-dev pkg-config libgirepository1.0-dev",
                # Install old GS.
                "wget -q https://github.com/ArtifexSoftware/ghostpdl-downloads/releases/download/gs927/ghostscript-9.27.tar.gz -O /tmp/ghostscript-9.27.tar.gz",
                "tar -xzf /tmp/ghostscript-9.27.tar.gz -C /tmp",
                "cd /tmp/ghostscript-9.27 && ./configure && make && make install",
                "export GS_OPTIONS='-dNOSAFER'",
                "cd -",
            ],
            "pip_packages": [
                "contourpy==1.1.0",
                "cycler==0.11.0",
                "fonttools==4.42.1",
                "ghostscript",
                "kiwisolver==1.4.5",
                "numpy==1.25.2",
                "'packaging>=25'",
                "pillow==10.0.0",
                "pikepdf",
                "pyparsing==2.4.7",
                "python-dateutil==2.8.2",
                "six==1.16.0",
                "'setuptools>70'",
                "setuptools-scm==7.1.0",
                "typing-extensions==4.7.1",
                "latex",
                "pytest-xdist",
                "cppy",
                "'meson-python>=0.17'",
                "pycairo",
                "PyGObject",
            ],
            "test_cmd": TEST_MATPLOTLIB,
            "distributed_test_cmd": TEST_MATPLOTLIB_DISTRIBUTED,
        }
        for k in ["3.5", "3.6", "3.7", "3.8", "3.9"]
    }
)
SPECS_MATPLOTLIB.update(
    {
        k: {
            "python": "3.7",
            "packages": "requirements.txt",
            "install": "pip install -e .",
            "env_patches": [
                "apt-get -y update && apt-get -y upgrade && DEBIAN_FRONTEND=noninteractive apt-get install -y imagemagick ffmpeg libfreetype6 libfreetype6-dev pkg-config texlive texlive-latex-extra texlive-fonts-recommended texlive-xetex texlive-luatex cm-super libpng-dev libfontconfig1-dev xclip libcairo2-dev pkg-config libgirepository1.0-dev",
                # Install old GS.
                "wget -q https://github.com/ArtifexSoftware/ghostpdl-downloads/releases/download/gs927/ghostscript-9.27.tar.gz -O /tmp/ghostscript-9.27.tar.gz",
                "tar -xzf /tmp/ghostscript-9.27.tar.gz -C /tmp",
                "cd /tmp/ghostscript-9.27 && ./configure && make && make install",
                "export GS_OPTIONS='-dNOSAFER'",
                "cd -",
                "sed -i '/sphinx-gallery/d' $HOME/requirements.txt",
            ],
            # TODO: Pytest changed how it catches no exceptions, requires 7.0 or smaller, but won't fix for now bc idk what the oter extension versions should be.
            "pip_packages": [
                "'pytest'",
                "ipython",
                "latex",
                "'setuptools<=59.5.0'",
                "'setuptools-scm<=7.1.0'",
                "'pandas<=1.1.5'",
                "'numpy<1.20'",
                "pytest-xdist",
                "cppy",
                "kiwisolver",
                "'cycler<=0.11'",
                "pycairo",
                "PyGObject",
            ],
            "test_cmd": TEST_MATPLOTLIB,
            "distributed_test_cmd": TEST_MATPLOTLIB_DISTRIBUTED,
        }
        for k in ["3.1", "3.2", "3.3", "3.4"]
    }
)
SPECS_MATPLOTLIB.update(
    {
        k: {
            "python": "3.6",
            "packages": "requirements.txt",
            "install": "pip install -e .",
            "env_patches": [
                "apt-get -y update && apt-get -y upgrade && apt-get install -y imagemagick ffmpeg libfreetype6 libfreetype6-dev pkg-config libpng-dev libfontconfig1-dev xclip libcairo2-dev pkg-config libgirepository1.0-dev",
                # Install old GS.
                "wget -q https://github.com/ArtifexSoftware/ghostpdl-downloads/releases/download/gs927/ghostscript-9.27.tar.gz -O /tmp/ghostscript-9.27.tar.gz",
                "tar -xzf /tmp/ghostscript-9.27.tar.gz -C /tmp",
                "cd /tmp/ghostscript-9.27 && ./configure && make && make install",
                "export GS_OPTIONS='-dNOSAFER'",
                "cd -",
            ],
            "pip_packages": [
                "pytest",
                "latex",
                "'setuptools<=59.5.0'",
                "'setuptools-scm<=7.1.0'",
                "'pandas<=1.1.5'",
                "'numpy<1.20'",
                "pytest-xdist",
                "cppy",
                "kiwisolver",
                "'cycler<=0.11'",
                "pycairo",
                "PyGObject",
            ],
            "test_cmd": TEST_MATPLOTLIB,
            "distributed_test_cmd": TEST_MATPLOTLIB_DISTRIBUTED,
        }
        for k in ["3.0"]
    }
)
SPECS_MATPLOTLIB.update(
    {
        k: {
            "python": "3.5",
            "install": "python setup.py build; python setup.py install",
            "env_patches": [
                "apt-get -y update && apt-get -y upgrade && apt-get install -y imagemagick ffmpeg libfreetype6 libfreetype6-dev libpng-dev libfontconfig1-dev xclip libcairo2-dev pkg-config libgirepository1.0-dev",
                # Install old GS.
                "wget -q https://github.com/ArtifexSoftware/ghostpdl-downloads/releases/download/gs927/ghostscript-9.27.tar.gz -O /tmp/ghostscript-9.27.tar.gz",
                "tar -xzf /tmp/ghostscript-9.27.tar.gz -C /tmp",
                "cd /tmp/ghostscript-9.27 && ./configure && make && make install",
                "export GS_OPTIONS='-dNOSAFER'",
                "cd -",
            ],
            "pip_packages": [
                "pytest",
                "latex",
                "'setuptools<=59.5.0'",
                "'setuptools-scm<=7.1.0'",
                "pytest-xdist",
                "'pandas<=1.1.5'",
                "'numpy<1.20'",
                "cppy",
                "kiwisolver",
                "'cycler<=0.11'",
                "pycairo",
                "PyGObject",
            ],
            "execute_test_as_nonroot": True,
            "test_cmd": TEST_MATPLOTLIB,
            "distributed_test_cmd": TEST_MATPLOTLIB_DISTRIBUTED,
        }
        for k in ["2.0", "2.1", "2.2", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5"]
    }
)

SPECS_SPHINX = {
    k: {
        "python": "3.9",
        "pip_packages": ["tox==4.16.0", "tox-current-env==0.0.11"],
        "install": "pip install -e .[test]",
        "pre_install": ["sed -i 's/pytest/pytest -rA/' tox.ini"],
        "test_cmd": TEST_SPHINX,
    }
    for k in ["1.5", "1.6", "1.7", "1.8", "2.0", "2.1", "2.2", "2.3", "2.4", "3.0"]
    + ["3.1", "3.2", "3.3", "3.4", "3.5", "4.0", "4.1", "4.2", "4.3", "4.4"]
    + ["4.5", "5.0", "5.1", "5.2", "5.3", "6.0", "6.2", "7.0", "7.1", "7.2"]
    + ["7.3", "7.4", "8.0"]
}
for k in ["3.0", "3.1", "3.2", "3.3", "3.4", "3.5", "4.0", "4.1", "4.2", "4.3", "4.4"]:
    SPECS_SPHINX[k]["pre_install"].extend(
        [
            "sed -i 's/Jinja2>=2.3/Jinja2<3.0/' setup.py",
            "sed -i 's/sphinxcontrib-applehelp/sphinxcontrib-applehelp<=1.0.7/' setup.py",
            "sed -i 's/sphinxcontrib-devhelp/sphinxcontrib-devhelp<=1.0.5/' setup.py",
            "sed -i 's/sphinxcontrib-qthelp/sphinxcontrib-qthelp<=1.0.6/' setup.py",
            "sed -i 's/alabaster>=0.7,<0.8/alabaster>=0.7,<0.7.12/' setup.py",
            "sed -i \"s/'packaging',/'packaging', 'markupsafe<=2.0.1',/\" setup.py",
        ]
    )
    if k in ["4.2", "4.3", "4.4"]:
        SPECS_SPHINX[k]["pre_install"].extend(
            [
                "sed -i 's/sphinxcontrib-htmlhelp>=2.0.0/sphinxcontrib-htmlhelp>=2.0.0,<=2.0.4/' setup.py",
                "sed -i 's/sphinxcontrib-serializinghtml>=1.1.5/sphinxcontrib-serializinghtml>=1.1.5,<=1.1.9/' setup.py",
            ]
        )
    elif k == "4.1":
        SPECS_SPHINX[k]["pre_install"].extend(
            [
                (
                    "grep -q 'sphinxcontrib-htmlhelp>=2.0.0' setup.py && "
                    "sed -i 's/sphinxcontrib-htmlhelp>=2.0.0/sphinxcontrib-htmlhelp>=2.0.0,<=2.0.4/' setup.py || "
                    "sed -i 's/sphinxcontrib-htmlhelp/sphinxcontrib-htmlhelp<=2.0.4/' setup.py"
                ),
                (
                    "grep -q 'sphinxcontrib-serializinghtml>=1.1.5' setup.py && "
                    "sed -i 's/sphinxcontrib-serializinghtml>=1.1.5/sphinxcontrib-serializinghtml>=1.1.5,<=1.1.9/' setup.py || "
                    "sed -i 's/sphinxcontrib-serializinghtml/sphinxcontrib-serializinghtml<=1.1.9/' setup.py"
                ),
            ]
        )
    else:
        SPECS_SPHINX[k]["pre_install"].extend(
            [
                "sed -i 's/sphinxcontrib-htmlhelp/sphinxcontrib-htmlhelp<=2.0.4/' setup.py",
                "sed -i 's/sphinxcontrib-serializinghtml/sphinxcontrib-serializinghtml<=1.1.9/' setup.py",
            ]
        )
for k in ["7.2", "7.3", "7.4", "8.0"]:
    SPECS_SPHINX[k]["pre_install"] += ["apt-get update && apt-get install -y graphviz"]

SPECS_ASTROPY = {
    k: {
        "python": "3.9",
        "pre_install": [
            "[[ -f pyproject.toml ]] && sed -i 's/cython>=3\\.0\\.0,[[:space:]]*<4/cython>=3.0.0,<3.1/' pyproject.toml",
        ],
        "install": "pip install -e .[dev_all]",
        "pip_packages": [
            "attrs==23.1.0",
            "exceptiongroup==1.1.3",
            "execnet==2.0.2",
            "hypothesis>=6.83",
            "iniconfig==2.0.0",
            "numpy==1.25.2",
            "packaging==23.1",
            "pluggy==1.3.0",
            "psutil==5.9.5",
            "pyerfa==2.0.0.3",
            "pytest-arraydiff==0.5.0",
            "pytest-astropy-header==0.2.2",
            "pytest-astropy==0.10.0",
            "pytest-cov==4.1.0",
            "pytest-doctestplus==1.0.0",
            "pytest-filter-subpackage==0.1.2",
            "pytest-mock==3.11.1",
            "pytest-openfiles==0.5.0",
            "pytest-remotedata==0.4.0",
            "pytest-xdist==3.3.1",
            "pytest==7.4.0",
            "PyYAML==6.0.1",
            "'setuptools<65'",
            "sortedcontainers==2.4.0",
            "tomli==2.0.1",
            "'cython<3.1'",
        ],
        "test_cmd": TEST_ASTROPY_PYTEST,
    }
    for k in ["4.1", "4.2", "4.3", "5.0", "5.1", "5.2", "v5.3"]
}
SPECS_ASTROPY["v5.3"]["python"] = "3.11"

SPECS_ASTROPY.update(
    {
        k: {
            "python": "3.7",
            "pre_install": [
                "[[ -f pyproject.toml ]] && sed -i 's/cython>=3\\.0\\.0,[[:space:]]*<4/cython>=3.0.0,<3.1/' pyproject.toml",
            ],
            "install": "pip install -e .[dev_all]",
            "pip_packages": [
                "attrs==23.1.0",
                "exceptiongroup==1.1.3",
                "execnet==2.0.2",
                "hypothesis>=6.83",
                "iniconfig==2.0.0",
                "numpy==1.18",
                "packaging==23.1",
                "pluggy",
                "psutil==5.9.5",
                "'pyerfa<2'",
                "pytest-arraydiff==0.5.0",
                "pytest-astropy-header==0.2.2",
                "pytest-astropy==0.10.0",
                "pytest-cov==4.1.0",
                "pytest-doctestplus==1.0.0",
                "pytest-filter-subpackage==0.1.2",
                "pytest-mock==3.11.1",
                "pytest-openfiles==0.5.0",
                "pytest-remotedata==0.4.0",
                "pytest-xdist==3.3.1",
                "pytest==7.4.0",
                "PyYAML==6.0.1",
                "sortedcontainers==2.4.0",
                "tomli==2.0.1",
                "'cython<3.1'",
                "'setuptools<60'",
            ],
            "test_cmd": TEST_ASTROPY_PYTEST,
        }
        for k in ["3.0", "3.1", "3.2"]
    }
)


SPECS_ASTROPY.update(
    {
        k: {
            "python": "3.6",
            "install": "pip install -e .[test]",
            "packages": "setuptools==38.2.4",
            "pip_packages": [
                "attrs==17.3.0",
                "exceptiongroup==0.0.0a0",
                "execnet==1.5.0",
                "hypothesis==3.44.2",
                "cython==0.27.3",
                "jinja2==2.10",
                "MarkupSafe==1.0",
                "numpy==1.16.0",
                "packaging==16.8",
                "pluggy==0.6.0",
                "psutil==5.4.2",
                "pyerfa==1.7.0",
                "pytest-arraydiff==0.1",
                "pytest-astropy-header==0.1",
                "pytest-astropy==0.2.1",
                "pytest-cov==2.5.1",
                "pytest-doctestplus==0.1.2",
                "pytest-filter-subpackage==0.1",
                "pytest-forked==0.2",
                "pytest-mock==1.6.3",
                "pytest-openfiles==0.2.0",
                "pytest-remotedata==0.2.0",
                "pytest-xdist==1.20.1",
                "pytest==3.3.1",
                "PyYAML==3.12",
                "sortedcontainers==1.5.9",
                "tomli==0.2.0",
            ],
            "test_cmd": TEST_ASTROPY_PYTEST,
        }
        for k in ["0.1", "0.2", "0.3", "0.4", "1.1", "1.2", "1.3"]
    }
)

set_setuptools_only = 'if [ -f pyproject.toml ]; then sed -i \'s/requires = \\["setuptools",/requires = \\["setuptools==68.0.0",/\' pyproject.toml; fi; '
set_markupsafe_version_and_setuptools = 'if [ -f pyproject.toml ]; then sed -i \'/"markupsafe[<>=]*[0-9.]*"/d; /requires = \\[/a\\            "markupsafe==2.0.1",\' pyproject.toml; sed -i \'s/requires = \\["setuptools",/requires = \\["setuptools==68.0.0",/\' pyproject.toml; fi; '

for k in ["3.2", "4.1", "4.2", "4.3"]:
    # Replace the MarkupSafe/jinja2 dependency?
    SPECS_ASTROPY[k].setdefault("pre_install", []).append(
        set_markupsafe_version_and_setuptools
    )

for k in ["5.0", "5.1", "5.2", "v5.3"]:
    SPECS_ASTROPY[k].setdefault("pre_install", []).append(set_setuptools_only)


SPECS_SYMPY = {}

SPECS_SYMPY.update(
    {
        k: {
            "python": "3.6",
            "packages": "mpmath flake8",
            "pip_packages": [
                "mpmath==1.3.0",
                "flake8-comprehensions",
                "gmpy2",
                "numpy",
            ],
            "install": "pip install -e .",
            "test_cmd": TEST_SYMPY,
            "distributed_test_cmd": TEST_SYMPY_DISTRIBUTED,
        }
        for k in ["1.0", "1.1", "1.2", "1.4", "1.5", "1.6"]
    }
)

SPECS_SYMPY.update(
    {
        k: {
            "python": "3.8",
            "packages": "mpmath flake8",
            "pip_packages": [
                "mpmath==1.3.0",
                "flake8-comprehensions",
                "gmpy2",
                "numpy",
            ],
            "install": "pip install -e .",
            "test_cmd": TEST_SYMPY,
            "distributed_test_cmd": TEST_SYMPY_DISTRIBUTED,
        }
        for k in ["1.6", "1.7", "1.8", "1.9", "1.10", "1.11", "1.12"]
    }
)

SPECS_SYMPY.update(
    {
        k: {
            "python": "3.10",
            "packages": "mpmath flake8",
            "install": "pip install -e .",
            "pip_packages": [
                "mpmath==1.3.0",
                "pytest",
                "pytest-xdist",
                "pytest-timeout",
                "pytest-split",
                "pytest-doctestplus",
                "hypothesis",
                "flake8-comprehensions",
                "gmpy2",
                "numpy",
            ],
            "test_cmd": TEST_SYMPY,
            "distributed_test_cmd": TEST_SYMPY_DISTRIBUTED,
        }
        for k in ["1.13", "1.14"]
    }
)

SPECS_PYLINT = {
    k: {
        "python": "3.9",
        "packages": "requirements.txt",
        "install": "pip install -e .",
        "test_cmd": TEST_PYTEST,
    }
    for k in [
        "2.10",
        "2.11",
        "2.13",
        "2.14",
        "2.15",
        "2.16",
        "2.17",
        "2.8",
        "2.9",
        "3.0",
        "3.1",
        "3.2",
        "3.3",
    ]
}
SPECS_PYLINT["2.8"]["pip_packages"] = ["pyenchant==3.2"]
SPECS_PYLINT["2.8"]["pre_install"] = [
    "apt-get update && apt-get install -y libenchant-2-dev hunspell-en-us"
]
SPECS_PYLINT.update(
    {
        k: {
            **SPECS_PYLINT[k],
            "pip_packages": ["astroid==3.0.0a6", "setuptools"],
        }
        for k in ["3.0"]
    }
)
for v in ["2.14", "2.15", "2.17", "3.0"]:
    SPECS_PYLINT[v]["nano_cpus"] = int(2e9)

SPECS_XARRAY = {
    k: {
        "python": "3.10",
        "packages": "environment.yml",
        "install": "pip install -e . --no-build-isolation",
        "pip_packages": [
            "numpy==1.24.0",
            "packaging==23.1",
            "pandas==1.5.3",
            "pytest==7.4.0",
            "python-dateutil==2.8.2",
            "pytz==2023.3",
            "six==1.16.0",
            "scipy==1.11.1",
            "'setuptools<=59.5.0'",
            "dask==2022.8.1",
            "gcsfs",
        ],
        "no_use_env": True,
        "test_cmd": TEST_PYTEST,
        "distributed_test_cmd": TEST_PYTEST_DISTRIBUTED,
    }
    for k in ["0.12", "0.18", "0.19", "0.20", "0.21", "2022.03", "2022.06", "2022.09"]
}

SPECS_XARRAY.update(
    {
        k: {
            "python": "3.10",
            "packages": "environment.yml",
            "install": "pip install -e . --no-build-isolation",
            "no_use_env": True,
            "test_cmd": TEST_PYTEST,
            "distributed_test_cmd": TEST_PYTEST_DISTRIBUTED,
            "pip_packages": [
                "gcsfs",
                "xarray-datatree",
            ],
        }
        for k in ["2024.05", "2023.07", "2023.03", "2023.04", "2023.05", "2023.06"]
    }
)

SPECS_SQLFLUFF = {
    k: {
        "python": "3.9",
        "packages": "requirements.txt",
        "install": "pip install -e .",
        "test_cmd": TEST_PYTEST,
    }
    for k in [
        "0.10",
        "0.11",
        "0.12",
        "0.13",
        "0.4",
        "0.5",
        "0.6",
        "0.8",
        "0.9",
        "1.0",
        "1.1",
        "1.2",
        "1.3",
        "1.4",
        "2.0",
        "2.1",
        "2.2",
    ]
}

SPECS_DBT_CORE = {
    k: {
        "python": "3.9",
        "packages": "requirements.txt",
        "install": "pip install -e .",
    }
    for k in [
        "0.13",
        "0.14",
        "0.15",
        "0.16",
        "0.17",
        "0.18",
        "0.19",
        "0.20",
        "0.21",
        "1.0",
        "1.1",
        "1.2",
        "1.3",
        "1.4",
        "1.5",
        "1.6",
        "1.7",
    ]
}

SPECS_PYVISTA = {
    k: {
        "python": "3.9",
        "pre_install": [
            "sudo apt update -y && sudo apt install -y python-qt4 libgl1-mesa-glx"
        ],
        "install": "pip install -e .",
        "pip_packages": ["pytest"],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["0.20", "0.21", "0.22", "0.23"]
}
SPECS_PYVISTA.update(
    {
        k: {
            "python": "3.9",
            "pre_install": [
                "sudo apt update -y && sudo apt install -y python-qt4 libgl1-mesa-glx"
            ],
            "packages": "requirements.txt",
            "install": "pip install -e .",
            "pip_packages": ["pytest"],
            "test_cmd": TEST_PYTEST,
        }
        for k in [
            "0.24",
            "0.25",
            "0.26",
            "0.27",
            "0.28",
            "0.29",
            "0.30",
            "0.31",
            "0.32",
            "0.33",
            "0.34",
            "0.35",
            "0.36",
            "0.37",
            "0.38",
            "0.39",
            "0.40",
            "0.41",
            "0.42",
            "0.43",
            "0.44",
        ]
    }
)

SPECS_PYVISTA.update(
    {
        k: {
            "python": "3.10",
            "pre_install": [
                "sudo apt update -y && sudo apt install -y python-qt4 libgl1-mesa-glx"
            ],
            "install": "pip install -e .[test]",
            "pip_packages": ["pytest"],
            "test_cmd": TEST_PYTEST,
        }
        for k in [
            "0.45",
            "0.46",
        ]
    }
)

SPECS_ASTROID = {
    k: {
        "python": "3.9",
        "install": "pip install -e .",
        "pip_packages": ["pytest"],
        "test_cmd": TEST_PYTEST,
    }
    for k in [
        "2.10",
        "2.12",
        "2.13",
        "2.14",
        "2.15",
        "2.16",
        "2.5",
        "2.6",
        "2.7",
        "2.8",
        "2.9",
        "3.0",
    ]
}

SPECS_MARSHMALLOW = {
    k: {
        "python": "3.9",
        "install": "pip install -e '.[dev]'",
        "test_cmd": TEST_PYTEST,
    }
    for k in [
        "2.18",
        "2.19",
        "2.20",
        "3.0",
        "3.1",
        "3.10",
        "3.11",
        "3.12",
        "3.13",
        "3.15",
        "3.16",
        "3.19",
        "3.2",
        "3.4",
        "3.8",
        "3.9",
    ]
}

SPECS_PVLIB = {
    k: {
        "python": "3.9",
        "install": "pip install -e .[all]",
        "packages": "pandas scipy",
        "pip_packages": ["jupyter", "ipython", "matplotlib", "pytest", "flake8"],
        "test_cmd": TEST_PYTEST,
    }
    for k in ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
}

SPECS_PYDICOM = {
    k: {
        "python": "3.6",
        "install": "pip install -e .",
        "packages": "numpy",
        "pip_packages": ["pytest"],
        "test_cmd": TEST_PYTEST,
    }
    for k in [
        "1.0",
        "1.1",
        "1.2",
        "1.3",
        "1.4",
        "2.0",
        "2.1",
        "2.2",
        "2.3",
        "2.4",
        "3.0",
    ]
}
SPECS_PYDICOM.update({k: {**SPECS_PYDICOM[k], "python": "3.8"} for k in ["1.4", "2.0"]})
SPECS_PYDICOM.update({k: {**SPECS_PYDICOM[k], "python": "3.9"} for k in ["2.1", "2.2"]})
SPECS_PYDICOM.update({k: {**SPECS_PYDICOM[k], "python": "3.10"} for k in ["2.3"]})
SPECS_PYDICOM.update(
    {k: {**SPECS_PYDICOM[k], "python": "3.11"} for k in ["2.4", "3.0"]}
)

SPECS_HUMANEVAL = {k: {"python": "3.9", "test_cmd": "python"} for k in ["1.0"]}

# Constants - Task Instance Instllation Environment
MAP_REPO_VERSION_TO_SPECS = {
    "astropy/astropy": SPECS_ASTROPY,
    "dbt-labs/dbt-core": SPECS_DBT_CORE,
    "django/django": SPECS_DJANGO,
    "matplotlib/matplotlib": SPECS_MATPLOTLIB,
    "marshmallow-code/marshmallow": SPECS_MARSHMALLOW,
    "mwaskom/seaborn": SPECS_SEABORN,
    "pallets/flask": SPECS_FLASK,
    "psf/requests": SPECS_REQUESTS,
    "pvlib/pvlib-python": SPECS_PVLIB,
    "pydata/xarray": SPECS_XARRAY,
    "pydicom/pydicom": SPECS_PYDICOM,
    "pylint-dev/astroid": SPECS_ASTROID,
    "pylint-dev/pylint": SPECS_PYLINT,
    "pytest-dev/pytest": SPECS_PYTEST,
    "pyvista/pyvista": SPECS_PYVISTA,
    "scikit-learn/scikit-learn": SPECS_SKLEARN,
    "sphinx-doc/sphinx": SPECS_SPHINX,
    "sqlfluff/sqlfluff": SPECS_SQLFLUFF,
    "swe-bench/humaneval": SPECS_HUMANEVAL,
    "sympy/sympy": SPECS_SYMPY,
}

# Constants - Repository Specific Installation Instructions
MAP_REPO_TO_INSTALL = {}


# Constants - Task Instance Requirements File Paths
MAP_REPO_TO_REQS_PATHS = {
    "dbt-labs/dbt-core": ["dev-requirements.txt", "dev_requirements.txt"],
    "django/django": ["tests/requirements/py3.txt"],
    "matplotlib/matplotlib": [
        "requirements/dev/dev-requirements.txt",
        "requirements/testing/travis_all.txt",
    ],
    "pallets/flask": ["requirements/dev.txt"],
    "pylint-dev/pylint": ["requirements_test.txt"],
    "pyvista/pyvista": ["requirements_test.txt", "requirements.txt"],
    "sqlfluff/sqlfluff": ["requirements_dev.txt"],
    "sympy/sympy": ["requirements-dev.txt"],
    "Project-MONAI/MONAI": ["requirements-dev.txt"],
    "HypothesisWorks/hypothesis": ["requirements/tools.txt"],
    "facebookresearch/hydra": ["requirements/dev.txt"],
    "numpy/numpy": ["test_requirements.txt", "requirements/test_requirements.txt"],
    "explosion/spacy": ["requirements.txt"],
    "modin-project/modin": ["requirements-dev.txt"],
}


# Constants - Task Instance environment.yml File Paths
MAP_REPO_TO_ENV_YML_PATHS = {
    "matplotlib/matplotlib": ["environment.yml"],
    "pydata/xarray": ["ci/requirements/environment.yml", "environment.yml"],
    "bokeh/bokeh": [
        # for v3
        "conda/environment-test-3.10.yml",
        # for v2
        "environment.yml",
        # for v1
    ],
    "modin-project/modin": ["environment-dev.yml"],
    "dask/dask": [
        "continuous_integration/environment-3.10.yaml",
        "continuous_integration/environment-3.9.yaml",
        "continuous_integration/environment-3.8.yaml",
        "continuous_integration/travis/travis-37.yaml",
        "ci/environment.yml",
    ],
    "spyder-ide/spyder": [
        "requirements/main.yml",
    ],
    "pandas-dev/pandas": [
        "environment.yml",
    ],
    "numpy/numpy": ["environment.yml"],
}

# Constants - Evaluation Keys
KEY_INSTANCE_ID = "instance_id"
KEY_MODEL = "model_name_or_path"
KEY_PREDICTION = "model_patch"


# Constants - Logging
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
APPLY_PATCH_PASS = ">>>>> Applied Patch"
INSTALL_FAIL = ">>>>> Init Failed"
INSTALL_PASS = ">>>>> Init Succeeded"
INSTALL_TIMEOUT = ">>>>> Init Timed Out"
RESET_FAILED = ">>>>> Reset Failed"
TESTS_ERROR = ">>>>> Tests Errored"
TESTS_FAILED = ">>>>> Some Tests Failed"
TESTS_PASSED = ">>>>> All Tests Passed"
TESTS_TIMEOUT = ">>>>> Tests Timed Out"


# Constants - Patch Types
class PatchType(Enum):
    PATCH_GOLD = "gold"
    PATCH_PRED = "pred"
    PATCH_PRED_TRY = "pred_try"
    PATCH_PRED_MINIMAL = "pred_minimal"
    PATCH_PRED_MINIMAL_TRY = "pred_minimal_try"
    PATCH_TEST = "test"

    def __str__(self):
        return self.value


# Constants - Miscellaneous
NON_TEST_EXTS = [
    ".json",
    ".png",
    "csv",
    ".txt",
    ".md",
    ".jpg",
    ".jpeg",
    ".pkl",
    ".yml",
    ".yaml",
    ".toml",
]
SWE_BENCH_URL_RAW = "https://raw.githubusercontent.com/"
USE_X86 = {
    "astropy__astropy-7973",
    "django__django-10087",
    "django__django-10097",
    "django__django-10213",
    "django__django-10301",
    "django__django-10316",
    "django__django-10426",
    "django__django-11383",
    "django__django-12185",
    "django__django-12497",
    "django__django-13121",
    "django__django-13417",
    "django__django-13431",
    "django__django-13447",
    "django__django-14155",
    "django__django-14164",
    "django__django-14169",
    "django__django-14170",
    "django__django-15180",
    "django__django-15199",
    "django__django-15280",
    "django__django-15292",
    "django__django-15474",
    "django__django-15682",
    "django__django-15689",
    "django__django-15695",
    "django__django-15698",
    "django__django-15781",
    "django__django-15925",
    "django__django-15930",
    "django__django-5158",
    "django__django-5470",
    "django__django-7188",
    "django__django-7475",
    "django__django-7530",
    "django__django-8326",
    "django__django-8961",
    "django__django-9003",
    "django__django-9703",
    "django__django-9871",
    "matplotlib__matplotlib-13983",
    "matplotlib__matplotlib-13984",
    "matplotlib__matplotlib-13989",
    "matplotlib__matplotlib-14043",
    "matplotlib__matplotlib-14471",
    "matplotlib__matplotlib-22711",
    "matplotlib__matplotlib-22719",
    "matplotlib__matplotlib-22734",
    "matplotlib__matplotlib-22767",
    "matplotlib__matplotlib-22815",
    "matplotlib__matplotlib-22835",
    "matplotlib__matplotlib-22865",
    "matplotlib__matplotlib-22871",
    "matplotlib__matplotlib-22883",
    "matplotlib__matplotlib-22926",
    "matplotlib__matplotlib-22929",
    "matplotlib__matplotlib-22931",
    "matplotlib__matplotlib-22945",
    "matplotlib__matplotlib-22991",
    "matplotlib__matplotlib-23031",
    "matplotlib__matplotlib-23047",
    "matplotlib__matplotlib-23049",
    "matplotlib__matplotlib-23057",
    "matplotlib__matplotlib-23088",
    "matplotlib__matplotlib-23111",
    "matplotlib__matplotlib-23140",
    "matplotlib__matplotlib-23174",
    "matplotlib__matplotlib-23188",
    "matplotlib__matplotlib-23198",
    "matplotlib__matplotlib-23203",
    "matplotlib__matplotlib-23266",
    "matplotlib__matplotlib-23267",
    "matplotlib__matplotlib-23288",
    "matplotlib__matplotlib-23299",
    "matplotlib__matplotlib-23314",
    "matplotlib__matplotlib-23332",
    "matplotlib__matplotlib-23348",
    "matplotlib__matplotlib-23412",
    "matplotlib__matplotlib-23476",
    "matplotlib__matplotlib-23516",
    "matplotlib__matplotlib-23562",
    "matplotlib__matplotlib-23563",
    "matplotlib__matplotlib-23573",
    "matplotlib__matplotlib-23740",
    "matplotlib__matplotlib-23742",
    "matplotlib__matplotlib-23913",
    "matplotlib__matplotlib-23964",
    "matplotlib__matplotlib-23987",
    "matplotlib__matplotlib-24013",
    "matplotlib__matplotlib-24026",
    "matplotlib__matplotlib-24088",
    "matplotlib__matplotlib-24111",
    "matplotlib__matplotlib-24149",
    "matplotlib__matplotlib-24177",
    "matplotlib__matplotlib-24189",
    "matplotlib__matplotlib-24224",
    "matplotlib__matplotlib-24250",
    "matplotlib__matplotlib-24257",
    "matplotlib__matplotlib-24265",
    "matplotlib__matplotlib-24334",
    "matplotlib__matplotlib-24362",
    "matplotlib__matplotlib-24403",
    "matplotlib__matplotlib-24431",
    "matplotlib__matplotlib-24538",
    "matplotlib__matplotlib-24570",
    "matplotlib__matplotlib-24604",
    "matplotlib__matplotlib-24619",
    "matplotlib__matplotlib-24627",
    "matplotlib__matplotlib-24637",
    "matplotlib__matplotlib-24691",
    "matplotlib__matplotlib-24749",
    "matplotlib__matplotlib-24768",
    "matplotlib__matplotlib-24849",
    "matplotlib__matplotlib-24870",
    "matplotlib__matplotlib-24912",
    "matplotlib__matplotlib-24924",
    "matplotlib__matplotlib-24970",
    "matplotlib__matplotlib-24971",
    "matplotlib__matplotlib-25027",
    "matplotlib__matplotlib-25052",
    "matplotlib__matplotlib-25079",
    "matplotlib__matplotlib-25085",
    "matplotlib__matplotlib-25122",
    "matplotlib__matplotlib-25126",
    "matplotlib__matplotlib-25129",
    "matplotlib__matplotlib-25238",
    "matplotlib__matplotlib-25281",
    "matplotlib__matplotlib-25287",
    "matplotlib__matplotlib-25311",
    "matplotlib__matplotlib-25332",
    "matplotlib__matplotlib-25334",
    "matplotlib__matplotlib-25340",
    "matplotlib__matplotlib-25346",
    "matplotlib__matplotlib-25404",
    "matplotlib__matplotlib-25405",
    "matplotlib__matplotlib-25425",
    "matplotlib__matplotlib-25430",
    "matplotlib__matplotlib-25433",
    "matplotlib__matplotlib-25442",
    "matplotlib__matplotlib-25479",
    "matplotlib__matplotlib-25498",
    "matplotlib__matplotlib-25499",
    "matplotlib__matplotlib-25515",
    "matplotlib__matplotlib-25547",
    "matplotlib__matplotlib-25551",
    "matplotlib__matplotlib-25565",
    "matplotlib__matplotlib-25624",
    "matplotlib__matplotlib-25631",
    "matplotlib__matplotlib-25640",
    "matplotlib__matplotlib-25651",
    "matplotlib__matplotlib-25667",
    "matplotlib__matplotlib-25712",
    "matplotlib__matplotlib-25746",
    "matplotlib__matplotlib-25772",
    "matplotlib__matplotlib-25775",
    "matplotlib__matplotlib-25779",
    "matplotlib__matplotlib-25785",
    "matplotlib__matplotlib-25794",
    "matplotlib__matplotlib-25859",
    "matplotlib__matplotlib-25960",
    "matplotlib__matplotlib-26011",
    "matplotlib__matplotlib-26020",
    "matplotlib__matplotlib-26024",
    "matplotlib__matplotlib-26078",
    "matplotlib__matplotlib-26089",
    "matplotlib__matplotlib-26101",
    "matplotlib__matplotlib-26113",
    "matplotlib__matplotlib-26122",
    "matplotlib__matplotlib-26160",
    "matplotlib__matplotlib-26184",
    "matplotlib__matplotlib-26208",
    "matplotlib__matplotlib-26223",
    "matplotlib__matplotlib-26232",
    "matplotlib__matplotlib-26249",
    "matplotlib__matplotlib-26278",
    "matplotlib__matplotlib-26285",
    "matplotlib__matplotlib-26291",
    "matplotlib__matplotlib-26300",
    "matplotlib__matplotlib-26311",
    "matplotlib__matplotlib-26341",
    "matplotlib__matplotlib-26342",
    "matplotlib__matplotlib-26399",
    "matplotlib__matplotlib-26466",
    "matplotlib__matplotlib-26469",
    "matplotlib__matplotlib-26472",
    "matplotlib__matplotlib-26479",
    "matplotlib__matplotlib-26532",
    "pydata__xarray-2905",
    "pydata__xarray-2922",
    "pydata__xarray-3095",
    "pydata__xarray-3114",
    "pydata__xarray-3151",
    "pydata__xarray-3156",
    "pydata__xarray-3159",
    "pydata__xarray-3239",
    "pydata__xarray-3302",
    "pydata__xarray-3305",
    "pydata__xarray-3338",
    "pydata__xarray-3364",
    "pydata__xarray-3406",
    "pydata__xarray-3520",
    "pydata__xarray-3527",
    "pydata__xarray-3631",
    "pydata__xarray-3635",
    "pydata__xarray-3637",
    "pydata__xarray-3649",
    "pydata__xarray-3677",
    "pydata__xarray-3733",
    "pydata__xarray-3812",
    "pydata__xarray-3905",
    "pydata__xarray-3976",
    "pydata__xarray-3979",
    "pydata__xarray-3993",
    "pydata__xarray-4075",
    "pydata__xarray-4094",
    "pydata__xarray-4098",
    "pydata__xarray-4182",
    "pydata__xarray-4184",
    "pydata__xarray-4248",
    "pydata__xarray-4339",
    "pydata__xarray-4356",
    "pydata__xarray-4419",
    "pydata__xarray-4423",
    "pydata__xarray-4442",
    "pydata__xarray-4493",
    "pydata__xarray-4510",
    "pydata__xarray-4629",
    "pydata__xarray-4683",
    "pydata__xarray-4684",
    "pydata__xarray-4687",
    "pydata__xarray-4695",
    "pydata__xarray-4750",
    "pydata__xarray-4758",
    "pydata__xarray-4759",
    "pydata__xarray-4767",
    "pydata__xarray-4802",
    "pydata__xarray-4819",
    "pydata__xarray-4827",
    "pydata__xarray-4879",
    "pydata__xarray-4911",
    "pydata__xarray-4939",
    "pydata__xarray-4940",
    "pydata__xarray-4966",
    "pydata__xarray-4994",
    "pydata__xarray-5033",
    "pydata__xarray-5126",
    "pydata__xarray-5131",
    "pydata__xarray-5180",
    "pydata__xarray-5187",
    "pydata__xarray-5233",
    "pydata__xarray-5362",
    "pydata__xarray-5365",
    "pydata__xarray-5455",
    "pydata__xarray-5580",
    "pydata__xarray-5662",
    "pydata__xarray-5682",
    "pydata__xarray-5731",
    "pydata__xarray-6135",
    "pydata__xarray-6386",
    "pydata__xarray-6394",
    "pydata__xarray-6400",
    "pydata__xarray-6461",
    "pydata__xarray-6548",
    "pydata__xarray-6598",
    "pydata__xarray-6599",
    "pydata__xarray-6601",
    "pydata__xarray-6721",
    "pydata__xarray-6744",
    "pydata__xarray-6798",
    "pydata__xarray-6804",
    "pydata__xarray-6823",
    "pydata__xarray-6857",
    "pydata__xarray-6882",
    "pydata__xarray-6889",
    "pydata__xarray-6938",
    "pydata__xarray-6971",
    "pydata__xarray-6992",
    "pydata__xarray-6999",
    "pydata__xarray-7003",
    "pydata__xarray-7019",
    "pydata__xarray-7052",
    "pydata__xarray-7089",
    "pydata__xarray-7101",
    "pydata__xarray-7105",
    "pydata__xarray-7112",
    "pydata__xarray-7120",
    "pydata__xarray-7147",
    "pydata__xarray-7150",
    "pydata__xarray-7179",
    "pydata__xarray-7203",
    "pydata__xarray-7229",
    "pydata__xarray-7233",
    "pydata__xarray-7347",
    "pydata__xarray-7391",
    "pydata__xarray-7393",
    "pydata__xarray-7400",
    "pydata__xarray-7444",
    "pytest-dev__pytest-10482",
    "scikit-learn__scikit-learn-10198",
    "scikit-learn__scikit-learn-10297",
    "scikit-learn__scikit-learn-10306",
    "scikit-learn__scikit-learn-10331",
    "scikit-learn__scikit-learn-10377",
    "scikit-learn__scikit-learn-10382",
    "scikit-learn__scikit-learn-10397",
    "scikit-learn__scikit-learn-10427",
    "scikit-learn__scikit-learn-10428",
    "scikit-learn__scikit-learn-10443",
    "scikit-learn__scikit-learn-10452",
    "scikit-learn__scikit-learn-10459",
    "scikit-learn__scikit-learn-10471",
    "scikit-learn__scikit-learn-10483",
    "scikit-learn__scikit-learn-10495",
    "scikit-learn__scikit-learn-10508",
    "scikit-learn__scikit-learn-10558",
    "scikit-learn__scikit-learn-10577",
    "scikit-learn__scikit-learn-10581",
    "scikit-learn__scikit-learn-10687",
    "scikit-learn__scikit-learn-10774",
    "scikit-learn__scikit-learn-10777",
    "scikit-learn__scikit-learn-10803",
    "scikit-learn__scikit-learn-10844",
    "scikit-learn__scikit-learn-10870",
    "scikit-learn__scikit-learn-10881",
    "scikit-learn__scikit-learn-10899",
    "scikit-learn__scikit-learn-10908",
    "scikit-learn__scikit-learn-10913",
    "scikit-learn__scikit-learn-10949",
    "scikit-learn__scikit-learn-10982",
    "scikit-learn__scikit-learn-10986",
    "scikit-learn__scikit-learn-11040",
    "scikit-learn__scikit-learn-11042",
    "scikit-learn__scikit-learn-11043",
    "scikit-learn__scikit-learn-11151",
    "scikit-learn__scikit-learn-11160",
    "scikit-learn__scikit-learn-11206",
    "scikit-learn__scikit-learn-11235",
    "scikit-learn__scikit-learn-11243",
    "scikit-learn__scikit-learn-11264",
    "scikit-learn__scikit-learn-11281",
    "scikit-learn__scikit-learn-11310",
    "scikit-learn__scikit-learn-11315",
    "scikit-learn__scikit-learn-11333",
    "scikit-learn__scikit-learn-11346",
    "scikit-learn__scikit-learn-11391",
    "scikit-learn__scikit-learn-11496",
    "scikit-learn__scikit-learn-11542",
    "scikit-learn__scikit-learn-11574",
    "scikit-learn__scikit-learn-11578",
    "scikit-learn__scikit-learn-11585",
    "scikit-learn__scikit-learn-11596",
    "scikit-learn__scikit-learn-11635",
    "scikit-learn__scikit-learn-12258",
    "scikit-learn__scikit-learn-12421",
    "scikit-learn__scikit-learn-12443",
    "scikit-learn__scikit-learn-12462",
    "scikit-learn__scikit-learn-12471",
    "scikit-learn__scikit-learn-12486",
    "scikit-learn__scikit-learn-12557",
    "scikit-learn__scikit-learn-12583",
    "scikit-learn__scikit-learn-12585",
    "scikit-learn__scikit-learn-12625",
    "scikit-learn__scikit-learn-12626",
    "scikit-learn__scikit-learn-12656",
    "scikit-learn__scikit-learn-12682",
    "scikit-learn__scikit-learn-12704",
    "scikit-learn__scikit-learn-12733",
    "scikit-learn__scikit-learn-12758",
    "scikit-learn__scikit-learn-12760",
    "scikit-learn__scikit-learn-12784",
    "scikit-learn__scikit-learn-12827",
    "scikit-learn__scikit-learn-12834",
    "scikit-learn__scikit-learn-12860",
    "scikit-learn__scikit-learn-12908",
    "scikit-learn__scikit-learn-12938",
    "scikit-learn__scikit-learn-12961",
    "scikit-learn__scikit-learn-12973",
    "scikit-learn__scikit-learn-12983",
    "scikit-learn__scikit-learn-12989",
    "scikit-learn__scikit-learn-13010",
    "scikit-learn__scikit-learn-13013",
    "scikit-learn__scikit-learn-13017",
    "scikit-learn__scikit-learn-13046",
    "scikit-learn__scikit-learn-13087",
    "scikit-learn__scikit-learn-13124",
    "scikit-learn__scikit-learn-13135",
    "scikit-learn__scikit-learn-13142",
    "scikit-learn__scikit-learn-13143",
    "scikit-learn__scikit-learn-13157",
    "scikit-learn__scikit-learn-13165",
    "scikit-learn__scikit-learn-13174",
    "scikit-learn__scikit-learn-13221",
    "scikit-learn__scikit-learn-13241",
    "scikit-learn__scikit-learn-13253",
    "scikit-learn__scikit-learn-13280",
    "scikit-learn__scikit-learn-13283",
    "scikit-learn__scikit-learn-13302",
    "scikit-learn__scikit-learn-13313",
    "scikit-learn__scikit-learn-13328",
    "scikit-learn__scikit-learn-13333",
    "scikit-learn__scikit-learn-13363",
    "scikit-learn__scikit-learn-13368",
    "scikit-learn__scikit-learn-13392",
    "scikit-learn__scikit-learn-13436",
    "scikit-learn__scikit-learn-13439",
    "scikit-learn__scikit-learn-13447",
    "scikit-learn__scikit-learn-13454",
    "scikit-learn__scikit-learn-13467",
    "scikit-learn__scikit-learn-13472",
    "scikit-learn__scikit-learn-13485",
    "scikit-learn__scikit-learn-13496",
    "scikit-learn__scikit-learn-13497",
    "scikit-learn__scikit-learn-13536",
    "scikit-learn__scikit-learn-13549",
    "scikit-learn__scikit-learn-13554",
    "scikit-learn__scikit-learn-13584",
    "scikit-learn__scikit-learn-13618",
    "scikit-learn__scikit-learn-13620",
    "scikit-learn__scikit-learn-13628",
    "scikit-learn__scikit-learn-13641",
    "scikit-learn__scikit-learn-13704",
    "scikit-learn__scikit-learn-13726",
    "scikit-learn__scikit-learn-13779",
    "scikit-learn__scikit-learn-13780",
    "scikit-learn__scikit-learn-13828",
    "scikit-learn__scikit-learn-13864",
    "scikit-learn__scikit-learn-13877",
    "scikit-learn__scikit-learn-13910",
    "scikit-learn__scikit-learn-13915",
    "scikit-learn__scikit-learn-13933",
    "scikit-learn__scikit-learn-13960",
    "scikit-learn__scikit-learn-13974",
    "scikit-learn__scikit-learn-13983",
    "scikit-learn__scikit-learn-14012",
    "scikit-learn__scikit-learn-14024",
    "scikit-learn__scikit-learn-14053",
    "scikit-learn__scikit-learn-14067",
    "scikit-learn__scikit-learn-14087",
    "scikit-learn__scikit-learn-14092",
    "scikit-learn__scikit-learn-14114",
    "scikit-learn__scikit-learn-14125",
    "scikit-learn__scikit-learn-14141",
    "scikit-learn__scikit-learn-14237",
    "scikit-learn__scikit-learn-14309",
    "scikit-learn__scikit-learn-14430",
    "scikit-learn__scikit-learn-14450",
    "scikit-learn__scikit-learn-14458",
    "scikit-learn__scikit-learn-14464",
    "scikit-learn__scikit-learn-14496",
    "scikit-learn__scikit-learn-14520",
    "scikit-learn__scikit-learn-14544",
    "scikit-learn__scikit-learn-14591",
    "scikit-learn__scikit-learn-14629",
    "scikit-learn__scikit-learn-14704",
    "scikit-learn__scikit-learn-14706",
    "scikit-learn__scikit-learn-14710",
    "scikit-learn__scikit-learn-14732",
    "scikit-learn__scikit-learn-14764",
    "scikit-learn__scikit-learn-14806",
    "scikit-learn__scikit-learn-14869",
    "scikit-learn__scikit-learn-14878",
    "scikit-learn__scikit-learn-14890",
    "scikit-learn__scikit-learn-14894",
    "scikit-learn__scikit-learn-14898",
    "scikit-learn__scikit-learn-14908",
    "scikit-learn__scikit-learn-14983",
    "scikit-learn__scikit-learn-14999",
    "scikit-learn__scikit-learn-15028",
    "scikit-learn__scikit-learn-15084",
    "scikit-learn__scikit-learn-15086",
    "scikit-learn__scikit-learn-15094",
    "scikit-learn__scikit-learn-15096",
    "scikit-learn__scikit-learn-15100",
    "scikit-learn__scikit-learn-15119",
    "scikit-learn__scikit-learn-15120",
    "scikit-learn__scikit-learn-15138",
    "scikit-learn__scikit-learn-15393",
    "scikit-learn__scikit-learn-15495",
    "scikit-learn__scikit-learn-15512",
    "scikit-learn__scikit-learn-15524",
    "scikit-learn__scikit-learn-15535",
    "scikit-learn__scikit-learn-15625",
    "scikit-learn__scikit-learn-3840",
    "scikit-learn__scikit-learn-7760",
    "scikit-learn__scikit-learn-8554",
    "scikit-learn__scikit-learn-9274",
    "scikit-learn__scikit-learn-9288",
    "scikit-learn__scikit-learn-9304",
    "scikit-learn__scikit-learn-9775",
    "scikit-learn__scikit-learn-9939",
    "sphinx-doc__sphinx-11311",
    "sphinx-doc__sphinx-7910",
    "sympy__sympy-12812",
    "sympy__sympy-14248",
    "sympy__sympy-15222",
    "sympy__sympy-19201",
}


# mypy and python versoin are tightly coupled
SPECS_MYPY = {
    k: {
        "pre_install": [
            "git submodule update --init mypy/typeshed || true",
        ],
        "python": "3.12",
        # see https://github.com/python/mypy/mypy/test/testcheck.py#L39
        "install": "pip install -r test-requirements.txt; pip install -e .; hash -r",
        "test_cmd": "pytest -rA -k",
    }
    for k in ["1.7", "1.8", "1.9", "1.10", "1.11"]
}

SPECS_MYPY.update(
    # Working
    {
        k: {
            "pre_install": [
                "git submodule update --init mypy/typeshed || true",
            ],
            "python": "3.11",
            "install": "pip install -r test-requirements.txt; pip install -e .; hash -r",
            "test_cmd": "pytest -n0 -rA -k",
        }
        for k in ["1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6"]
    }
)

SPECS_MYPY.update(
    # Working
    {
        k: {
            "pre_install": [
                "git submodule update --init mypy/typeshed || true",
            ],
            "python": "3.10",
            "install": "pip install -r test-requirements.txt; pip install -e .; pip install pytest pytest-xdist; hash -r",
            "test_cmd": "pytest -n0 -rA -k",
        }
        for k in ["0.990", "0.980", "0.970", "0.960", "0.950", "0.940"]
    }
)
SPECS_MYPY.update(
    # Working
    {
        k: {
            "pre_install": [
                "git submodule update --init mypy/typeshed || true",
                "sed -i '1i types-typing-extensions==3.7.3' test-requirements.txt",
            ],
            "python": "3.9",
            # types-typing-extensions is yanked, we need to set a specific version manually
            "install": "pip install -r test-requirements.txt; pip install -e .; pip install pytest pytest-xdist; hash -r;",
            "test_cmd": "pytest -n0 -rA -k",
        }
        for k in ["0.920", "0.910", "0.820", "0.810", "0.800"]
    }
)

# python/mypy versions prior to 0.800 are hard to install, skipping for now
# SPECS_MYPY.update(
#     {
#         k: {
#             "pre_install": [
#                 "apt-get -y update && apt-get -y upgrade && apt-get install -y gcc",
#                 "apt-get install libxml2-dev libxslt1-dev"
#             ],
#             "python": "3.8",
#                 "apt-get update && apt-get install -y libenchant-2-dev hunspell-en-us"
#             "install": "pip install -r test-requirements.txt; pip install -e .; pip install pytest; hash -r;",
#             "test_cmd": "pytest -rA -k"
#         }
#         for k in []
#     }
# )
# mypy 0.2, with 14 instances, is too old and requires deprecated python 3.4.
# not worth it for now


MAP_REPO_VERSION_TO_SPECS.update({"python/mypy": SPECS_MYPY})


TEST_MOTO = "pytest -n0 -rA"
SPECS_MOTO = {
    k: {
        "python": "3.12",
        # see https://github.com/getmoto/moto/blob/master/CONTRIBUTING.md
        "install": "make init",
        "test_cmd": TEST_MOTO,
    }
    for k in [
        "0.4",
        "1.0",
        "1.2",
        "1.3",
        "2.0",
        "2.1",
        "2.2",
        "2.3",
        "3.0",
        "3.1",
        "4.0",
        "4.1",
        "4.2",
        "5.0",
    ]
}
MAP_REPO_VERSION_TO_SPECS.update({"getmoto/moto": SPECS_MOTO})

TEST_CONAN = "pytest -n0 -rA"


# extra args before cython3.0 https://github.com/conan-io/conan/issues/14319
SPECS_CONAN = {
    k: {
        "python": "3.10",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y build-essential cmake",
        ],
        "install": "echo 'cython<3' > /tmp/constraint.txt; export PIP_CONSTRAINT=/tmp/constraint.txt; pip install -r conans/requirements.txt; pip install -r conans/requirements_server.txt; pip install -r conans/requirements_dev.txt ",
        "eval_commands": [
            "export PYTHONPATH=${PYTHONPATH:-}:$(pwd)",
        ],
        "test_cmd": TEST_CONAN,
    }
    for k in [
        "1.33",
        "1.34",
        "1.36",
        "2.0",
        "1.35",
        "1.37",
        "1.46",
        "1.38",
        "1.39",
        "1.40",
        "1.41",
        "1.42",
        "1.45",
        "1.43",
        "1.44",
        "1.47",
        "1.48",
        "1.49",
        "1.50",
        "1.51",
        "1.52",
        "1.53",
        "1.55",
        "1.54",
        "1.57",
        "1.58",
        "1.59",
    ]
}

SPECS_CONAN.update(
    {
        k: {
            "python": "3.10",
            "pre_install": [
                "apt-get -y update && apt-get -y upgrade && apt-get install -y build-essential cmake",
            ],
            "install": "pip install -r conans/requirements.txt; pip install -r conans/requirements_server.txt; pip install -r conans/requirements_dev.txt ",
            "eval_commands": [
                "export PYTHONPATH=${PYTHONPATH:-}:$(pwd)",
            ],
            "test_cmd": TEST_CONAN,
        }
        for k in ["2.1", "1.60", "1.61", "1.62", "2.2", "2.3", "2.4"]
    }
)
MAP_REPO_VERSION_TO_SPECS.update({"conan-io/conan": SPECS_CONAN})


TEST_DASK = 'pytest -n0 -rA -W "ignore::DeprecationWarning"  --continue-on-collection-errors --color=no'
# pandas 2.0 is a breaking change, need to separate from there
SPECS_DASK = dict()


SPECS_DASK.update(
    {
        k: {
            # "python": "3.10",
            "env_patches": [
                # dask installs latest dask from github in environment.yml
                # remove these lines and delay dask installation later
                "sed -i '/^[[:space:]]*- pip:/,$d' environment.yml",
                "sed -i '/^\s*-\s*pip\s*$/d' environment.yml",
                "sed -i 's/@master/@main/g' environment.yml",
                r"sed -E -i'' 's/^([[:space:]]*-[[:space:]]*)sqlalchemy[[:alnum:][:punct:]]*/\1sqlalchemy<=1.4.26/' environment.yml",
                "conda config --set channel_priority strict",
            ],
            "packages": "environment.yml",
            "install": "pip install --no-deps -e .",
            "test_cmd": TEST_DASK,
        }
        for k in [
            "2.1",
            "2.2",
            "2.4",
            "2.6",
            "2.7",
            "2.9",
            "2.10",
            "2.11",
            "2.12",
            "2.13",
            "2.14",
            "2.15",
            "2.16",
            "2.17",
            "2.18",
            "2.19",
            "2.20",
            "2.21",
            "2.22",
            "2.23",
            "2.25",
            "2.26",
            "2.27",
            "2.28",
            "2.29",
            "2.30",
            "2020.12",
            "2021.01",
            "2021.02",
            "2021.03",
            "2021.04",
            "2021.05",
            "2021.06",
            "2021.07",
            "2021.08",
            "2021.09",
            "2021.10",
            "2021.11",
            "2021.12",
            "2022.01",
            "2022.02",
            "2022.03",
            "2022.04",
            "2022.05",
            "2022.6",
            "2022.7",
            "2022.8",
            "2022.9",
            "2022.10",
            "2022.11",
            "2022.12",
            "2023.1",
            "2023.2",
            "2023.3",
            "2023.4",
            "2023.5",
            "2023.6",
            "2023.7",
            "2023.8",
            "2023.9",
            "2023.10",
            "2023.11",
            "2023.12",
            "2024.1",
            "2024.2",
            "2024.3",
            "2024.4",
            "2024.5",
            "2024.6",
            "2024.7",
            "2024.8",
            "2024.9",
            "2024.10",
            "2024.11",
            "2024.12",
            "2025.1",
            "2025.2",
            "2025.3",
        ]
    }
)
for k in [
    "2.1",
    "2.2",
    "2.4",
    "2.6",
    "2.7",
    "2.9",
    "2.10",
    "2.11",
    "2.12",
    "2.13",
    "2.14",
    "2.15",
    "2.16",
    "2.17",
    "2.18",
    "2.19",
    "2.20",
    "2.21",
    "2.22",
    "2.23",
    "2.25",
    "2.26",
    "2.27",
    "2.28",
    "2.29",
    "2.30",
    "2020.12",
    "2021.01",
    "2021.02",
    "2021.03",
    "2021.04",
    "2021.05",
    "2021.06",
    "2021.07",
    "2021.08",
    "2021.09",
    "2021.10",
    "2021.11",
    "2021.12",
    "2022.01",
    "2022.02",
    "2022.03",
    "2022.04",
    "2022.05",
    "2022.6",
    "2022.7",
    "2022.8",
    "2022.9",
    "2022.10",
    "2022.11",
    "2022.12",
]:
    SPECS_DASK[k]["pip_packages"] = ["'pytest<8'", "'numpy<2'", "'pandas<2'"]

for k in [
    "2024.1",
    "2024.2",
    "2024.3",
    "2024.4",
    "2024.5",
    "2024.6",
    "2024.7",
    "2024.8",
    "2024.9",
    "2024.10",
    "2024.11",
    "2024.12",
    "2025.1",
    "2025.2",
    "2025.3",
]:
    SPECS_DASK[k]["pip_packages"] = ["dask-expr"]

# #"'numba==0.56.4'", "'pyarrow<8'", "'numpy<1.20'"

# for k in ['2020.12', '2021.01', '2021.02', '2021.03', '2021.04', '2021.05', '2021.06', '2021.07', '2021.08', '2021.09', '2021.10', '2021.11', '2021.12', '2022.01', '2022.02', '2022.03', '2022.04', '2022.05', '2022.6', '2022.7', '2022.8', '2022.9', '2022.10', '2022.11', '2022.12', '2023.1', '2023.2', '2023.3', '2023.4', '2023.5', '2023.6', '2023.7', '2023.8', '2023.9', '2023.10', '2023.11', '2023.12', '2024.1', '2024.2', '2024.3', '2024.4', '2024.5', '2024.6', '2024.7', '2024.8', "2024.9", "2024.10", "2024.11", "2024.12", "2025.1", "2025.2", "2025.3"]:
#     SPECS_DASK[k]["pip_packages"] = ["'pytest<9'", "xarray", "fastavro", "gmpy2"]

# for k in ['2.1', '2.2', '2.4', '2.6', '2.7', '2.9', '2.10', '2.11', '2.12', '2.13', '2.14', '2.15', ]:
#     SPECS_DASK[k]["pip_packages"] = ["'pytest<9'", "xarray", "fastavro", ]

MAP_REPO_VERSION_TO_SPECS.update({"dask/dask": SPECS_DASK})

TEST_MONAI = "pytest -rA "
SPECS_MONAI = {
    k: {
        "python": "3.8",
        # monai's requirements.txt calls each other, hard to standardize in swefficiency constant format
        # "packages": "requirements.txt",
        # "install": "pip install -U pip; pip install scikit-build; pip install types-pkg-resources==0.1.3 pytest; pip install -U -r requirements-dev.txt; python setup.py develop;",
        # "env_patches": [
        #     # monai installs itself from git
        #     # remove these lines and delay dask installation later
        #     "sed -i '/^git+https:\/\/github.com\/Project-MONAI\//d' ~/requirements.txt"
        # ],
        "install": "sed -i '/^git+https:\/\/github.com\/Project-MONAI\//d' requirements-dev.txt; pip install types-pkg-resources==0.1.3 pytest; pip install -r requirements-dev.txt;python setup.py develop;",
        "test_cmd": TEST_MONAI,
    }
    for k in [
        "0.1",
        "0.2",
        "0.3",
        "0.4",
        "0.5",
        "0.6",
        "0.7",
        "0.8",
        "0.9",
        "0.11",
        "0.105",
        "1.0",
        "1.1",
        "1.2",
        "1.3",
    ]
}
MAP_REPO_VERSION_TO_SPECS.update({"Project-MONAI/MONAI": SPECS_MONAI})

# dvc
TEST_DVC = "pytest -rA"
SPECS_DVC = {
    k: {
        "python": "3.10",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y cmake",
            # fix moto dev version missing issue
            "[ -f setup.py ] && sed -E -i 's/moto==([0-9]+\.[0-9]+\.[0-9]+)\.dev[0-9]+/moto==\\1/' setup.py",
            # fix pyarrow version issue
            "[ -f setup.py ] && sed -i 's/pyarrow==0.15.1/pyarrow==0.16/' setup.py"
            # fix boto version conflict
            "[ -f setup.py ] && sed -i 's/boto3==1.9.115/boto3==1.9.201/' setup.py",
        ],
        "install": 'pip install --upgrade pip wheel GitPython; pip install "cython<3.0.0" && pip install --no-build-isolation pyyaml==5.4.1; pip install git+https://github.com/iterative/mock-ssh-server.git || true; pip install -r tests/requirements.txt || true; pip install -r test-requirements.txt || true; pip install -e ".[tests,dev,all_remotes,all,testing]";',
        "test_cmd": TEST_DVC,
    }
    for k in [
        "0.1",
        "0.8",
        "0.9",
        "0.12",
        "0.13",
        "0.14",
        "0.15",
        "0.16",
        "0.17",
        "0.18",
        "0.19",
        "0.20",
        "0.21",
        "0.22",
        "0.23",
        "0.24",
        "0.27",
        "0.28",
        "0.29",
        "0.30",
        "0.31",
        "0.32",
        "0.33",
        "0.34",
        "0.35",
        "0.40",
        "0.41",
        "0.50",
        "0.51",
        "0.52",
        "0.53",
        "0.54",
        "0.55",
        "0.56",
        "0.57",
        "0.58",
        "0.59",
        "0.60",
        "0.61",
        "0.62",
        "0.63",
        "0.65",
        "0.66",
        "0.68",
        "0.69",
        "0.70",
        "0.71",
        "0.74",
        "0.75",
        "0.76",
        "0.77",
        "0.78",
        "0.80",
        "0.81",
        "0.82",
        "0.83",
        "0.84",
        "0.85",
        "0.86",
        "0.87",
        "0.88",
        "0.89",
        "0.90",
        "0.91",
        "0.92",
        "0.93",
        "1.0",
        "1.1",
        "1.2",
        "1.3",
        "1.4",
        "1.5",
        "1.6",
        "1.7",
        "1.8",
        "1.9",
        "1.10",
        "1.11",
        "2.0",
        "2.1",
        "2.2",
        "2.3",
        "2.4",
        "2.5",
        "2.6",
        "2.7",
        "2.8",
        "2.9",
        "2.10",
        "2.11",
        "2.12",
        "2.13",
        "2.15",
        "2.17",
        "2.19",
        "2.20",
        "2.21",
        "2.22",
        "2.23",
        "2.24",
        "2.27",
        "2.28",
        "2.30",
        "2.33",
        "2.34",
        "2.35",
        "2.38",
        "2.41",
        "2.43",
        "2.44",
        "2.45",
        "2.46",
        "2.48",
        "2.50",
        "2.51",
        "2.52",
        "2.54",
        "2.55",
        "2.56",
        "2.57",
        "2.58",
        "3.0",
        "3.1",
        "3.2",
        "3.3",
        "3.4",
        "3.5",
        "3.6",
        "3.10",
        "3.11",
        "3.12",
        "3.13",
        "3.14",
        "3.15",
        "3.17",
        "3.19",
        "3.23",
        "3.24",
        "3.28",
        "3.29",
        "3.36",
        "3.37",
        "3.38",
        "3.43",
        "3.47",
        "3.48",
        "3.49",
    ]
}
for k in [
    "0.1",
    "0.8",
    "0.9",
    "0.12",
    "0.13",
    "0.14",
    "0.15",
    "0.16",
    "0.17",
    "0.18",
    "0.19",
    "0.20",
    "0.21",
    "0.22",
    "0.23",
    "0.24",
    "0.27",
    "0.28",
    "0.29",
    "0.30",
    "0.31",
    "0.32",
    "0.33",
    "0.34",
    "0.35",
    "0.40",
    "0.41",
    "0.50",
    "0.51",
    "0.52",
    "0.53",
    "0.54",
    "0.55",
    "0.56",
    "0.57",
    "0.58",
    "0.59",
    "0.60",
    "0.61",
    "0.62",
    "0.63",
    "0.65",
    "0.66",
    "0.68",
    "0.69",
    "0.70",
    "0.71",
    "0.74",
    "0.75",
    "0.76",
    "0.77",
    "0.78",
    "0.80",
    "0.81",
    "0.82",
    "0.83",
    "0.84",
    "0.85",
    "0.86",
    "0.87",
    "0.88",
    "0.89",
    "0.90",
    "0.91",
    "0.92",
    "0.93",
]:
    SPECS_DVC[k]["python"] = "3.8"
    SPECS_DVC[k]["install"] += ' pip install "numpy<=1.20";'
    # pytest 8 breaks pytest-lazy-fixture
    SPECS_DVC[k]["install"] += ' pip install "pytest<8";'

for k in [
    "1.0",
    "1.1",
    "1.2",
    "1.3",
    "1.4",
    "1.5",
    "1.6",
    "1.7",
    "1.8",
    "1.9",
    "1.10",
    "1.11",
    "2.0",
    "2.1",
    "2.2",
    "2.3",
    "2.4",
    "2.5",
    "2.6",
    "2.7",
    "2.8",
    "2.9",
    "2.10",
    "2.11",
    "2.12",
    "2.13",
    "2.15",
    "2.17",
    "2.19",
    "2.20",
    "2.21",
    "2.22",
    "2.23",
    "2.24",
    "2.27",
    "2.28",
    "2.30",
    "2.33",
    "2.34",
    "2.35",
    "2.38",
    "2.41",
    "2.43",
    "2.44",
    "2.45",
    "2.46",
    "2.48",
    "2.50",
    "2.51",
    "2.52",
    "2.54",
    "2.55",
    "2.56",
    "2.57",
    "2.58",
    "3.0",
    "3.1",
    "3.2",
    "3.3",
]:
    SPECS_DVC[k]["python"] = "3.9"
    SPECS_DVC[k]["install"] += ' pip install "numpy<=1.20";'
    # pytest 8 breaks pytest-lazy-fixture
    SPECS_DVC[k]["install"] += ' pip install "pytest<8";'
MAP_REPO_VERSION_TO_SPECS.update({"iterative/dvc": SPECS_DVC})

# bokeh
# https://docs.bokeh.org/en/latest/docs/dev_guide/setup.html
TEST_BOKEH = "pytest -rA -n0"
# for k in ['0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.10', '0.11', '0.12', '0.13', '0.1181316818', '1.0', '1.1', '1.2', '1.3', '1.4', '2.0', '2.1', '2.3', '2.4', '3.0', '3.3', '3.4', '3.5']
SPECS_BOKEH = {
    k: {
        "python": "3.10",
        "packages": "environment.yml",
        "pre_install": [
            "cd bokehjs && npm install --location=global npm && npm ci && cd ../"
        ],
        "install": "pip install -e .; pip install bokeh_sampledata;",
        "test_cmd": TEST_BOKEH,
    }
    for k in ["3.0", "3.3", "3.4", "3.5"]
}

SPECS_BOKEH.update(
    {
        k: {
            "python": "3.8",
            "packages": "environment.yml",
            "env_patches": [
                ": \"${CONDA_MKL_INTERFACE_LAYER_BACKUP:=''}\"",
                # "sed -i 's/  - setuptools/  - setuptools<66/' environment.yml"
            ],
            "pre_install": [
                "cd bokehjs && npm install --location=global npm && npm ci && cd ../",
            ],
            "install": 'pip install "setuptools<66" "jinja2<3.1"; printf "1\n" | python setup.py develop; bokeh sampledata;',
            "test_cmd": TEST_BOKEH,
        }
        for k in ["2.0", "2.1", "2.3", "2.4"]
    }
)

SPECS_BOKEH.update(
    {
        k: {
            "python": "3.8",
            "packages": "environment.yml",
            "env_patches": [
                ": \"${CONDA_MKL_INTERFACE_LAYER_BACKUP:=''}\"",
                # "sed -i 's/  - setuptools/  - setuptools<66/' environment.yml"
            ],
            "pre_install": [
                "cd bokehjs && npm install --location=global npm && npm ci && cd ../",
            ],
            "install": 'pip install "setuptools<66" "jinja2<3.1"; printf "1\n" | python setup.py develop; bokeh sampledata;',
            "test_cmd": TEST_BOKEH,
        }
        for k in [
            "0.4",
            "0.5",
            "0.6",
            "0.7",
            "0.8",
            "0.9",
            "0.10",
            "0.11",
            "0.12",
            "0.13",
            "0.1181316818",
            "1.0",
            "1.1",
            "1.2",
            "1.3",
            "1.4",
        ]
    }
)
MAP_REPO_VERSION_TO_SPECS.update({"bokeh/bokeh": SPECS_BOKEH})


# modin
# https://github.com/modin-project/modin/pull/7312
# numpy2.0 is supported in June 2024, we will need to restrict numpy version to be before 2.0
# Skipping '0.1', '0.2', '0.3', '0.4', '0.6',
TEST_MODIN = "pytest -n0 -rA"
SPECS_MODIN = {
    k: {
        "python": "3.6",
        # "pre_install": [
        #     "apt-get -y update && apt-get -y upgrade && apt-get install -y libpq-dev",
        #     'export PATH="$HOME/.local/bin:$PATH"',
        #     'export PATH="/usr/bin:$PATH"',
        # ],
        "packages": "environment.yml",
        # "packages": "requirements.txt",
        "install": "pip install -e . --no-build-isolation;",
        # "install": "pip install 'numpy<2.0'; pip install --upgrade Cython; pip install -r requirements-dev.txt; pip install -e .",
        "test_cmd": TEST_MODIN,
    }
    for k in [
        "0.7",
        "0.8",
        "0.9",
        "0.10",
        "0.11",
        "0.12",
        "0.13",
        "0.14",
        "0.15",
        "0.16",
        "0.17",
        "0.18",
        "0.19",
        "0.20",
        "0.21",
        "0.22",
        "0.23",
        "0.24",
        "0.25",
        "0.26",
        "0.27",
        "0.28",
        "0.29",
        "0.30",
        "0.31",
        "0.32",
    ]
}
for k in ["0.7", "0.8", "0.9", "0.10", "0.11", "0.12", "0.13", "0.14", "0.15"]:
    SPECS_MODIN[k]["python"] = "3.6"

for k in ["0.16", "0.17", "0.18", "0.19", "0.20", "0.21"]:
    SPECS_MODIN[k]["python"] = "3.8"

for k in [
    "0.22",
    "0.23",
    "0.24",
    "0.25",
    "0.26",
    "0.27",
    "0.28",
    "0.29",
    "0.30",
    "0.31",
    "0.32",
]:
    SPECS_MODIN[k]["python"] = "3.10"

MAP_REPO_VERSION_TO_SPECS.update({"modin-project/modin": SPECS_MODIN})

# spyder
# https://github.com/spyder-ide/spyder/blob/master/CONTRIBUTING.md
TEST_SPYDER = "pytest -n0 -rA"
SPECS_SPYDER = {
    k: {
        "python": "3.9",
        "packages": "environment.yml",
        "pre_install": [
            "conda env update --file requirements/linux.yml",
            "conda env update --file requirements/tests.yml",
        ],
        "install": "pip install -e .;",
        # "install": "pip install 'numpy<2.0'; pip install --upgrade Cython; pip install -r requirements-dev.txt; pip install -e .",
        "test_cmd": TEST_SPYDER,
    }
    for k in []
}

MAP_REPO_VERSION_TO_SPECS.update({"spyder-ide/spyder": SPECS_SPYDER})

# hypothesis
# https://github.com/HypothesisWorks/hypothesis/blob/eaafdfcad3f362e75746863472101d4cfabbc33d/CONTRIBUTING.rst
TEST_HYPOTHESIS = "pytest -n0 -rA --tb=no --no-header"
SPECS_HYPOTHESIS = {
    k: {
        "python": "3.10",
        "packages": "requirements.txt",  # this installs tools.txt
        "install": "pip install -r requirements/test.txt; pip install -e hypothesis-python/;",
        "test_cmd": TEST_HYPOTHESIS,
    }
    for k in [
        "3.55",
        "3.61",
        "3.60",
        "3.59",
        "3.63",
        "3.66",
        "3.67",
        "3.68",
        "3.69",
        "3.70",
        "5.1",
        "5.5",
        "5.24",
        "5.6",
        "5.9",
        "5.8",
        "5.10",
        "5.12",
        "5.15",
        "5.20",
        "5.23",
        "5.36",
        "5.32",
        "5.33",
        "5.38",
        "5.41",
        "5.42",
        "5.43",
        "5.47",
        "6.1",
        "6.4",
        "6.6",
        "6.8",
        "6.14",
        "6.13",
        "6.18",
        "6.21",
        "6.24",
        "6.28",
        "6.29",
        "3.73",
        "3.71",
        "3.75",
        "3.79",
        "3.82",
        "3.85",
        "3.88",
        "4.0",
        "3.86",
        "4.2",
        "4.4",
        "4.15",
        "4.12",
        "4.14",
        "4.18",
        "4.23",
        "4.24",
        "4.26",
        "4.32",
        "4.38",
        "4.40",
        "4.42",
        "4.46",
        "4.44",
        "4.50",
        "4.54",
        "4.55",
        "5.2",
        "5.4",
        "6.30",
        "6.31",
        "6.36",
        "6.40",
        "6.43",
        "6.53",
        "6.45",
        "6.46",
        "6.47",
        "6.50",
        "6.54",
        "6.59",
        "6.62",
        "6.66",
        "6.71",
        "6.74",
        "6.77",
        "6.81",
        "6.87",
        "6.88",
        "6.93",
        "6.98",
        "6.99",
        "6.100",
        "6.102",
    ]
}
for k in [
    "3.55",
    "3.61",
    "3.60",
    "3.59",
    "3.63",
    "3.66",
    "3.67",
    "3.68",
    "3.69",
    "3.70",
    "5.1",
    "5.5",
    "5.24",
    "5.6",
    "5.9",
    "5.8",
    "5.10",
    "5.12",
    "5.15",
    "5.20",
    "5.23",
    "5.36",
    "5.32",
    "5.33",
    "5.38",
    "5.41",
    "5.42",
    "5.43",
    "5.47",
    "6.1",
    "6.4",
    "6.6",
    "6.8",
    "6.14",
    "6.13",
    "6.18",
    "6.21",
    "6.24",
    "6.28",
    "6.29",
    "3.73",
    "3.71",
    "3.75",
    "3.79",
    "3.82",
    "3.85",
    "3.88",
    "4.0",
    "3.86",
    "4.2",
    "4.4",
    "4.15",
    "4.12",
    "4.14",
    "4.18",
    "4.23",
    "4.24",
    "4.26",
    "4.32",
    "4.38",
    "4.40",
    "4.42",
    "4.46",
    "4.44",
    "4.50",
    "4.54",
    "4.55",
    "5.2",
    "5.4",
    "6.30",
    "6.31",
]:
    SPECS_HYPOTHESIS[k]["python"] = "3.9"

MAP_REPO_VERSION_TO_SPECS.update({"HypothesisWorks/hypothesis": SPECS_HYPOTHESIS})

# pydantic
# https://docs.pydantic.dev/latest/contributing/
# TEST_PYDANTIC = 'export PATH="$HOME/.local/bin:$PATH"; pdm run coverage run -m pytest -rA --tb=short --no-header'
TEST_PYDANTIC = "pytest -rA --tb=short -vv -o console_output_style=classic --no-header"
SPECS_PYDANTIC = {
    k: {
        "python": "3.8",
        "pre_install": [
            "apt-get update && apt-get install -y locales",
            "apt-get install -y pipx",
            "pipx ensurepath",
            # well, this in fact uses python 3.10 as default by pipx
            "pipx install pdm",
            'export PATH="$HOME/.local/bin:$PATH"',
            "which python",
            "python --version",
        ],
        "install": 'export PATH="$HOME/.local/bin:$PATH"; pdm add pre-commit; make install;',
        "test_cmd": TEST_PYDANTIC,
    }
    for k in [
        "0.2",
        "0.41",
        "0.4",
        "0.6",
        "0.9",
        "0.10",
        "0.11",
        "0.13",
        "0.14",
        "0.151",
        "0.15",
        "0.17",
        "0.18",
        "0.201",
        "0.20",
        "0.24",
        "0.27",
        "0.29",
        "1.01",
        "0.32",
        "1.4",
        "1.31",
        "1.41",
        "1.51",
        "1.5",
        "1.71",
        "1.6",
        "1.7",
        "1.8",
        "1.9",
        "1.10",
        "2.0",
        "2.01",
        "2.02",
        "2.03",
        "2.04",
        "2.6",
        "2.5",
        "2.4",
        "2.7",
    ]
}

for k in [
    "0.2",
    "0.41",
    "0.4",
    "0.6",
    "0.9",
    "0.10",
    "0.11",
    "0.13",
    "0.14",
    "0.151",
    "0.15",
    "0.17",
    "0.18",
    "0.201",
    "0.20",
    "0.24",
    "0.27",
    "0.29",
    "1.01",
    "0.32",
    "1.4",
    "1.31",
    "1.41",
    "1.51",
    "1.5",
    "1.71",
    "1.6",
    "1.7",
    "1.8",
    "1.9",
    "1.10",
]:
    # not working yet
    SPECS_PYDANTIC[k]["pre_install"] = [
        "apt-get update && apt-get install -y locales",
        "apt-get install -y pipx",
        "pipx ensurepath",
        # well, this in fact uses python 3.10 as default by pipx
        "pipx install pdm  --python python3.7",
        'export PATH="$HOME/.local/bin:$PATH"',
        "which python",
        "python --version",
    ]
    SPECS_PYDANTIC[k]["python"] = "3.7"

MAP_REPO_VERSION_TO_SPECS.update({"pydantic/pydantic": SPECS_PYDANTIC})

# pandas
# https://pandas.pydata.org/pandas-docs/dev/development/contributing_environment.html
TEST_PANDAS = (
    TEST_PYTEST_VERBOSE + ' -m "not slow and not network and not db and not single_cpu"'
)
TEST_PANDAS_DISTRIBUTED = (
    TEST_PYTEST_DISTRIBUTED
    + ' -m "not slow and not network and not db and not single_cpu" '
)  # Pandas tests are just slow, generaly pytest-xdist is available in all.

SPECS_PANDAS = {
    k: {
        "packages": "environment.yml",
        "env_patches": [
            "sed -i '/^[[:space:]]*- pip:/,$d' environment.yml",
            # Use strict channel priority.
            # "conda config --set channel_priority strict",
            # Pinning cython to exact equality.
            # "sed -i 's/cython>=/cython==/g' environment.yml",
            "sed -Ei 's/(cython>=[^,<]+)(.*)/\\1,<3\\2/g' environment.yml",
            # Remove hypothesis due to backwards compatibility and conda not doing this correctly, install later with pip.
            "sed -i '/^\s*-\s*hypothesis[<=>\s]*.*$/d' environment.yml",
            # "grep -qE '^\\s*-\\s*(defaults|nodefaults)\\s*$' environment.yml || sed -Ei '/^channels:/ { n; :a; /^\\s*-/ { s/$/\\n  - nodefaults/; b }; n; ba }' environment.yml",
            "awk '/^  # code checks$/ { skip=1; next } /^  # document/ { skip=1; next } /^  # web/ { skip=1; next } skip && /^[[:space:]]*-/ { next } skip { skip=0 } { print }' environment.yml > environment.yml.new && mv environment.yml.new environment.yml",
            "sed -i -E 's/^([[:space:]]*-[[:space:]]*c-blosc2)[^#]*/\\1/' environment.yml",  # Remove this hard dependency and rely on numpy to do it.
            # Remove pycodestyle lines.
            "sed -i '/pycodestyle/d' environment.yml",
            "sed -i '/sphinx/d' environment.yml",
            "sed -i '/mypy/d' environment.yml",
            # Remove pyreadstat.
            "sed -i '/pyreadstat/d' environment.yml",
        ],
        "pre_install": [
            "git remote add upstream https://github.com/pandas-dev/pandas.git",
            "git fetch upstream --tags",
            'export CFLAGS="$CFLAGS -fwrapv -fno-strict-overflow -fno-strict-aliasing -Wno-error -Wno-array-bounds -Wno-deprecated-declarations -Wno-sign-compare -Wno-strict-prototypes -Wno-cpp -Wno-unused-function -Wno-missing-prototypes -Wno-implicit-function-declaration -Wno-return-type"',
        ],
        # "install": "pip install -ve . --no-build-isolation -Ceditable-verbose=true; pip uninstall pytest-qt -y; pip install hypothesis;", pip uninstall -y pandas;
        "install": "pip install -e . --no-build-isolation; pip uninstall pytest-qt -y;",
        "test_cmd": TEST_PANDAS,
        "distributed_test_cmd": TEST_PANDAS_DISTRIBUTED,
        "pip_packages": [
            "hypothesis",
            "pytest",
        ],
    }
    for k in [
        "0.23",
        "0.24",
        "0.25",
        "0.26",
        "0.6",
        "1.0",
        "1.1",
        "1.2",
        "1.3",
        "1.4",
        "1.5",
        "2.0",
        "2.1",
        "2.2",
        "3.0",
    ]
}
# We ignore ['0.10', '0.11', '0.12', '0.13', '0.14', '0.15', '0.16', '0.17', '0.18', '0.19', '0.20', '0.21', '0.22']

for k in ["1.1", "1.2", "1.3", "1.4", "1.5", "2.0", "2.1"]:
    # numpy 2 is supported in pandas 2.2
    SPECS_PANDAS[k]["pre_install"].extend(["pip install 'numpy<2'; "])
    # SPECS_PANDAS[k]['install'] = "pip install 'numpy<2'; " + SPECS_PANDAS[k]['install']
    # SPECS_PANDAS[k]['env_patches'].append("sed -i 's/python=3[[:space:]]*$/python=3.8/g' environment.yml")

for k in ["1.0"]:
    # numpy 2 is supported in pandas 2.2
    SPECS_PANDAS[k]["pre_install"].extend(["pip install 'numpy<1.20.0'; "])
    # SPECS_PANDAS[k]['install'] = SPECS_PANDAS[k]['install']
    # SPECS_PANDAS[k]['env_patches'].append("sed -i 's/python=3[[:space:]]*$/python=3.8/g' environment.yml")

for k in ["0.23", "0.24", "0.25", "0.26"]:
    # Replace occurrences of "  - python=3" with "  - python=3.7"
    SPECS_PANDAS[k]["pre_install"].extend(
        ["pip install 'numpy<=1.17.4' 'hypothesis<4'; "]
    )
    SPECS_PANDAS[k][
        "install"
    ] = "pip install -e . --no-build-isolation --no-use-pep517; pip uninstall pytest-qt -y;"

for k in ["0.23", "0.24", "0.25", "0.26", "1.0"]:  # Removed 1.0, 1.1, 1.2
    SPECS_PANDAS[k]["pre_install"].append("pip install 'matplotlib<3.3'; ")

for k in ["0.23", "0.24", "0.25", "0.26", "1.0"]:  # Removed 1.0, 1.1, 1.2
    SPECS_PANDAS[k]["env_patches"].extend(
        [
            "sed -i 's/python=3$/python<3.8/' environment.yml",
            "cat environment.yml",
        ]
    )

for k in ["1.1", "1.2", "1.3", "1.4", "1.5"]:
    SPECS_PANDAS[k]["env_patches"].extend(
        [
            "sed -i 's/python=3$/python<=3.10/' environment.yml",
            "cat environment.yml",
        ]
    )

MAP_REPO_VERSION_TO_SPECS.update({"pandas-dev/pandas": SPECS_PANDAS})

# hydra
TEST_HYDRA = "pytest -rA --tb=long"
SPECS_HYDRA = {
    k: {
        "python": "3.8",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y openjdk-17-jdk openjdk-17-jre",
        ],
        "install": "pip install -r requirements/dev.txt; pip install -e .;",
        "test_cmd": TEST_HYDRA,
    }
    for k in ["0.1", "0.9", "0.10", "0.11", "0.12", "1.0", "1.1", "1.2", "1.3", "1.4"]
}
for k in ["0.1", "0.9", "0.10", "0.11", "0.12", "1.0", "1.1", "1.2"]:
    # fix omegaconf pip version issue
    SPECS_HYDRA[k]["install"] = (
        '{ tail -n1 requirements/requirements.txt | grep -q "." && echo ""; } >> requirements/requirements.txt; echo "pip==24.0" >> requirements/requirements.txt;'
        + 'pip install "pip==24.0"; '
        + SPECS_HYDRA[k]["install"]
    )
    # isort is moved to PyCQA now
    SPECS_HYDRA[k]["install"] = (
        "sed -i 's|isort@git+git://github.com/timothycrosley/isort|isort@git+https://github.com/timothycrosley/isort|g' requirements/dev.txt; "
        + SPECS_HYDRA[k]["install"]
    )
MAP_REPO_VERSION_TO_SPECS.update({"facebookresearch/hydra": SPECS_HYDRA})


SPECS_NUMPY = {}

TEST_NUMPY = TEST_PYTEST
TEST_NUMPY_DISTRIBUTED = TEST_PYTEST_DISTRIBUTED

for k in ["1.15", "1.16"]:
    SPECS_NUMPY[k] = {
        "python": "3.6",
        "packages": "'cython==0.28.4' 'pytest' swig",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y libopenblas-dev",
            r"""[ -f pytest.ini ] && sed -i.bak '/^\[pytest\]/a\markers =\n    slow: marks tests as slow' pytest.ini""",
            "pip install 'setuptools>=48,<60'",
        ],
        "install": "pip install --no-build-isolation -e .",
        # "install": "python setup.py build_ext --inplace -j 4",
        "test_cmd": TEST_NUMPY,
        "distributed_test_cmd": TEST_NUMPY_DISTRIBUTED,
        "pip_packages": [
            "'cython==0.28.3'",
            "pytest-env",
            "pytest-xdist",
            "'setuptools>=48,<60'",
        ],
    }

for k in ["1.17"]:
    SPECS_NUMPY[k] = {
        "python": "3.6",
        "packages": "'cython==0.29.2' 'pytest' swig",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y libopenblas-dev",
            r"""[ -f pytest.ini ] && sed -i.bak '/^\[pytest\]/a\markers =\n    slow: marks tests as slow' pytest.ini""",
            "pip install 'setuptools>=48,<60'",
        ],
        "install": "pip install --no-build-isolation -e .",
        # "install": "python setup.py build_ext --inplace -j 4",
        "test_cmd": TEST_NUMPY,
        "distributed_test_cmd": TEST_NUMPY_DISTRIBUTED,
        "pip_packages": [
            "'cython==0.29.13'",
            "pytest-env",
            "pytest-xdist",
            "'setuptools>=48,<60'",
        ],
    }

for k in ["1.18", "1.19"]:
    SPECS_NUMPY[k] = {
        "python": "3.7",
        "packages": "'cython<3' pytest swig",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y libopenblas-dev && rm -rf branding/",
            "pip install 'setuptools>=48,<60'",
        ],
        "install": "pip install --no-build-isolation --no-use-pep517 -e .",
        "test_cmd": TEST_NUMPY,
        "distributed_test_cmd": TEST_NUMPY_DISTRIBUTED,
        "pip_packages": [
            "'cython<3'",
            "'hypothesis==5.3.0'",
            "pytest-env",
            "pytest-xdist",
            "'setuptools>=48,<60'",
        ],
    }

for k in ["1.20", "1.21", "1.22", "1.23", "1.24"]:
    SPECS_NUMPY[k] = {
        "python": "3.9",
        "packages": "requirements.txt",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y libopenblas-dev && rm -rf branding/ && git submodule update --init",
            "pip install 'setuptools>=48,<60'",
        ],
        "install": "pip install --no-build-isolation -e .",
        "test_cmd": TEST_NUMPY,
        "distributed_test_cmd": TEST_NUMPY_DISTRIBUTED,
        "pip_packages": [
            "'cython>=0.29.34,<3'",
            "'setuptools==60'",
            "pytest-xdist",
            "meson",
            "meson-python",
            "ninja",
        ],
    }

for k in ["1.25"]:
    SPECS_NUMPY[k] = {
        "packages": "environment.yml",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y libopenblas-dev && rm -rf branding/ && git submodule update --init",
        ],
        "install": "pip install --no-build-isolation -e .",
        "test_cmd": TEST_NUMPY,
        "distributed_test_cmd": TEST_NUMPY_DISTRIBUTED,
        "pip_packages": [
            "'cython==3.0.6'",
            "setuptools",
            "pytest-xdist",
        ],
    }

for k in ["1.26"]:
    SPECS_NUMPY[k] = {
        "packages": "environment.yml",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y libopenblas-dev && rm -rf branding/ && git submodule update --init",
        ],
        "install": "pip install --no-build-isolation -e .",
        "test_cmd": TEST_NUMPY,
        "distributed_test_cmd": TEST_NUMPY_DISTRIBUTED,
        "pip_packages": [
            "'cython==3.0.6'",
            "setuptools",
            "pytest-xdist",
        ],
    }

for k in ["2.0", "2.1", "2.2"]:
    SPECS_NUMPY[k] = {
        "packages": "environment.yml",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y libopenblas-dev && git submodule update --init",
            "python -m ensurepip --upgrade",
            "pip install --upgrade setuptools",
        ],
        "install": "pip install -e . --no-build-isolation",
        "test_cmd": TEST_NUMPY,
        "distributed_test_cmd": TEST_NUMPY_DISTRIBUTED,
        "pip_packages": [
            "pip",
            "wheel",
            "pytest-xdist",
        ],
    }

MAP_REPO_VERSION_TO_SPECS.update({"numpy/numpy": SPECS_NUMPY})


SPECS_SCIPY = {}

# Note that one must provide the file name as a module file here?
SCIPY_OLD_RUNTEST = "runtests.py -t"
TEST_SCIPY = TEST_PYTEST
TEST_SCIPY_DISTRIBUTED = TEST_PYTEST_DISTRIBUTED


SCIPY_FLAGS = """\
export CFLAGS="-fcommon"; export FFLAGS="-fcommon -fallow-argument-mismatch -fPIC"; export FCFLAGS="-fcommon -fallow-argument-mismatch -fPIC"
""".strip()

for k in ["0.18", "0.19"]:
    SPECS_SCIPY[k] = {
        "python": "3.6",
        "packages": "'cython==0.28.5'",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y libopenblas-dev",
            "rm -f pyproject.toml",  # Remove this because its causing issues.
            SCIPY_FLAGS,
        ],
        "install": f"pip install --no-build-isolation -e .",
        "test_cmd": SCIPY_OLD_RUNTEST,
        "pip_packages": [
            "'cython<0.29.18'",
            "'setuptools<60'",
            "'numpy==1.12.1'",
            "wheel",
            "nose",
            "pytest-env",
            "pytest-xdist",
            "pytest-timeout",
        ],
    }

for k in ["1.0", "1.1"]:
    SPECS_SCIPY[k] = {
        "python": "3.7",
        "packages": "'pytest<8'",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y libopenblas-dev && git submodule update --init",
            SCIPY_FLAGS,
        ],
        "install": f"pip install --no-build-isolation -e . --no-use-pep517",
        "test_cmd": TEST_SCIPY,
        "pip_packages": [
            "'cython<=0.29.18'",
            "'setuptools<60'",
            "'numpy==1.15.4'",
            "'pybind11==2.4.3'",
            "wheel",
            "pytest-env",
            "pytest-xdist",
            "pytest-timeout",
            "pillow",
        ],
    }

for k in ["1.2"]:
    SPECS_SCIPY[k] = {
        "python": "3.7",
        "packages": "'pytest<8'",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y libopenblas-dev && git submodule update --init",
            SCIPY_FLAGS,
        ],
        "install": f"pip install --no-build-isolation -e .",
        "test_cmd": TEST_SCIPY,
        "pip_packages": [
            "'cython<=0.29.18'",
            "'setuptools<60'",
            "'numpy==1.15.4'",
            "'pybind11==2.4.3'",
            "wheel",
            "pytest-env",
            "pytest-xdist",
            "pytest-timeout",
            "pillow",
        ],
    }


for k in ["1.3", "1.4", "1.5"]:
    SPECS_SCIPY[k] = {
        "python": "3.7",
        "packages": "'pytest<8' suitesparse swig scikit-umfpack",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y libopenblas-dev liblapack-dev libopenblas-dev libsuitesparse-dev libc6-dev && git submodule update --init",
            SCIPY_FLAGS,
        ],
        "install": f"pip install --no-build-isolation -e .",
        "test_cmd": TEST_SCIPY,
        "pip_packages": [
            "'cython<=0.29.20'",
            "'setuptools<60'",
            "'numpy<=1.17.3'",
            "'pybind11==2.4.3'",
            "wheel",
            "pytest-cov",
            "pytest-xdist",
            "pytest-env",
            "pytest-timeout",
            "pillow",
            "mpmath",
            "matplotlib",
            "xarray",
        ],
    }


for k in ["1.6", "1.7"]:
    SPECS_SCIPY[k] = {
        "python": "3.8",
        "packages": "'pytest<8' 'meson-python<0.9.0' suitesparse swig scikit-umfpack",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y libopenblas-dev liblapack-dev libopenblas-dev libsuitesparse-dev libc6-dev && git submodule update --init",
            SCIPY_FLAGS,
        ],
        "install": f"pip install --no-build-isolation -e .",
        "test_cmd": TEST_SCIPY,
        "pip_packages": [
            "'cython>=0.29.18,<3.0'",
            "'setuptools<60'",
            "'numpy==1.17.3'",
            "'pybind11==2.10.1'",
            "'wheel<0.38.0'",
            "'pythran<0.15'",
            # Test dependencies.
            "pytest-cov",
            "pytest-xdist",
            "pytest-timeout",
            "pytest-env",
            "asv",
            "mpmath",
            "gmpy2",
            "threadpoolctl",
            "'pyproject-metadata==0.6.0'",
            "xarray",
        ],
    }

for k in ["1.8"]:
    SPECS_SCIPY[k] = {
        "python": "3.8",
        "packages": "pytest 'meson==0.63' 'meson-python==0.8.0' suitesparse swig scikit-umfpack",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y libopenblas-dev liblapack-dev libopenblas-dev libsuitesparse-dev patchelf libc6-dev && git submodule update --init",
            SCIPY_FLAGS,
        ],
        "install": f"pip install --no-build-isolation -e .",
        "test_cmd": TEST_SCIPY,
        "pip_packages": [
            "'cython>=0.29.21,<3.0'",
            "'numpy==1.19.5'",
            "'pybind11<2.11.1'",
            "'pythran<0.15'",
            "'setuptools<60'",
            "'pyproject-metadata==0.6.0'",
            "wheel",
            # Test dependencies.
            "pytest-cov",
            "pytest-xdist",
            "pytest-timeout",
            "pytest-env",
            "asv",
            "mpmath",
            "gmpy2",
            "threadpoolctl",
            "xarray",
        ],
    }

for k in ["1.9"]:
    SPECS_SCIPY[k] = {
        "python": "3.8",
        "packages": "pytest 'meson==0.64' 'meson-python==0.12.0' suitesparse swig scikit-umfpack",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y libopenblas-dev liblapack-dev libopenblas-dev libsuitesparse-dev patchelf libc6-dev && git submodule update --init",
            SCIPY_FLAGS,
        ],
        "install": f"pip install --no-build-isolation -e .",
        "test_cmd": TEST_SCIPY,
        "pip_packages": [
            "'cython>=0.29.32,<3.0'",
            "'numpy==1.22.2'",
            "'pybind11<2.11.1'",
            "'pythran<0.15'",
            "'setuptools<60'",
            "'pyproject-metadata==0.6.0'",
            "'wheel<0.39.0'",
            # Test dependencies.
            "pytest-cov",
            "pytest-xdist",
            "pytest-timeout",
            "pytest-env",
            "asv",
            "mpmath",
            "gmpy2",
            "threadpoolctl",
            "xarray",
        ],
    }


for k in ["1.10", "1.11"]:
    SPECS_SCIPY[k] = {
        "python": "3.10",
        "packages": "pytest 'numpy>=1.22.4,<1.29.0' 'meson-python==0.14.0' ninja suitesparse swig scikit-umfpack 'pythran<0.18.0' pybind11 'cython<3' pkg-config openblas",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y libopenblas-dev liblapack-dev libopenblas-dev libsuitesparse-dev libc6-dev && git submodule update --init",
            SCIPY_FLAGS,
        ],
        "install": f"pip install --no-build-isolation -e .",
        "test_cmd": TEST_SCIPY,
        "pip_packages": [
            "'setuptools<65'",
            "'wheel==0.41.2'",
            # Test dependencies,
            "pytest-cov",
            "pytest-timeout",
            "pytest-xdist",
            "asv",
            "mpmath",
            "gmpy2",
            "threadpoolctl",
            "pooch",
            "hypothesis",
            "xarray",
        ],
    }


for k in ["1.12", "1.13"]:
    SPECS_SCIPY[k] = {
        "python": "3.10",
        "packages": "pytest 'numpy>=1.22.4,<1.29.0' 'meson-python==0.15.0' ninja suitesparse swig scikit-umfpack 'pythran<0.18.0' pybind11 'cython>=0.29.35,<=3.1' pkg-config openblas",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y libopenblas-dev liblapack-dev libopenblas-dev libsuitesparse-dev libc6-dev && git submodule update --init",
            SCIPY_FLAGS,
        ],
        "install": f"pip install --no-build-isolation -e .",
        "test_cmd": TEST_SCIPY,
        "pip_packages": [
            "'setuptools==66.1.1'",
            # Test dependencies,
            "pytest-cov",
            "pytest-timeout",
            "pytest-xdist",
            "pytest-env",
            "asv",
            "mpmath",
            "gmpy2",
            "threadpoolctl",
            "scikit-umfpack",
            "pooch",
            "hypothesis",
            "xarray",
        ],
    }


for k in ["1.14", "1.15"]:
    SPECS_SCIPY[k] = {
        "python": "3.11",
        "packages": "pytest 'numpy>=1.23.5,<2.3' compilers 'meson-python' ninja suitesparse swig scikit-umfpack 'pythran<0.18.0' pybind11 'cython<=3.1' pkg-config openblas",
        "pre_install": [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y libopenblas-dev liblapack-dev libopenblas-dev libsuitesparse-dev libc6-dev && git submodule update --init",
            SCIPY_FLAGS,
        ],
        "install": f"pip install --no-build-isolation -e .",
        "test_cmd": TEST_SCIPY,
        "pip_packages": [
            "'setuptools==66.1.1'",
            # Test dependencies,
            "pytest",
            "pytest-cov",
            "pytest-timeout",
            "pytest-xdist",
            "pytest-env",
            "asv",
            "mpmath",
            "gmpy2",
            "threadpoolctl",
            "scikit-umfpack",
            "pooch",
            "'hypothesis>=6.30'",
            "'array-api-strict>=2.0'",
            "xarray",
        ],
    }

MAP_REPO_VERSION_TO_SPECS.update({"scipy/scipy": SPECS_SCIPY})

SPECS_SPACY = {}
for k in ["2.3", "3.0", "3.1", "3.2", "3.3", "3.4"]:
    SPECS_SPACY[k] = {
        "python": "3.6",
        "packages": "requirements.txt",
        "env_patches": [
            "sed -i 's/cython>=/cython==/g' $HOME/requirements.txt",
            "cat $HOME/requirements.txt",
        ],
        "pre_install": ["conda install murmurhash"],
        "install": "pip install --no-build-isolation -e .",
        "test_cmd": TEST_PYTEST,
        "pip_packages": ["pytest", "'setuptools<60'", "'wheel'", "'cython<0.29'"],
    }

# for k in ['3.5', '3.6', '3.7']:
#     SPECS_SPACY[k] = {
#         "python": "3.7",
#         "packages": "requirements.txt",
#         "env_patches": [
#             "sed -i 's/cython>=/cython==/g' $HOME/requirements.txt",
#             "cat $HOME/requirements.txt",
#         ],
#         "install": "pip install --no-build-isolation -e .",
#         "test_cmd": TEST_PYTEST,
#         "pip_packages": [
#             "pytest",
#             "'setuptools<65'",
#             "wheel",
#         ]
#     }

# for k in ['3.8', '4.0']:
#     SPECS_SPACY[k] = {
#         "python": "3.9",
#         "packages": "requirements.txt",
#         "env_patches": [
#             "sed -i 's/cython>=/cython==/g' $HOME/requirements.txt",
#             "cat $HOME/requirements.txt",
#         ],
#         "install": "pip install --no-build-isolation -e .",
#         "test_cmd": TEST_PYTEST,
#         "pip_packages": [
#             "pytest",
#             "setuptools",
#             "wheel",
#         ]
#     }


MAP_REPO_VERSION_TO_SPECS.update({"explosion/spacy": SPECS_SPACY})

# TODO: not tested
SPECS_STATSMODELS = {}
for k in ["0.8", "0.9", "0.10", "0.11"]:
    SPECS_STATSMODELS[k] = {
        "python": "3.6",
        "packages": "pytest 'cython<0.28' 'numpy<1.13'",
        "env_patches": [
            'export FFLAGS="-fallow-argument-mismatch -fPIC"',
            'export FCFLAGS="-fallow-argument-mismatch -fPIC"',
        ],
        "install": "pip install --no-build-isolation -e .",
        "test_cmd": TEST_PYTEST,
        "pip_packages": [
            "'patsy==0.4.0'",
            "'pandas==0.19'",
            "'scipy==0.18'",
            "'setuptools<60'",
        ],
    }

for k in ["0.11", "0.12"]:
    SPECS_STATSMODELS[k] = {
        "python": "3.7",
        "packages": "pytest 'cython==0.29' 'numpy==1.17'",
        "install": "pip install --no-build-isolation -e .",
        "test_cmd": TEST_PYTEST,
        "pip_packages": [
            "'patsy==0.5.1'",
            "'pandas<1'",
            "'scipy==1.3'",
            "'setuptools<65'",
        ],
    }

for k in ["0.13", "0.14"]:
    SPECS_STATSMODELS[k] = {
        "python": "3.9",
        "packages": "pytest 'numpy==1.22.3'",
        "install": 'SETUPTOOLS_ENABLE_FEATURES="legacy-editable" pip install --no-build-isolation -e .',
        "test_cmd": TEST_PYTEST,
        "pip_packages": [
            "'cython==0.29.33'",
            "'patsy==0.5.6'",
            "'pandas==1.4'",
            "'scipy==1.8'",
            "'setuptools<65'",
        ],
    }

MAP_REPO_VERSION_TO_SPECS.update({"statsmodels/statsmodels": SPECS_STATSMODELS})

# TODO not tested.

SPECS_SCIKIT_IMAGE = {}
for k in [
    "0.12",
    "0.13",
    "0.14",
    "0.15",
    "0.16",
    "0.17",
    "0.18",
    "0.19",
    "0.20",
    "0.21",
    "0.22",
    "0.23",
    "0.24",
    "0.25",
    "0.26",
]:
    SPECS_SCIKIT_IMAGE[k] = {
        "python": "3.6",
        "pre_install": ["pip install -r requirements.txt"],
        "install": "pip install -e .",
        "test_cmd": TEST_PYTEST,
    }

for k in ["0.18", "0.19"]:
    SPECS_SCIKIT_IMAGE[k]["python"] = "3.8"

for k in ["0.20", "0.21", "0.22", "0.23", "0.24", "0.25", "0.26"]:
    SPECS_SCIKIT_IMAGE[k]["python"] = "3.10"

MAP_REPO_VERSION_TO_SPECS.update({"scikit-image/scikit-image": SPECS_SCIKIT_IMAGE})


for k in [
    "0.17",
    "0.18",
    "0.20",
    "0.22",
    "0.23",
    "0.25",
    "0.26",
    "0.28",
    "0.29",
    "0.30",
    "0.31",
    "0.32",
    "0.33",
]:
    # pymc-devs/pytensor
    pass


for k in [
    "1.9",
    "2.0",
    "2.61",
    "2.62",
    "2.63",
    "2.7",
    "2.71",
    "2.8",
    "3.01",
    "3.02",
    "3.11",
    "3.20",
    "3.30",
    "3.31",
    "3.40",
    "3.41",
    "3.50",
    "3.51",
]:
    # networkx/networkx
    pass


# Exceptions to allow stackframe..
STACKFRAME_CHECK_EXCEPTONS = {"pandas-dev__pandas-45247"}


# All keys should be in lower case
LOWER_MAP_REPO_VERSION_TO_SPECS = {
    k.lower(): v for k, v in MAP_REPO_VERSION_TO_SPECS.items()
}
MAP_REPO_VERSION_TO_SPECS = LOWER_MAP_REPO_VERSION_TO_SPECS
