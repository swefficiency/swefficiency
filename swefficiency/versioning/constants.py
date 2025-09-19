# Constants - Task Instance Version File
MAP_REPO_TO_VERSION_PATHS = {
    "bokeh/bokeh": ["bokeh/__init__.py"],
    "conan-io/conan": ["conans/__init__.py", "conans/client/conf/__init__.py"],
    "dask/dask": ["dask/__init__.py"],
    "dbt-labs/dbt-core": ["core/dbt/version.py", "core/dbt/__init__.py"],
    "django/django": ["django/__init__.py"],
    "facebookresearch/hydra": ["hydra/__init__.py"],
    "getmoto/moto": ["moto/__init__.py"],
    "huggingface/transformers": ["src/transformers/__init__.py"],
    "HypothesisWorks/hypothesis": ["hypothesis/__init__.py", "hypothesis/version.py"],
    "iterative/dvc": ["dvc/__init__.py", "dvc/version.py"],
    "marshmallow-code/marshmallow": ["src/marshmallow/__init__.py"],
    "modin-project/modin": ["modin/__init__.py"],
    "mwaskom/seaborn": ["seaborn/__init__.py"],
    "pallets/flask": ["src/flask/__init__.py", "flask/__init__.py"],
    "pandas-dev/pandas": ["pandas/__init__.py"],
    "Project-MONAI/MONAI": ["monai/__init__.py"],
    "psf/requests": ["requests/__version__.py", "requests/__init__.py", "src/requests/__version__.py"],
    "pyca/cryptography": [
        "src/cryptography/__about__.py",
        "src/cryptography/__init__.py",
    ],
    "pydantic/pydantic": ["pydantic/__init__.py", "pydantic/version.py"],
    "pylint-dev/astroid": ["astroid/__pkginfo__.py", "astroid/__init__.py"],
    "pylint-dev/pylint": ["pylint/__pkginfo__.py", "pylint/__init__.py"],
    "pytest-dev/pytest": ["src/_pytest/_version.py", "_pytest/_version.py"],
    "python/mypy": ["mypy/version.py", "mypy/__init__.py"],
    "pyvista/pyvista": ["pyvista/_version.py", "pyvista/__init__.py"],
    "Qiskit/qiskit": ["qiskit/VERSION.txt"],
    "scikit-learn/scikit-learn": ["sklearn/__init__.py"],
    "sphinx-doc/sphinx": ["sphinx/__init__.py"],
    "spyder-ide/spyder": ["spyder/__init__.py", "spyder/version.py"],
    "sympy/sympy": ["sympy/release.py", "sympy/__init__.py"],
    "explosion/spaCy": ["spacy/about.py"],
    "scikit-image/scikit-image": ["skimage/__init__.py"],
}

# Cosntants - Task Instance Version Regex Pattern
MAP_REPO_TO_VERSION_PATTERNS = {
    k: [r'__version__ = [\'"](.*)[\'"]', r"VERSION = \((.*)\)"]
    for k in [
        "bokeh/bokeh",
        "conan-io/conan",
        "dask/dask",
        "dbt-labs/dbt-core",
        "django/django",
        "facebookresearch/hydra",
        "getmoto/moto",
        "huggingface/transformers",
        "HypothesisWorks/hypothesis",
        "iterative/dvc",
        "marshmallow-code/marshmallow",
        "modin-project/modin",
        "mwaskom/seaborn",
        "pallets/flask",
        "pandas-dev/pandas",
        "Project-MONAI/MONAI",
        "psf/requests",
        "pyca/cryptography",
        "pydantic/pydantic",
        "pylint-dev/astroid",
        "pylint-dev/pylint",
        "python/mypy",
        "scikit-learn/scikit-learn",
        "sphinx-doc/sphinx",
        "spyder-ide/spyder",
        "sympy/sympy",
        "modin-project/modin",
        "facebookresearch/hydra",
        "explosion/spaCy",
        "scikit-image/scikit-image",    
    ]
}
MAP_REPO_TO_VERSION_PATTERNS.update(
    {
        k: [
            r'__version__ = [\'"](.*)[\'"]',
            r'__version__ = version = [\'"](.*)[\'"]',
            r"VERSION = \((.*)\)",
        ]
        for k in ["pytest-dev/pytest", "matplotlib/matplotlib"]
    }
)
MAP_REPO_TO_VERSION_PATTERNS.update({k: [r"(.*)"] for k in ["Qiskit/qiskit"]})
MAP_REPO_TO_VERSION_PATTERNS.update({k: [r"version_info = [\d]+,[\d\s]+,"] for k in ["pyvista/pyvista"]})

SWE_BENCH_URL_RAW = "https://raw.githubusercontent.com/"

# python/mypy
MAP_REPO_TO_VERSION_PATHS.update({"python/mypy": ["mypy/version.py"]})
MAP_REPO_TO_VERSION_PATTERNS.update({"python/mypy": [r'__version__ = [\'"](.*)[\'"]', r"VERSION = \((.*)\)"]})

# getmoto/moto
MAP_REPO_TO_VERSION_PATHS.update({"getmoto/moto": ["moto/__init__.py"]})
MAP_REPO_TO_VERSION_PATTERNS.update({"getmoto/moto": [r'__version__ = [\'"](.*)[\'"]', r"VERSION = \((.*)\)"]})

# conan-io/conan
MAP_REPO_TO_VERSION_PATHS.update({"conan-io/conan": ["conans/__init__.py"]})
MAP_REPO_TO_VERSION_PATTERNS.update({"conan-io/conan": [r'__version__ = [\'"](.*)[\'"]', r"VERSION = \((.*)\)"]})

# networkx/networkx
MAP_REPO_TO_VERSION_PATHS.update({"networkx/networkx": ["networkx/release.py", "networkx/__init__.py"]})
MAP_REPO_TO_VERSION_PATTERNS.update({"networkx/networkx": [r'__version__ = [\'"](.*)[\'"]', r"VERSION = \((.*)\)", r'major\s*=\s*["\'](\d+)["\']\s*[\r\n]+minor\s*=\s*["\'](\d+)["\']']})