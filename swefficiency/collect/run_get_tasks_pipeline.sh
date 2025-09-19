#!/usr/bin/env bash

# If you'd like to parallelize, do the following:
# * Create a .env file in this folder
# * Declare GITHUB_TOKENS=token1,token2,token3...

# astropy/astropy Astronomy and astrophysics core library BSD 3-Clause
# django/django Web framework for building web applications BSD 3-Clause
# pallets/flask Lightweight framework for small web apps BSD 3-Clause
# matplotlib/matplotlib Plotting library for creating visuals Custom
# pylint-dev/pylint Static code analyser for Python 2 or 3 GPL 2.0
# pytest-dev/pytest Testing framework for Python MIT
# psf/requests Simple, elegant library for writing HTTP requests Apache-2.0
# scikit-learn/scikit-learn Machine Learning in Python BSD 3-Clause
# mwaskom/seaborn Statistical data visualization in Python BSD 3-Clause
# sphinx-doc/sphinx Library for creating documentation Custom
# sympy/sympy Computer algebra system written in Python Custom
# pydata/xarray N-D labeled arrays and datasets in Python Apache-2.0

PR_PATH=artifacts/pull_requests/
TASK_PATH=artifacts/tasks/

mkdir -p $PR_PATH
mkdir -p $TASK_PATH

    # --repos 'getmoto/moto' 'Project-MONAI/MONAI' 'pandas-dev/pandas' 'python/mypy' 'dask/dask' 'iterative/dvc' 'conan-io/conan' 'pydantic/pydantic' 'facebookresearch/hydra' 'bokeh/bokeh' 'modin-project/modin' \
    #     --repos 'numpy/numpy' 'scipy/scipy' 'statsmodels/statsmodels' \ 'sympy/sympy' 'conan-io/conan' 'statsmodels/statsmodels' 'astropy/astropy' 

# REDO pola-rs/polars
# -repos 'networkx/networkx' 'scikit-image/scikit-image' 'numba/numba' 'scipy/scipy' 'pyvista/pyvista' 'pydantic/pydantic' 'explosion/spaCy' 'piskvorky/gensim' 'python-pillow/pillow' 'jmschrei/pomegranate'  'pymc-devs/pymc' \
python get_tasks_pipeline.py \
    --repos 'scrapy/scrapy' 'pymc-devs/pytensor' \
    --path_prs ${PR_PATH} \
    --path_tasks ${TASK_PATH}

 # --repos 'astropy/astropy', 'django/django', 'pallets/flask', 'matplotlib/matplotlib', 'pylint-dev/pylint', 'pytest-dev/pytest', 'psf/requests', 'scikit-learn/scikit-learn', 'mwaskom/seaborn', 'sphinx-doc/sphinx', 'sympy/sympy', 'pydata/xarray' \