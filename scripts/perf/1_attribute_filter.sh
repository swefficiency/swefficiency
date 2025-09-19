#!/bin/bash

# Define the different options to iterate over
# options=("scikit-learn" )  # Replace these with actual values

options=(
    "scikit-learn" "dask" "numpy" "pandas" "matplotlib" "seaborn" "xarray" "sympy" "astropy" "scipy" \
    "conan" "pydantic" "statsmodels" "pyvista" "spaCy" "gensim" "numba" "pillow" "scikit-image" \
    "pomegranate" "networkx" "pvlib-python" "pymc" "modin" "pytensor"
)  # Replace these with actual values

# scipy
# sympy
# astropy
# spaCy
# scikit-image
# networkx
# modin

# py-tensor
# scrapy
# statsmodels


# pyvista
#

# Next up:
# numba/numba
# python-pillow


# pydantic/pydantic
# explosion/spaCy
# piskvorky/gensim low perf
# numba/numba
# polars/polars
# python-pillow/pillow
# jmschrei/pomegranate
# pymc-devs/pymc


# pvlib?

# Loop through each option and run the command
for option in "${options[@]}"; do
    echo "Running command with option: $option"

    python ./swefficiency/perf_filter/attributes/filter.py \
        --prs_path ./artifacts/pull_requests/${option}-prs.jsonl \
        --instances_path ./artifacts/tasks/${option}-task-instances.jsonl.all \
        --output_dir ./artifacts/1_attributes/
done


