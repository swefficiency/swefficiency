#!/bin/bash

PER_JOB_CORES=8  # Number of cores per job, adjust as needed
PER_JOB_MEMORY="128G"  # Memory per job, adjust as needed
PARTITION="test"  # Partition to submit to, adjust as needed
NUM_PARALLEL_JOBS=16  # Number of parallel jobs, adjust as needed

REPOS=(
    # "numpy"
    # "scipy"
    "matplotlib"
    # "scikit-learn"
    # # "xarray"
    # "sympy"
    # # "astropy"
    # # "dask"
    # "pandas"
    # "modin"
)

TIMES=(
    # "0-12:00"  # numpy
    # "1-00:00"  # scipy
    "0-12:00"  # matplotlib
    # "0-12:00"  # scikit-learn
    # # "0-12:00"  # xarray
    # "0-12:00"  # sympy
    # # "1-00:00"  # astropy
    # # "1-00:00"  # dask
    # "0-12:00"  # pandas
    # "1-00:00"  # modin
)

# Assert that REPOS and TIMES have the same length
if [ ${#REPOS[@]} -ne ${#TIMES[@]} ]; then
    echo "Error: REPOS and TIMES arrays must have the same length."
    exit 1
fi

# Zip the REPOS and TIMES arrays together
for i in "${!REPOS[@]}"; do
    repo="${REPOS[i]}"
    time="${TIMES[i]}"

    echo "Submitting job for $repo"
    sbatch <<EOT
#!/bin/bash
#SBATCH -c ${PER_JOB_CORES}            # Number of cores
#SBATCH -t ${time}                     # Runtime in D-HH:MM
#SBATCH -p ${PARTITION}                # Partition to submit to
#SBATCH --mem=${PER_JOB_MEMORY}        # Memory for all cores
#SBATCH --j "coverage_${repo}_${PARTITION}"      # Job name
#SBATCH -o coverage_${repo}_${PARTITION}_%A_%a.out  # STDOUT file, %A = job ID, %a = array index
#SBATCH -e coverage_${repo}_${PARTITION}_%A_%a.err  # STDERR file
#SBATCH --constraint=sapphirerapids
#SBATCH --array=1-${NUM_PARALLEL_JOBS}  # Array job with N tasks, adjust as needed

source ~/.bashrc

. \$CONDA_ROOT/etc/profile.d/conda.sh
conda activate sweperf

# Check if theres a file /var/tmp/cleaned, if not, clean /scratch and /var/tmp
# if [ ! -f /var/tmp/cleaned.tmp ]; then
    echo "Cleaning /scratch and /var/tmp directories..."
    rm -rf /scratch/*
    rm -rf /var/tmp/*
    touch /var/tmp/cleaned.tmp

export UNIQUE_ID="\${SLURM_JOB_ID}_${repo}_\$SLURM_ARRAY_TASK_ID"
export TMPDIR="/scratch/$USER/\${UNIQUE_ID}"

podman system migrate 
podman system prune -f
# podman info

# Set up podman socket for docker compatibility.
SOCKET_PATH="/scratch/\$USER/\${SLURM_JOB_ID}_${repo}_\$SLURM_ARRAY_TASK_ID/podman"
SOCKET_URL="unix://\$SOCKET_PATH/podman.sock"
mkdir -p "\$SOCKET_PATH"
podman system service --time=0 \$SOCKET_URL &
svc_pid=\$!

# Export the DOCKER_HOST environment variable to point to the podman socket.
export DOCKER_HOST="\$SOCKET_URL"
echo "Using DOCKER_HOST: \$DOCKER_HOST"

# 2 . Wait until the daemon responds (max 30 s)
timeout=30   # seconds
until podman --url "\$SOCKET_URL" info >/dev/null 2>&1; do
    if (( timeout-- == 0 )); then
        echo "🛑 Podman service did not come up in time" >&2
        kill "\$svc_pid" 2>/dev/null || true
        exit 1
    fi
    sleep 1
done


# HACK: Parallelization. Slurm does not support CPU pinning for podman, so we use a workaround.
VERSION_DATA=artifacts/2_versioning/$repo-task-instances_attribute_versions.non-empty.jsonl
LABELLED_INSTANCES_TO_RUN=\$(python scripts/slurm/get_instances_split.py --dataset \$VERSION_DATA --split_index \$SLURM_ARRAY_TASK_ID --num_splits \$SLURM_ARRAY_TASK_COUNT)

echo "Running $repo"
echo "Using dataset: \$VERSION_DATA"
echo "Using labelled instances: \$LABELLED_INSTANCES_TO_RUN"

echo "Running coverage validation for \$LABELLED_INSTANCES_TO_RUN"
python swefficiency/harness/run_validation.py \
    --dataset_name \$VERSION_DATA \
    --run_id debugging_coverage_fix_import_alt3_$repo \
    --cache_level env \
    --max_build_workers 4 \
    --max_workers 1 \
    --timeout 3_600 \
    --run_coverage true \
    --instance_ids \$LABELLED_INSTANCES_TO_RUN 

EOT
done

