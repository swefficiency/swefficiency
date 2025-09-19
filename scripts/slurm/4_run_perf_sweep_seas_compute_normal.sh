#!/bin/bash

# Get all CLI args, which are optional instance IDs to specifically run.
INSTANCE_IDS=("$@")

PER_JOB_CORES=96  # Number of cores per job, adjust as needed
PER_JOB_MEMORY="128G"  # Memory per job, adjust as needed
PARTITION="test"  # Partition to submit to, adjust as needed
NUM_PARALLEL_JOBS=1  # Number of parallel jobs, adjust as needed
NUM_JOB_WORKERS=1  # Number of workers per job, adjust as needed

# For debugging settings.
# PER_JOB_CORES=8   # Number of cores per job, adjust as needed
# PER_JOB_MEMORY="64G"  # Memory per job, adjust as needed
# PARTITION="test"  # Partition to submit to, adjust as needed
# NUM_PARALLEL_JOBS=1  # Number of parallel jobs, adjust as needed
# NUM_JOB_WORKERS=1  # Number of workers per job, adjust as needed

REPO_TIME_TUPLES=(
    "pandas:0-12:00"
    # "numpy:0-4:00"
    # "scipy:0-12:00"
    # "matplotlib:0-12:00"
    # "scikit-learn:0-3:00"
    # "xarray:0-2:00"
    # "sympy:0-4:00"
    # "astropy:0-12:00"
    # "dask:0-4:00"
)

# Parse REPO_TIME_TUPLES into REPOS and TIMES arrays
REPOS=()
TIMES=()
for tuple in "${REPO_TIME_TUPLES[@]}"; do
    repo="${tuple%%:*}"
    time="${tuple#*:}"
    REPOS+=("$repo")
    TIMES+=("$time")
done

# Assert that REPOS and TIMES have the same length
if [ ${#REPOS[@]} -ne ${#TIMES[@]} ]; then
    echo "Error: REPOS and TIMES arrays must have the same length."
    exit 1
fi

# Zip the REPOS and TIMES arrays together
for i in "${!REPOS[@]}"; do
    repo="${REPOS[i]}"
    time="${TIMES[i]}"

    # If INSTANCE_IDS is not empty, filter the list to only these repo's instances.
    INSTANCE_RUN_ARG=""
    if [ ${#INSTANCE_IDS[@]} -gt 0 ]; then
        # Filter the repo to only include instances that match the provided IDs.
        INSTANCE_IDS_TO_RUN=()
        for instance_id in "${INSTANCE_IDS[@]}"; do
            if [[ "$instance_id" == *"$repo"* ]]; then
                INSTANCE_IDS_TO_RUN+=("$instance_id")
            fi
        done
        if [ ${#INSTANCE_IDS_TO_RUN[@]} -eq 0 ]; then
            echo "No instances found for repo $repo with provided IDs. Skipping."
            continue
        fi
        INSTANCE_RUN_ARG="--instance_ids ${INSTANCE_IDS_TO_RUN[*]}"
        echo "Running instances for $repo: ${INSTANCE_IDS_TO_RUN[*]}"
    fi
    
    ADDITIONAL_ARGS=""
    # if [[ "$repo" == "pandas" ]]; then
    #     # Pandas images take way too long to build, so we use the dockerhub image.
    #     ADDITIONAL_ARGS="--use_dockerhub_images true"
    # fi

    sbatch <<EOT
#!/bin/bash
#SBATCH -c ${PER_JOB_CORES}            # Number of cores
#SBATCH -t ${time}                     # Runtime in D-HH:MM
#SBATCH -p ${PARTITION}                # Partition to submit to
#SBATCH --mem=${PER_JOB_MEMORY}        # Memory for all cores
#SBATCH --j "perf_${repo}_${PARTITION}"      # Job name
#SBATCH -o perf_${repo}_${PARTITION}_%A_%a.out  # STDOUT file, %A = job ID, %a = array index
#SBATCH -e perf_${repo}_${PARTITION}_%A_%a.err  # STDERR file
#SBATCH --constraint=sapphirerapids
#SBATCH --array=1-${NUM_PARALLEL_JOBS}  # Array job with N tasks, adjust as needed

source ~/.bashrc

. \$CONDA_ROOT/etc/profile.d/conda.sh
conda activate sweperf

# rm -rf /scratch/*
# rm -rf /var/tmp/*

export UNIQUE_ID="\${SLURM_JOB_ID}_${repo}_\$SLURM_ARRAY_TASK_ID"

export TMPDIR="/scratch/$USER/\${UNIQUE_ID}"

podman system migrate 
podman system reset -f
podman system prune -f -a

export GH_USERNAME="TODO"  # Set your GitHub username here
export CR_PAT=<YOUR_GITHUB_TOKEN_HERE>  # Set your GitHub token here
echo \$CR_PAT | docker login ghcr.io -u  --password-stdin

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

VERSION_DATA="artifacts/2_versioning/$repo-task-instances_attribute_versions.non-empty.jsonl"

echo "Running $repo"
echo "Using dataset: \$VERSION_DATA"

echo "Running coverage validation for \$LABELLED_INSTANCES_TO_RUN"
python swefficiency/harness/run_validation.py \
    --dataset_name \$VERSION_DATA \
    --run_id perf_profiling_20250612_$repo \
    --cache_level env \
    --max_build_workers 16 \
    --max_workers $NUM_JOB_WORKERS \
    --timeout 7_200 \
    --run_coverage false \
    --run_perf true \
    --run_correctness true \
    --gdrive_annotation_sheet global_sweperf_all_data_annotate \
    --push_to_dockerhub true \
    $ADDITIONAL_ARGS $INSTANCE_RUN_ARG \

EOT

done





