#!/bin/bash

# Get all CLI args, which are optional instance IDs to specifically run.
INDEX=$1


PER_JOB_CORES=96  # Number of cores per job, adjust as needed
PER_JOB_MEMORY="784G"  # Memory per job, adjust as needed
PARTITION="test"  # Partition to submit to, adjust as needed
NUM_PARALLEL_JOBS=1  # Number of parallel jobs, adjust as needed
NUM_JOB_WORKERS=24  # Number of workers per job, adjust as needed
TIME="0-12:00"  # Time limit for each job, adjust as needed

# # DEBUGGING
# PARTITION="test"  # Partition to submit to, adjust as needed
# TIME="0-12:00"  # Time limit for each job, adjust as needed

sbatch <<EOT
#!/bin/bash
#SBATCH -c ${PER_JOB_CORES}            # Number of cores
#SBATCH -t ${TIME}                     # Runtime in D-HH:MM
#SBATCH -p ${PARTITION}                # Partition to submit to
#SBATCH --mem=${PER_JOB_MEMORY}        # Memory for all cores
#SBATCH --j "perf_eval_single_thread_${PARTITION}"      # Job name
#SBATCH -o perf_eval_single_thread_${repo}_${PARTITION}_%A_%a.out  # STDOUT file, %A = job ID, %a = array index
#SBATCH -e perf_eval_single_thread_${repo}_${PARTITION}_%A_%a.err  # STDERR file
#SBATCH --constraint=sapphirerapids
#SBATCH --array=1-${NUM_PARALLEL_JOBS}  # Array job with N tasks, adjust as needed

source ~/.bashrc

. \$CONDA_ROOT/etc/profile.d/conda.sh
conda activate sweperf

rm -rf /scratch/*
rm -rf /var/tmp/*

export UNIQUE_ID="\${SLURM_JOB_ID}_\$SLURM_ARRAY_TASK_ID"

export TMPDIR="/scratch/$USER/\${UNIQUE_ID}"

export GH_USERNAME="TODO"  # Set your GitHub username here
export CR_PAT=<YOUR_GITHUB_TOKEN_HERE>  # Set your GitHub token here
echo \$CR_PAT | docker login ghcr.io -u \$GH_USERNAME --password-stdin

podman system migrate
podman system reset -f
podman system prune -f -a

# podman info

# Set up podman socket for docker compatibility.
SOCKET_PATH="/scratch/\$USER/\${SLURM_JOB_ID}_\$SLURM_ARRAY_TASK_ID/podman"
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

scripts/vm/run_eval.sh jeff_perf_ground_truth_recheck$INDEX $NUM_JOB_WORKERS

EOT
