# Helper scripts to setup GCP container for reproducible performance benchmarking.

`setup_docker.sh`: Please read it first, then run with `sudo setup_docker.sh --also-containerd <MAX_MEM> <HIGH_MEM>`. This script limits Docker daemon resources and pins it to a set of physical CPUs.
`undo.sh`: This undoes the effects of the above script.
`setup_vm`: Simple helper script to do a first time setup and installation on a GCP VM.