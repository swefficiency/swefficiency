#!/bin/bash
set -e

# 1. Install Miniconda (conda-forge)
CONDA_DIR="$HOME/miniconda3"
if [ ! -d "$CONDA_DIR" ]; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
    export PATH="$CONDA_DIR/bin:$PATH"
    "$CONDA_DIR/bin/conda" init
else
    echo "Miniconda already installed."
fi

# Ensure conda-forge is the highest priority channel
"$CONDA_DIR/bin/conda" config --add channels conda-forge
"$CONDA_DIR/bin/conda" config --set channel_priority strict

# 2. Install Docker and dependencies
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "Docker installed successfully."

# Please make sure to add GHCR token CR_PAT and HF token using huggingface-cli login after this!
# 1) echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin
# 2) huggingface-cli login