# Environment setup (Docker)

This document shows a quick, reproducible way to run the project inside a Docker container with GPU support. It includes the main host commands you need to prepare the machine, build the image, and run the container.

Prerequisites
- A Linux host with a recent NVIDIA driver installed (for CUDA 12.x, driver >= 535 recommended).
- sudo access for installing system packages.
- (Optional) `uv` used below is from Astral for syncing helper scripts — adjust if you don't use it.

1) Install Docker and NVIDIA container runtime

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

sudo apt update
sudo apt install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

If your system has systemd proxy drop-ins for Docker, verify and remove them if necessary:

```bash
# Check for proxy drop-ins
sudo systemctl show docker | grep -i proxy

# If proxy files are present, remove and reload systemd
sudo rm -f /etc/systemd/system/docker.service.d/http-proxy.conf
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl restart docker
```

2) Adjust Dockerfile snippet (optional)

Some base images attempt to install `just` using an external installer. If your Dockerfile contains a `RUN` that uses `just` installer and you prefer a deterministic binary download, replace it with a direct download + install. Example (replace the existing `RUN`):

```dockerfile
# old (example):
# RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin --tag 1.42.4

# recommended replacement (deterministic):
RUN curl -L -o /tmp/just.tar.gz \
    https://github.com/casey/just/releases/download/1.42.4/just-1.42.4-x86_64-unknown-linux-musl.tar.gz && \
    tar -xzf /tmp/just.tar.gz -C /tmp && \
    mv /tmp/just /usr/local/bin/just || true && \
    chmod +x /usr/local/bin/just
```

3) Build the Docker image

Use a large `nofile` ulimit when building to avoid warnings on systems with many files:

```bash
docker build --ulimit nofile=131071:131071 -f Dockerfile . -t cosmos-predict2.5:latest
```

4) Run the container with GPU access and mount the repo

Recommended run command mounts the working directory and enables all GPUs. Adjust `--network` and mounts to your needs.

```bash
docker run --name cosmos-predict25-container --gpus all --network host -v "$(pwd)":/workspace -v /workspace/.venv -it cosmos-predict2.5:latest
```

Helpful Docker commands

```bash
# List running containers
docker ps

# Exec into an existing container
docker exec -it <container-id> /bin/bash

# Stop a container by name
docker stop cosmos-predict25-container
```

5) Pre-download checkpoints on host (optional, recommended)

To avoid network downloads from inside the container and to speed up runs, download the required checkpoints on the host and place them under a project-local `checkpoints/` directory. The project contains logic in `cosmos_predict2/_src/imaginaire/utils/checkpoint_db.py` that will prefer local checkpoints under `./checkpoints/<repo_id>`.

Example using Hugging Face CLI (host):

```bash
# create a local checkpoints folder and download a HF model
mkdir -p checkpoints/nvidia/Cosmos-Reason1-7B
huggingface-cli download --repo-id nvidia/Cosmos-Reason1-7B --output-dir checkpoints/nvidia/Cosmos-Reason1-7B
```

After downloading, you can mount the repository and the `checkpoints/` directory into the container so the code picks them up without a network call.

6) Run an example (inside container or host Python)

```bash
python examples/action_conditioned.py \
  -i assets/action_conditioned/basic/inference_params.json \
  -o outputs/action_conditioned/basic
```

Notes
- If you prefer to download checkpoints inside the container, the code will fall back to the Hugging Face hub when a local copy is not found.
- If you see issues with CUDA drivers or `torch` binaries, confirm the container image CUDA/toolkit variants match your host drivers.

If you want, I can also add a short `docker-compose.yml` for easier local runs, or modify the `checkpoint_db.py` to look for specific `.pt` files first — say the word and I will do it.