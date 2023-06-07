# CVPR 2023 Survey

[![MIT License](https://img.shields.io/github/license/cvpaperchallenge/CVPR2023_Survey?color=green)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Code style: flake8](https://img.shields.io/badge/code%20style-flake8-black)](https://github.com/PyCQA/flake8)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Typing: mypy](https://img.shields.io/badge/typing-mypy-blue)](https://github.com/python/mypy)

## Prerequisites

- [Docker](https://www.docker.com/)
- [Docker Compose](https://github.com/docker/compose)

**NOTE**: Example codes in the README.md are written for `Docker Compose v2`.

## Prerequisites installation

Here, we show example prerequisites installation codes for Ubuntu. If prerequisites  are already installed your environment, please skip this section. If you want to install in another environment, please follow the officail documentations.

- Docker and Docker Compose: [Install Docker Engine](https://docs.docker.com/engine/install/)
- NVIDIA Container Toolkit (nvidia-docker2): [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

### Install Docker and Docker Compose

```bash
# Set up the repository
$ sudo apt update
$ sudo apt install ca-certificates curl gnupg lsb-release
$ sudo mkdir -p /etc/apt/keyrings
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
$ echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker and Docker Compose
$ sudo apt update
$ sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

If `sudo docker run hello-world` works, installation succeeded.

## Download all CVPR 2023 papers

### (Optional) Parse CVF page

NOTE: JSON file created by this step is already included in repo. So this step is optional.

Pleae run following command from inside of "core" container (This command generates `papers.json` under data directory). This process will take 10-20 minutes.

```bash
$ poetry run python3 src/scripts/parse_cvf_page.py 
```

### Download PDF

Pleae run following command from inside of "core" container.

```bash
$ poetry run python3 src/scripts/download_papers.py 
```
