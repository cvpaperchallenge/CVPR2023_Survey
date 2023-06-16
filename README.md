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

Here, we show example prerequisites installation codes for Ubuntu. If prerequisites  are already installed in your environment, please skip this section. If you want to install it in another environment, please follow the official documentation.

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

If `sudo docker run hello-world` works, the installation succeeded.

## Download all CVPR 2023 papers

### (Optional) Parse CVF page

NOTE: JSON file created by this step is already included in the repo. So this step is optional.

Please run the following command inside the "core" container (This command generates `papers.json` under the data directory). This process will take 10-20 minutes.

```bash
$ poetry run python3 src/scripts/parse_cvf_page.py 
```

### Download PDF

Please run the following command from inside the "core" container.

```bash
$ poetry run python3 src/scripts/download_papers.py 
```

## Generate summaries of CVPR 2023 papers

### Setup environmental variables

We use the following APIs to generate summaries:

- Mathpix API: To convert PDF into Latex format text.
- OpenAI API: To use LLM (GPT).

To use the above APIs, we need to set the following environmental variables:

- MATHPIX_API_ID
- MATHPIX_API_KEY
- OPENAI_API_KEY

So please run the following command to create an `envs.env` file and replace sample values with actual ones.

```bash
% cp environments/envs.env.sample environments/envs.env
```

Values written in the `envs.env` file are automatically loaded by docker and stored as environmental variables in the container.

### Convert PDF to Latex format text

Here convert PDF to Latex format using [Mathpix](https://mathpix.com/) API. This makes it possible to extract the original structure of papers.

Please run the following command from inside the "core" container.

```bash
$ poetry run python3 src/scripts/convert_to_latex.py
```

### Generate summaries

Now we are ready to generate summaries by using LLM (GPT). Please run the following command from inside the "core" container.

```bash
% poetry run python3 src/scripts/generate_summaries.py
```
