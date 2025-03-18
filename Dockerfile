FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive

# setup pyenv
ENV HOME /root
WORKDIR ${HOME}

RUN apt-get -y update && apt-get -y upgrade && \
    apt-get -y install --no-install-recommends \
            git \
            make \
            cmake \
            build-essential \
            python${PYTHON_VERSION}-dev \
            python3-pip \
            python${PYTHON_VERSION}-distutils \
            libssl-dev \
            zlib1g-dev \
            libbz2-dev \
            libreadline-dev \
            libsqlite3-dev \
            liblzma-dev \
            libffi-dev \
            curl

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"
ENV HF_HUB_CACHE="/nfs/.cache/huggingface"

CMD ["sleep", "INF"]
