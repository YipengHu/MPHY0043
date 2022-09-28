# Dockerfile for the module tutorial and coursework

FROM ubuntu:20.04

# git and conda
RUN apt-get update && apt-get install -y wget git \
 && rm -rf /var/lib/apt/lists/*
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && mkdir /root/.conda \
 && bash Miniconda3-latest-Linux-x86_64.sh -b \
 && rm -f Miniconda3-latest-Linux-x86_64.sh
ARG PATH="/root/miniconda3/bin:$PATH"
RUN conda init bash
ENV PATH="/root/miniconda3/bin:$PATH"

# clone the repo in "/workspace"
RUN git clone https://github.com/YipengHu/MPHY0043.git workspace/mphy0043 
WORKDIR /workspace

# create the tutorial/coursework conda environment "mphy0043"
ARG CONDA_ENV="mphy0043"
RUN conda create -n $CONDA_ENV tensorflow==2.5 \
 && conda activate mphy0043 \
 && pip install notebook matplotlib av "monai[nibabel, gdown, ignite]"
 && echo "source activate $CONDA_ENV" > ~/.bashrc 
