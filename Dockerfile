# Dockerfile
#
# Created on Tue Nov 14 2023 by Florian Pfleiderer
#
# Copyright (c) 2023 TU Wien

# Use an official PyTorch base image
# FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# nvidia cuda base image
FROM nvidia/cuda:11.6.1-base-ubuntu20.04

# Set the working directory in the container
WORKDIR /point-slam

# Set noninteractive mode
ENV DEBIAN_FRONTEND=noninteractive

# Install git and wget
RUN apt-get update && \
    apt-get install -y git wget cmake gcc g++ libgl1-mesa-glx libglib2.0-0 unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Reset noninteractive mode
ENV DEBIAN_FRONTEND=dialog

# Install miniconda.
ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
# Make non-activate conda commands available.
ENV PATH=$CONDA_DIR/bin:$PATH
# Make conda activate command available from /bin/bash --login shells.
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
# Make conda activate command available from /bin/bash --interative shells.
RUN conda init bash

# start shell in login mode
SHELL ["/bin/bash", "--login", "-c"]

# run updates
RUN conda update -n base -c defaults conda

# copy environment file
COPY env.yaml env.yaml

# Create a Conda environment
RUN conda env create -f env.yaml

# start container in cyws3d env
RUN touch ~/.bashrc && echo "conda activate point-slam" >> ~/.bashrc

COPY scripts/ scripts/
# RUN bash scripts/download_replica.sh

# copy source code
COPY configs/ configs/
COPY cull_replica_mesh/ cull_replica_mesh/
COPY pretrained/ pretrained/
COPY src/ src/
COPY repro.sh run.py test_deterministic.py ./

# Set the default command to run when the container starts
CMD ["bash"]
