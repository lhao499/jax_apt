FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Declare the image name
# ENV IMG_NAME=11.3.1-cudnn8-runtime-ubuntu20.04

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Set timezone
# RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
# ENV TZ="America/Los_Angeles" \
#     DEBIAN_FRONTEND=noninteractive
# Install tzdata and wireshark and prevent it from asking for input. Would be installed in setup script.
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles
# RUN apt-get install -y tzdata wireshark-common

# System packages.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-dev \
    python3-pip \
    unrar \
    wget \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Python as default
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
ln -sf /usr/bin/pip3 /usr/bin/pip

# MuJoCo.
ENV MUJOCO_GL egl
RUN mkdir -p /root/.mujoco && \
    wget -nv https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip && \
    unzip mujoco.zip -d /root/.mujoco && \
    rm mujoco.zip

# Python packages
COPY gpu_requirements.txt /tmp/
RUN pip install --no-cache-dir --requirement /tmp/gpu_requirements.txt

# # Atari ROMS.
# RUN wget -nv http://www.atarimania.com/roms/Roms.rar -P /tmp && \
#     unrar x -r /tmp/Roms.rar /tmp && \
#     unzip /tmp/ROMS.zip -d /tmp && \
#     python -m atari_py.import_roms /tmp/ROMS

WORKDIR /home
