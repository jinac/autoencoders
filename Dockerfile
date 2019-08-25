# Environment to run tensorflow 2.0 
#
# Author: John Inacay

FROM tensorflow/tensorflow:2.0.0b1-gpu-py3

# Install libraries that I want to have by default
RUN apt-get update -qq && apt-get install -y \
    python3-venv \
    python3-pip \
    libsm6 \
    libxext6 \
    libxrender-dev


RUN pip3 install --upgrade pip
RUN pip3 install \
    opencv-python \
    matplotlib \
    Pillow \
    scipy
