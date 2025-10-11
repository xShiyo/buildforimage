# Start from a base image with PyTorch and CUDA support
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install the libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- SageMaker Specific Instructions ---

# Set an environment variable that tells SageMaker which script to run
ENV SAGEMAKER_PROGRAM train.py

# Copy your training script into the location SageMaker expects
COPY train.py /opt/ml/code/train.py
