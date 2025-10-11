# Start from a base image with PyTorch and CUDA support
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install the libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- SageMaker Specific Instructions ---
# Copy the training script
COPY train.py /opt/ml/code/train.py

# Define the entry point for the container
ENV SAGEMAKER_PROGRAM train.py
