# Python 3.10 base image
FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /workspace

# Install system dependencies (including Rust for building sudachipy on ARM64)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add Rust to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy requirements file
COPY requirements.txt /workspace/requirements.txt

# Upgrade pip to ensure prebuilt wheels are used when available
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Verify critical dependencies
RUN python -c "import spacy, pandas, numpy, pytest; print('Dependencies installed successfully')"

# Set default user (non-root for security)
RUN useradd -m -s /bin/bash vscode
USER vscode

# Set environment variables
ENV PYTHONUNBUFFERED=1
