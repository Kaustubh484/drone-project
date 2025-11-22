# Use Python 3.10 on Debian 12 (Bookworm) for better stability than Trixie
FROM python:3.10-slim-bookworm

# 1. Install System Dependencies
# We include libgl1 and libglx-mesa0 to fix the PyBullet rendering issues
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ffmpeg \
    libgl1 \
    libglx-mesa0 \
    libgl1-mesa-dri \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python Libraries
# Install core ML/Sim libraries first
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch \
    numpy \
    pandas \
    tqdm \
    gymnasium \
    pybullet \
    matplotlib \
    scipy

# 3. Install gym-pybullet-drones from Source
WORKDIR /opt
RUN git clone https://github.com/utiasDSL/gym-pybullet-drones.git && \
    cd gym-pybullet-drones && \
    pip install -e .

# 4. Set Environment Variables
ENV PYTHONPATH="${PYTHONPATH}:/opt/gym-pybullet-drones"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# 5. Setup Working Directory
WORKDIR /app

# Default command
ENTRYPOINT ["python"]