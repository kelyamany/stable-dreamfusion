FROM nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu20.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

ENV TZ=Europe/Vienna
ENV DEBIAN_FRONTEND=noninteractive

# Install some basic utilities
RUN apt-get update && \
    apt install -y \
    tzdata \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    python3 \
    python3-pip \
    libglfw3-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Install Python Modules
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

RUN pip3 install -r requirements.optional.txt; sh scripts/install_ext.sh || echo "Failed to install opt/ext"

CMD ["python3","-u", "gradio_app.py"]

