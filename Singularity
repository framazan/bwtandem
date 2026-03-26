# Singularity definition file for bwt-algorithm
# This file builds a container with all dependencies and compiles Cython extensions

Bootstrap: docker
From: ubuntu:22.04

%labels
    Author Filip Ramazan
    Version v1.0
    Description BWT-based tandem repeat finder

%environment
    LC_ALL=C.UTF-8
    LANG=C.UTF-8
    PYTHONUNBUFFERED=1

%post
    # Install system dependencies
    apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python3-setuptools python3-wheel \
        build-essential cython3 \
        git \
        libopenblas-dev \
        && rm -rf /var/lib/apt/lists/*

    # Install Python dependencies
    pip3 install --upgrade pip
    pip3 install numpy numba pydivsufsort

    # Copy code into /opt/bwt-algorithm (done at build time)
    mkdir -p /opt/bwt-algorithm
    cp -r /mnt/* /opt/bwt-algorithm/
    cd /opt/bwt-algorithm/src

    # Compile Cython extension (if present)
    if [ -f _accelerators.pyx ]; then 
        cython3 _accelerators.pyx && gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python3.10 -I/usr/local/lib/python3.10/dist-packages/numpy/core/include -o _accelerators.so _accelerators.c
    fi

%files
    . /mnt

%runscript
    exec python3 /opt/bwt-algorithm/src/main.py "$@"
