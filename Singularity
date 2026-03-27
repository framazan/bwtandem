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
    PYTHONPATH=/opt/bwt-algorithm:${PYTHONPATH}

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
    pip3 install numpy numba pydivsufsort Cython

    cd /opt/bwt-algorithm

    # Compile Cython extension (if present)
    if [ -f src/_accelerators.pyx ]; then
        python3 - <<'PY'
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "src._accelerators",
        ["src/_accelerators.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    script_args=["build_ext", "--inplace"],
    ext_modules=cythonize(ext_modules, compiler_directives={"language_level": "3"}),
)
PY
    fi

%files
    . /opt/bwt-algorithm

%runscript
    exec python3 -m src.main "$@"
