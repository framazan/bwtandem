FROM --platform=linux/amd64 python:3.13-slim

LABEL author="Filip Ramazan"
LABEL version="v1.0"
LABEL description="BWTandem — BWT-based tandem repeat finder"

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONPATH=/opt/bwtandem
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 1. Install system tools + Python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    procps \
    && pip install --no-cache-dir \
    numpy \
    cython \
    numba \
    setuptools \
    pydivsufsort \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy source code
COPY . /opt/bwtandem/
WORKDIR /opt/bwtandem

# 3. Run all compilation tasks, THEN safely purge the compilers at the very end
RUN python -m src.c_extensions.build && \
    if [ -f src/_accelerators.pyx ]; then \
    python -c "from setuptools import setup, Extension; from Cython.Build import cythonize; import numpy as np; ext_modules = [Extension('src._accelerators', ['src/_accelerators.pyx'], include_dirs=[np.get_include()], extra_compile_args=['-std=c99'])]; setup(script_args=['build_ext', '--inplace'], ext_modules=cythonize(ext_modules, compiler_directives={'language_level': '3'}))"; \
    fi && \
    apt-get purge -y --auto-remove build-essential

ENTRYPOINT ["python3", "-m", "src.main"]