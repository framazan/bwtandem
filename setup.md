# Setup environment

1. First, run `conda env create -f environment.yml` and then `conda activate bwt`.
2. Then, if you happen to change the `cython` code, run this to compile it:
```bash
CFLAGS="-I $(python -c 'import numpy; print(numpy.get_include())')" cythonize -i -3 src/_accelerators.pyx
```

# To test ULTRA:

1. Clone: `git clone https://github.com/TravisWheelerLab/ULTRA`
2. Build: Run `cd ULTRA`, `cmake .`, and `make`
3. To add to global path: `echo "export PATH=\$PATH:$PWD" >> ~/.zshrc` and then `source ~/.zshrc` (replace with ~/.bashrc if that is the case).

## Actually run it:

1. Run `ultra --read_all -p 2000 -t 8 -i 2 -d 2 -o <output file> <input file>`.

# To test TRF:

TRF comes installed with the conda/mamba environment above.

1. Run `python scripts/run_trf.py <path of .fa file you want to test>`. For more args relating to TRF parameters, check out `python scripts/run_trf.py -h`.

# To test mreps:

mreps comes installed with the conda/mamba environment above.