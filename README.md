# Benchmarks for comparing magnetic simulations in SimPEG

## Get started

In order to run these benchmarks, you need to have a Python distribution like
[Miniforge][miniforge] installed.

Then, clone this repository:

```bash
git clone https://github.com/santisoler/simpeg-benchmarks-magnetic
cd simpeg-benchmarks-magnetic
```

And create a `conda` environment with all the required dependencies for running
these benchmarks:

```bash
conda env create -f environment.yml
```

## Run the benchmarks

All benchmarks can be run by executing the Python scripts in `code`
folder.
<!-- , and through the `benchmark-memory.sh` script. -->

Alternatively, we can run all benchmarks by executing the `run.sh` shell
script:

```bash
bash run.sh
```

> **Important**
> Most of the benchmarks were designed to be run on a machine with 125GB of ram and
> a minimum of 30 threads. If your system don't meet these specs, you can
> modify the scripts to adjust them to your needs.

<!-- > The benchmarks for the "large problem" require more memory: up to ~800GB. -->

[miniforge]: https://github.com/conda-forge/miniforge
