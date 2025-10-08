# Master's thesis project

## Scripts

This directory contains the scripts I used to help me complete this project.

### extract_benchmark_files.py

Extracts SMT2 problem files corresponding to successful Michelutti TBDD results and copies them into the [benchmark folder](data/benchmark/). To use it, copy
the results of the tests done by Massimo in a folder called *data/michelutti_tdds*, then run:

```bash
$ cd <root of this repository>
$ python3 scripts/extract_benchmark_files.py 
```
