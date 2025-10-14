# Master's thesis project

## Dependencies

This repository depends mostly on [tddnnf](https://github.com/ecivini/tddnnf), which is an implementation of ... (TODO: Add references to algorithm), based on the [TheoryConsistentDecisionDiagrams](https://github.com/MaxMichelutti/TheoryConsistentDecisionDiagrams) package.

In order to install it, run:
```bash
$ virtualenv env
$ source env/bin/activate
$ pip3 install theorydd@git+https://github.com/ecivini/tddnnf@main
```

## Scripts

This directory contains the scripts I used to help me complete this project.

### extract_benchmark_files.py

Extracts SMT2 problem files corresponding to successful Michelutti TBDD results and copies them into the [benchmark folder](data/benchmark/). To use it, copy
the results of the tests done by Massimo in a folder called *data/michelutti_tdds*, then run:

```bash
$ cd <root of this repository>
$ python3 scripts/extract_benchmark_files.py 
```
