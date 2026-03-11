# Experimental evaluation scripts for the paper "d-DNNF Modulo Theories: A General Framework for Polytime SMT Queries"

## Dependencies

```bash
# Install dependencies
$ pip3 install .

# Install MathSAT via PySMT
$ pysmt-install --msat
```

## How to run

### Compiling T-lemmas

In order to compile T-lemmas, use the instructions provided [in this repository](https://github.com/ecivini/tlemmas-enumeration-testbench).

### Compiling T-d-DNNF

First of all, you need to update the config.yaml file.
```bash
$ nano config.yaml
```

In particular, you need to specify `tlemmas_dir` with the directory containing your compiled T-lemmas. Then, you need to configure `benchmarks` with the
paths to the benchmark files to use. They must correspond to the ones
used for T-lemmas enumeration

Then, run the benchmark controller:
```bash
# Compile T-d-DNNF
$ python3 scripts/benchmark_controller.py tddnnfonly <output_path>
```

### Running queries

In your `config.yaml`, you need to specify `benchmark` as before, then
`tddnnf_dir` with the directory containing the output T-d-DNNFs.

Once done, run the benchmark controller:
```bash
# Run MC queries
$ python3 scripts/benchmark_controller.py query_mc <output_folder> 

# Run MC Under Assumptions queries
$ python3 scripts/benchmark_controller.py query_mcua <output_folder> 

# Run CE queries
$ python3 scripts/benchmark_controller.py query_ce <output_folder> 
```

In particular, for CE queries it's possible to decide to run using
SMT or incremental SMT (the default). To change this behavior,
edit the file `scripts/tasks/query_ce_task.py` at line 20.