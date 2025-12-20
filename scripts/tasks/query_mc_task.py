from theorydd.solvers.mathsat_total import MathSATTotalEnumerator
from pysmt.shortcuts import read_smtlib
from pysmt.fnode import FNode
from theorydd.formula import get_normalized
from theorydd.tdd.theory_bdd import TheoryBDD
from theorydd.tdd.theory_sdd import TheorySDD

import subprocess

import sys
import os
import time
import json


TIMES_TO_CONSIDER = [
    "lemmas loading time",
    "phi normalization time",
    "BC-S1.2 translation time",
    "refinement serialization time",
    "dDNNF compilation time",
    "pysmt translation time",
]


# Assumes clause and mapping are noramlized already
def tddnnf_mc(nnf_path: str, atoms_num: int) -> tuple[int, float]:
    start = time.time()
    # Run ddnnife
    tool_path = "./tools/ddnnife-x86_64-linux/bin/ddnnife"
    command = f"{tool_path} -i {nnf_path} --total-features {atoms_num} count"
    proc = subprocess.run(
        command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    end = time.time()

    count = int(proc.stdout.decode("utf-8"))
    return count, end - start


# Computes the model count of a formula. Takes the result from
# the generated tlemmas log to save time
# Returns a tuple (count, time_taken_ms)
def smt_mc(tlemmas_path: str) -> tuple[int, float]:
    dir_path = os.path.dirname(tlemmas_path)
    logs_path = os.path.join(dir_path, "logs.json")

    logs = None
    with open(logs_path, "r") as logs_f:
        logs = json.load(logs_f)

    return logs["T-DDNNF"]["Total models"], logs["T-DDNNF"]["All-SMT computation time"]


def tbdd_mc(phi: FNode, tbdd_base_path: str) -> tuple[int, float]:
    tbdd = TheoryBDD(phi, folder_name=tbdd_base_path)

    start = time.time()
    count = tbdd.count_models()
    end = time.time()

    return int(count), end - start


def tsdd_mc(phi: FNode, tsdd_base_path: str) -> tuple[int, float]:
    tsdd = TheorySDD(phi, folder_name=tsdd_base_path)

    start = time.time()
    count = tsdd.count_models()
    end = time.time()

    return int(count), end - start


def main():
    if len(sys.argv) != 7:
        print(
            "Usage: python3 scripts/tasks/query_ce_task.py <input formula> "
            "<base output path> <nnf_path> <tlemmas_path> "
            "<tbdd_base_path> <tsdd_base_path>"
        )
        sys.exit(1)

    # Check base output path exists, otherwise create it
    base_out_path = sys.argv[2]
    if not os.path.exists(base_out_path):
        os.makedirs(base_out_path)

    # Create a normalize solver
    normalizer_solver = MathSATTotalEnumerator()
    normalizer_converter = normalizer_solver.get_converter()

    # Normalize phi and tlemmas
    phi = read_smtlib(sys.argv[1])
    phi = get_normalized(phi, converter=normalizer_converter)

    # Get nnf and mapping path
    nnf_path = sys.argv[3]
    tlemmas_path = sys.argv[4]

    # Extract SMT result
    smt_count, smt_time = smt_mc(tlemmas_path)

    # Compute t-d-DNNF result
    atoms_num = len(phi.get_atoms())
    nnf_count, nnf_time = tddnnf_mc(nnf_path, atoms_num)

    # Compute TBDD result
    tbdd_base_path = sys.argv[5]
    tbdd_count, tbdd_time = (
        tbdd_mc(phi, tbdd_base_path) if tbdd_base_path != "none" else (None, None)
    )

    # Compute TSDD result
    tsdd_base_path = sys.argv[6]
    tsdd_count, tsdd_time = (
        tsdd_mc(phi, tsdd_base_path) if tsdd_base_path != "none" else (None, None)
    )

    logs = {
        "MC": {
            "AllSMT time": smt_time,
            "AllSMT count": smt_count,
            "d-DNNF time": nnf_time,
            "d-DNNF count": nnf_count,
            "T-BDD time": tbdd_time,
            "T-BDD count": tbdd_count,
            "T-SDD time": tsdd_time,
            "T-SDD count": tsdd_count,
        }
    }

    logs_path = os.path.join(base_out_path, "logs.json")
    with open(logs_path, "w+") as out_f:
        json.dump(logs, out_f, indent=4)


if __name__ == "__main__":
    main()
