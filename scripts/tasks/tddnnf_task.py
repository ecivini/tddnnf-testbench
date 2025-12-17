from theorydd.tddnnf.theory_ddnnf import TheoryDDNNF
from pysmt.shortcuts import read_smtlib
from theorydd.solvers.mathsat_total import MathSATTotalEnumerator

import sys
import os
import json
import time


TIMES_TO_CONSIDER = [
    "lemmas loading time",
    "phi normalization time",
    "BC-S1.2 translation time",
    "refinement serialization time",
    "dDNNF compilation time",
    "pysmt translation time",
]


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python3 scripts/tasks/compile_task.py <input formula> "
            "<base output path> <tlemmas path>"
        )
        sys.exit(1)

    # Check base output path exists, otherwise create it
    if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2])

    tlemmas_path = sys.argv[3]
    if not os.path.isfile(tlemmas_path):
        print("[-] Invalid tlemmas path")
        sys.exit(1)

    phi = read_smtlib(sys.argv[1])
    solver = MathSATTotalEnumerator()

    logger = {}
    start = time.time()
    try:
        _ = TheoryDDNNF(
            phi,
            computation_logger=logger,
            base_out_path=sys.argv[2],
            stop_after_allsmt=False,
            store_tlemmas=False,
            load_lemmas=tlemmas_path,
            sat_result=True,
            solver=solver,
        )
    except Exception as e:
        print(f"[-] Exception during compilation of {sys.argv[1]}")
        print(e)
        sys.exit(1)
    total_time = time.time() - start

    # Compute effective time
    effective_time = 0.0
    for key, value in logger["T-DDNNF"].items():
        if key in TIMES_TO_CONSIDER:
            effective_time += value

    logger["T-DDNNF"]["Effective time"] = effective_time

    # This includes also the time to store the output files
    logger["T-DDNNF"]["Total time"] = total_time

    log_path = os.path.join(sys.argv[2], "logs.json")
    with open(log_path, "w") as log_file:
        json.dump(logger, log_file, indent=4)


if __name__ == "__main__":
    main()
