from theorydd.tdd.theory_bdd import TheoryBDD
from pysmt.shortcuts import read_smtlib
from theorydd.solvers.mathsat_total import MathSATTotalEnumerator

import sys
import os
import json
import time


TIMES_TO_CONSIDER = [
    "lemmas loading time",
    "phi normalization time",
    "phi DD building time",
    "t-lemmas DD building time",
    "DD joining time",
    "fresh T-atoms detection time",
    "variable mapping creation time",
]


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python3 scripts/tasks/tbdd_task.py <input formula> "
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
    tbdd = None
    try:
        tbdd = TheoryBDD(
            phi,
            computation_logger=logger,
            load_lemmas=tlemmas_path,
            sat_result=True,
            solver=solver,
        )
    except Exception as e:
        print(f"[-] Exception during compilation of {sys.argv[1]}")
        print(e)
        sys.exit(1)
    total_time = time.time() - start

    if tbdd:
        tbdd.save_to_folder(sys.argv[2])

    # Compute effective time
    effective_time = 0.0
    for key, value in logger["T-BDD"].items():
        if key in TIMES_TO_CONSIDER:
            effective_time += value

    logger["T-BDD"]["Effective time"] = effective_time

    # This includes also the time to store the output files
    logger["T-BDD"]["Total time"] = total_time

    log_path = os.path.join(sys.argv[2], "logs.json")
    with open(log_path, "w") as log_file:
        json.dump(logger, log_file, indent=4)


if __name__ == "__main__":
    main()
