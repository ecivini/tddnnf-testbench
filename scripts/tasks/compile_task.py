from theorydd.tddnnf.theory_ddnnf import TheoryDDNNF
from pysmt.shortcuts import read_smtlib

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
    if len(sys.argv) != 5:
        print(
            "Usage: python3 scripts/tasks/compile_tasks.py <input formula> "
            "<base output path> <allsmt_processes> <generate tlemmas only>"
        )
        sys.exit(1)

    # Check base output path exists, otherwise create it
    if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2])

    phi = read_smtlib(sys.argv[1])
    generate_tlemmas_only = True if sys.argv[4].lower() == "true" else False

    logger = {}

    start = time.time()
    try:
        _ = TheoryDDNNF(
            phi,
            computation_logger=logger,
            base_out_path=sys.argv[2],
            parallel_allsmt_procs=int(sys.argv[3]),
            stop_after_allsmt=generate_tlemmas_only,
            store_tlemmas=True
        )
    except Exception:
        print(f"[-] Exception during compilation of {sys.argv[1]}")
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
