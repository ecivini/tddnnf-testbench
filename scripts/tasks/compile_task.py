from theorydd.tddnnf.theory_ddnnf import TheoryDDNNF
from pysmt.shortcuts import read_smtlib

import sys
import os
import json
import time

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 scripts/tasks/compile_tasks.py <input formula> <base output path>")
        sys.exit(1)

    # Check base output path exists, otherwise create it
    if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2])

    phi = read_smtlib(sys.argv[1])

    logger = {}
    tddnnf_file_path = os.path.join(sys.argv[2], "tddnnf.smt2")

    start = time.time()
    _ = TheoryDDNNF(
        phi,
        computation_logger=logger,
        out_path=tddnnf_file_path
    )
    time_taken = time.time() - start
    logger["T-DDNNF"]["Total time"] = time_taken

    log_path = os.path.join(sys.argv[2], "logs.json")
    with open(log_path, "w") as log_file:
        json.dump(logger, log_file, indent=4)   

if __name__ == "__main__":
    main()
