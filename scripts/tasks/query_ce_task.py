from theorydd.solvers.mathsat_total import MathSATTotalEnumerator
from pysmt.shortcuts import read_smtlib, And, Not, is_unsat
from query_utils import generate_ce_clauses, get_normalized
from nnf_utils import load_mapping
from pysmt.fnode import FNode

import subprocess

import sys
import os
import time


TIMES_TO_CONSIDER = [
    "lemmas loading time",
    "phi normalization time",
    "BC-S1.2 translation time",
    "refinement serialization time",
    "dDNNF compilation time",
    "pysmt translation time",
]


# Assumes clause and mapping are noramlized already
def tddnnf_ce(nnf_path: str, clause: FNode, mapping: dict) -> tuple[bool, float]:
    start = time.time()
    # Store query file
    with open("/tmp/query.txt", "w+") as query_f:
        query_content = ""
        if clause.is_or():
            for literal in clause.args():
                is_not = literal.is_not()
                atom = literal.arg(0) if is_not else literal
                mapped_atom = ("-" if is_not else "") + f"v{mapping[atom]}"
                query_content += mapped_atom + " "
        else:
            is_not = clause.is_not()
            atom = clause.arg(0) if is_not else clause
            mapped_atom = ("-" if is_not else "") + f"v{mapping[atom]}"
            query_content = mapped_atom

        query_content = query_content.strip()
        query_f.write(query_content)

    # Run ddnnife
    # TODO: Is it possible to run CE queries with ddnnife?
    tool_path = "./tools/ddnnife-x86_64-linux/bin/ddnnife"
    command = f"{tool_path} sat /tmp/query.txt"
    subprocess.run(command.split())
    end = time.time()

    return False, end - start


# Check if clause phi entails clause
# Returns a tuple (entailed, time_taken_ms)
def smt_ce(phi: FNode, clause: FNode) -> tuple[bool, float]:
    start = time.time()
    phi_and_not_clause = And(phi, Not(clause))
    entailed = is_unsat(phi_and_not_clause)
    end = time.time()

    return entailed, end - start


def main():
    if len(sys.argv) != 5:
        print(
            "Usage: python3 scripts/tasks/query_ce_task.py <input formula> "
            "<base output path> <nnf_path> <mapping_path>"
        )
        sys.exit(1)

    # Check base output path exists, otherwise create it
    if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2])

    # logger = {}

    # Create a normalize solver
    normalizer_solver = MathSATTotalEnumerator()
    normalizer_converter = normalizer_solver.get_converter()

    # Normalize phi and tlemmas
    phi = read_smtlib(sys.argv[1])
    phi = get_normalized(phi, converter=normalizer_converter)

    # TODO: Normalize tlemmas

    # Get nnf and mapping path
    nnf_path = sys.argv[3]
    mapping_path = sys.argv[4]
    mapping = load_mapping(normalizer_solver, mapping_path)

    # Generate clauses
    clauses = generate_ce_clauses(normalizer_solver, phi)

    # Run CE test for both SMT and t-d-DNNF
    for clause in clauses:
        # SMT test
        smt_result, smt_time = smt_ce(phi, clause)
        print("SMT:", smt_result, smt_time)

        nnf_result, nnf_time = tddnnf_ce(nnf_path, clause, mapping)
        print("NNF:", nnf_result, nnf_time)
        break

    # start = time.time()
    # try:
    #     _ = TheoryDDNNF(
    #         phi,
    #         computation_logger=logger,
    #         base_out_path=sys.argv[2],
    #         stop_after_allsmt=generate_tlemmas_only,
    #         store_tlemmas=True,
    #         solver=solver,
    #     )
    # except Exception:
    #     print(f"[-] Exception during compilation of {sys.argv[1]}")
    #     sys.exit(1)
    # total_time = time.time() - start

    # # Compute effective time
    # effective_time = 0.0
    # for key, value in logger["T-DDNNF"].items():
    #     if key in TIMES_TO_CONSIDER:
    #         effective_time += value

    # logger["T-DDNNF"]["Effective time"] = effective_time

    # # This includes also the time to store the output files
    # logger["T-DDNNF"]["Total time"] = total_time

    # log_path = os.path.join(sys.argv[2], "logs.json")
    # with open(log_path, "w") as log_file:
    #     json.dump(logger, log_file, indent=4)


if __name__ == "__main__":
    main()
