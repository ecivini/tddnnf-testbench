from theorydd.solvers.mathsat_total import MathSATTotalEnumerator
from pysmt.shortcuts import read_smtlib, And, Not, is_unsat, Symbol, BOOL
from pysmt.smtlib.printers import SmtPrinter
from query_utils import generate_ce_cubes, get_normalized
from nnf_utils import load_mapping
from pysmt.fnode import FNode
from io import StringIO
from ddnnife import Ddnnf

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
def tddnnf_ce(
    ddnnf: Ddnnf,
    cube: FNode,
    mapping: dict[FNode, int],
) -> tuple[bool, float]:
    """
    Checks whether the d-DNNF in nnf_path entails not(cube).

    Returns:
        (entailed: bool, runtime: float)
    """
    # tool_path = "./tools/ddnnife-x86_64-linux/bin/ddnnife"

    # # Extract cube literals
    # literals = list(cube.args()) if cube.is_and() else [cube]

    # # Build queries: one negated literal per line
    # query = ""
    # for lit in literals:
    #     is_neg = lit.is_not()
    #     atom = lit.arg(0) if is_neg else lit

    #     if atom in mapping:
    #         var = mapping[atom]
    #     else:
    #         var = str(atom)[1:]  # Removes the initial v
    #         assert int(var) in mapping.values()

    #     query_lit = f"-{var}" if is_neg else f"{var}"
    #     query += query_lit + " "
    # query = query.strip()

    # query_file = "/tmp/query.txt"
    # with open(query_file, "w") as f:
    #     f.write(query)

    # start = time.time()
    # command = [tool_path, "-i", nnf_path, "sat", query_file]
    # proc = subprocess.run(
    #     command,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    #     text=True,
    # )

    # end = time.time()

    # if proc.returncode != 0:
    #     raise RuntimeError(f"ddnnife failed:\n{proc.stderr}")

    # # Each line: (query, true|false)
    # for line in proc.stdout.splitlines():
    #     if not line.strip():
    #         continue
    #     _, sat = line.split(",")
    #     if sat.strip() == "true":
    #         # A model falsifying one cube literal exists
    #         return False, end - start

    # # All negated literals UNSAT
    # return True, end - start
    # TODO: Add number of features instead of None
    literals = list(cube.args()) if cube.is_and() else [cube]
    mapped_literals = []
    for lit in literals:
        is_neg = lit.is_not()
        atom = lit.arg(0) if is_neg else lit

        if atom in mapping:
            var = mapping[atom]
        else:
            var = str(atom)[1:]  # Removes the initial v
            assert int(var) in mapping.values()

        query_lit = -int(var) if is_neg else int(var)
        mapped_literals.append(query_lit)

    start = time.time()
    sat = ddnnf.as_mut().is_sat(mapped_literals)

    return not sat, time.time() - start


# Check if phi entails not(cube)
# Returns a tuple (entailed, time_taken_ms)
def smt_ce(phi: FNode, cube: FNode, phi_reading_time: float = 0) -> tuple[bool, float]:
    start = time.time()
    phi_and_not_clause = And(phi, cube)
    entailed = is_unsat(phi_and_not_clause)
    end = time.time()

    return entailed, end - start + phi_reading_time


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
    base_out_path = sys.argv[2]

    # logger = {}

    # Create a normalize solver
    normalizer_solver = MathSATTotalEnumerator()
    normalizer_converter = normalizer_solver.get_converter()

    # Normalize phi and tlemmas
    read_phi_start = time.time()
    phi = read_smtlib(sys.argv[1])
    phi = get_normalized(phi, converter=normalizer_converter)
    read_phi_time = time.time() - read_phi_start

    # Get nnf and mapping path
    nnf_path = sys.argv[3]
    mapping_path = sys.argv[4]
    abstraction, _ = load_mapping(normalizer_solver, mapping_path)

    # Compute abstraction in PySMT
    pysmt_abstraction = {}
    for atom in abstraction:
        pysmt_abstraction[atom] = Symbol(f"v{abstraction[atom]}", BOOL)

    # Generate clauses
    cubes = generate_ce_cubes(normalizer_solver, phi)

    # Run CE test for both SMT and t-d-DNNF
    computations = []
    ddnnf = Ddnnf.from_file(nnf_path, None)
    for cube in cubes:
        # SMT test
        smt_result, smt_time = smt_ce(phi, cube, phi_reading_time=0)
        print("SMT:", smt_result, smt_time)

        nnf_result, nnf_time = tddnnf_ce(ddnnf, cube, abstraction)
        print("NNF:", nnf_result, nnf_time)

        assert smt_result == nnf_result, "Problem with " + str(cube)

        log = {
            "SMT result": smt_result,
            "SMT time": smt_time,
            "d-DNNF result": nnf_result,
            "d-DNNF time": nnf_time,
        }
        computations.append(log)

    # Store logs
    logs_path = os.path.join(base_out_path, "logs.json")
    with open(logs_path, "w+") as logs_f:
        json.dump(computations, logs_f, indent=4)

    # Store cubes
    cubes_path = os.path.join(base_out_path, "cubes.json")
    with open(cubes_path, "w+") as cubes_f:
        cubes_strs = []
        for cube in cubes:
            buf = StringIO()
            printer = SmtPrinter(buf)
            printer.printer(cube)
            cubes_strs.append(buf.getvalue())
        json.dump(cubes_strs, cubes_f, indent=4)


if __name__ == "__main__":
    main()
