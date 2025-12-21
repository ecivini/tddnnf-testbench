from theorydd.solvers.mathsat_total import MathSATTotalEnumerator
from pysmt.shortcuts import read_smtlib, And, Not, is_unsat, Symbol, BOOL
from pysmt.smtlib.printers import SmtPrinter
from query_utils import generate_ce_cubes, get_normalized
from nnf_utils import load_mapping
from pysmt.fnode import FNode
from io import StringIO

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
    nnf_path: str,
    cube: FNode,
    mapping: dict[FNode, int],
) -> tuple[bool, float]:
    """
    Checks whether the d-DNNF in nnf_path entails the given cube.

    Returns:
        (entailed: bool, runtime: float)

    Semantics:
        Let C = l1 ∧ ... ∧ ln.
        Returns True iff for all i, Δ ∧ ¬li is UNSAT.
    """
    tool_path = "./tools/ddnnife-x86_64-linux/bin/ddnnife"

    # Extract cube literals
    literals = list(cube.args()) if cube.is_and() else [cube]

    # Build queries: one negated literal per line
    queries = []
    for lit in literals:
        is_neg = lit.is_not()
        atom = lit.arg(0) if is_neg else lit

        if atom in mapping:
            var = mapping[atom]
        else:
            var = str(atom)[1:]  # Removes the initial v
            assert int(var) in mapping.values()

        # Negate cube literal:
        #   x   -> -x
        #   ¬x  ->  x
        query_lit = f"{var}" if is_neg else f"-{var}"
        queries.append(query_lit)

    query_file = "/tmp/query.txt"
    with open(query_file, "w") as f:
        f.write("\n".join(queries))

    start = time.time()
    command = [tool_path, "-i", nnf_path, "sat", query_file]
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    end = time.time()

    if proc.returncode != 0:
        raise RuntimeError(f"ddnnife failed:\n{proc.stderr}")

    # Each line: (query, true|false)
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        _, sat = line.split(",")
        if sat.strip() == "true":
            # A model falsifying one cube literal exists
            return False, end - start

    # All negated literals UNSAT
    return True, end - start


# Check if a cube phi entails phi
# Returns a tuple (entailed, time_taken_ms)
def smt_ce(phi: FNode, cube: FNode) -> tuple[bool, float]:
    start = time.time()
    phi_and_not_clause = And(phi, Not(cube))
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
    base_out_path = sys.argv[2]

    # logger = {}

    # Create a normalize solver
    normalizer_solver = MathSATTotalEnumerator()
    normalizer_converter = normalizer_solver.get_converter()

    # Normalize phi and tlemmas
    phi = read_smtlib(sys.argv[1])
    phi = get_normalized(phi, converter=normalizer_converter)

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
    for cube in cubes:
        # SMT test
        smt_result, smt_time = smt_ce(phi, cube)
        print("SMT:", smt_result, smt_time)

        nnf_result, nnf_time = tddnnf_ce(nnf_path, cube, abstraction)
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
