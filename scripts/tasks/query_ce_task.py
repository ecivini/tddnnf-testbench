from theorydd.solvers.mathsat_total import MathSATTotalEnumerator
from theorydd.solvers.solver import SMTEnumerator
from pysmt.shortcuts import read_smtlib, Symbol, BOOL
from pysmt.smtlib.printers import SmtPrinter
from query_utils import generate_ce_cubes, get_normalized
from nnf_utils import load_mapping
from pysmt.fnode import FNode
from pysmt.shortcuts import Solver
from io import StringIO
from ddnnife import Ddnnf
from theorydd.tdd.theory_bdd import TheoryBDD
from theorydd.tdd.theory_sdd import TheorySDD

import sys
import os
import time
import json


INCREMENTAL_SMT = True


TIMES_TO_CONSIDER = [
    "lemmas loading time",
    "phi normalization time",
    "BC-S1.2 translation time",
    "refinement serialization time",
    "dDNNF compilation time",
    "pysmt translation time",
]


# Assumes cube is already the negated cube
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


# Check if phi entails C
# cube is already assumed to be equal to not(C)
# Returns a tuple (entailed, time_taken_ms)
def smt_ce(
    cube: FNode, phi_reading_time: float = 0, solver: Solver = None, phi: FNode = None
) -> tuple[bool, float]:
    start = time.time()
    if INCREMENTAL_SMT:
        assert solver is not None, "Incremental SMT without solver"
        solver.push()
        solver.add_assertion(cube)
        entailed = not solver.solve()
        solver.pop()
    else:
        assert phi is not None, "Non incremental SMT without PHI"
        solver = Solver()
        solver.add_assertion(phi)
        solver.add_assertion(cube)
        entailed = not solver.solve()
    end = time.time()

    return entailed, end - start + phi_reading_time


# Check if phi entails C
# cube is already assumed to be equal to not(C)
# Returns a tuple (entailed, time_taken_ms)
def tsdd_ce(
    phi: FNode, cube: FNode, base_path: str, normalizer: SMTEnumerator
) -> tuple[bool, float]:
    tsdd = TheorySDD(phi, folder_name=base_path, solver=normalizer)

    start = time.time()

    literals = list(cube.args()) if cube.is_and() else [cube]
    conditions = []
    for lit in literals:
        is_neg = lit.is_not()
        atom = lit.arg(0) if is_neg else lit

        abstr_atom = tsdd.abstraction[atom]
        label = -abstr_atom if is_neg else abstr_atom

        conditions.append(label)

    entailed = not tsdd.is_sat_with_condition(conditions)
    end = time.time()

    return entailed, end - start


# Check if phi entails C
# cube is already assumed to be equal to not(C)
# Returns a tuple (entailed, time_taken_ms)
def tbdd_ce(
    phi: FNode, cube: FNode, base_path: str, normalizer: SMTEnumerator
) -> tuple[bool, float]:
    tsdd = TheoryBDD(phi, folder_name=base_path, solver=normalizer)

    start = time.time()

    literals = list(cube.args()) if cube.is_and() else [cube]
    conditions = []
    for lit in literals:
        is_neg = lit.is_not()
        atom = lit.arg(0) if is_neg else lit

        abstr_atom = tsdd.abstraction[atom]
        label = f"-{abstr_atom}" if is_neg else abstr_atom

        conditions.append(label)

    entailed = not tsdd.is_sat_with_condition(conditions)
    end = time.time()

    return entailed, end - start


def main():
    if len(sys.argv) != 6:
        print(
            "Usage: python3 scripts/tasks/query_ce_task.py <input formula> "
            "<base output path> <nnf_path> <tsdd_path> <tbdd_path>"
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

    print("PHI:", sys.argv[1])
    print("NNF:", sys.argv[3])

    # Normalize phi and tlemmas
    read_phi_start = time.time()
    phi = read_smtlib(sys.argv[1])
    phi = get_normalized(phi, converter=normalizer_converter)
    read_phi_time = time.time() - read_phi_start

    # Get nnf and mapping path
    nnf_path = sys.argv[3]
    mapping_path = os.path.dirname(nnf_path)
    mapping_path = os.path.join(mapping_path, "mapping", "mapping.json")
    abstraction, _ = load_mapping(normalizer_solver, mapping_path)

    # Get DDs paths
    tsdd_base_path = sys.argv[4]
    use_tsdds = tsdd_base_path != "none"
    tbdd_base_path = sys.argv[5]
    use_tbdds = tbdd_base_path != "none"

    # Compute abstraction in PySMT
    pysmt_abstraction = {}
    for atom in abstraction:
        pysmt_abstraction[atom] = Symbol(f"v{abstraction[atom]}", BOOL)

    # Generate clauses
    cubes_num = min(50, len(phi.get_atoms()))
    cubes = generate_ce_cubes(normalizer_solver, phi, desired_cub_num=cubes_num)

    # Run CE test for both SMT and t-d-DNNF
    computations = {}

    ddnnf = Ddnnf.from_file(nnf_path, None)

    solver = Solver("msat")
    solver.add_assertion(phi)

    for cube in cubes:
        # SMT test
        smt_result, smt_time = smt_ce(cube, phi_reading_time=0, solver=solver, phi=phi)
        print("SMT:", smt_result, smt_time)

        nnf_result, nnf_time = tddnnf_ce(ddnnf, cube, abstraction)
        print("NNF:", nnf_result, nnf_time)

        tsdd_result, tsdd_time = None, None
        if use_tsdds:
            tsdd_result, tsdd_time = tsdd_ce(
                phi, cube, tsdd_base_path, normalizer_solver
            )
            print("T-SDD:", tsdd_result, tsdd_time)

        tbdd_result, tbdd_time = None, None
        if use_tbdds:
            tbdd_result, tbdd_time = tbdd_ce(
                phi, cube, tbdd_base_path, normalizer_solver
            )
            print("T-BDD:", tbdd_result, tbdd_time)

        assert smt_result == nnf_result, "Problem with " + str(cube)
        assert (
            not use_tsdds or tsdd_result == nnf_result
        ), "Problem with T-SDD for" + str(cube)
        assert (
            not use_tbdds or tbdd_result == nnf_result
        ), "Problem with T-BDD for" + str(cube)

        log = {
            "SMT result": smt_result,
            "SMT time": smt_time,
            "d-DNNF result": nnf_result,
            "d-DNNF time": nnf_time,
            "T-SDD result": tsdd_result,
            "T-SDD time": tsdd_time,
            "T-BDD result": tbdd_result,
            "T-BDD time": tbdd_time,
        }
        computations[str(cube)] = log

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
