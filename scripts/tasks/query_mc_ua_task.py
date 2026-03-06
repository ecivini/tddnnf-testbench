from enumerators.solvers.mathsat_total import MathSATTotalEnumerator
from enumerators.solvers.solver import SMTEnumerator
from pysmt.shortcuts import read_smtlib, And
from pysmt.fnode import FNode
from theorydd.formula import get_normalized
from theorydd.tdd.theory_bdd import TheoryBDD
from theorydd.tdd.theory_sdd import TheorySDD
from ddnnife import Ddnnf
from query_utils import generate_ce_cubes
from nnf_utils import load_mapping

import multiprocessing

import sys
import os
import time
import json

QUERY_NUMBER_LIMIT = 10
DEFAULT_TIMEOUT_SEC = 600


# Computes the model count of a formula. Takes the result from
# the generated tlemmas log to save time
# Returns a tuple (count, time_taken_ms)
def smt_mc_ua_worker(phi: FNode, query: FNode, queue: multiprocessing.Queue) -> None:
    # use stack interface to add query
    phi_and_query = And(phi, query)
    logger = {}
    solver = MathSATTotalEnumerator(
        computation_logger=logger, project_on_theory_atoms=False
    )
    atoms = phi.get_atoms()

    start = time.time()
    solver.check_all_sat(phi_and_query, atoms, store_models=False)
    end = time.time()

    count = solver.get_models_count()

    queue.put((count, end - start))


def smt_mc_ua_builder(phi: FNode, query: FNode) -> tuple[int, float]:
    args = " ".join(sys.argv)
    command = f"python3 {args} worker"
    command = command.split()

    queue = multiprocessing.Queue()
    try:
        process = multiprocessing.Process(
            target=smt_mc_ua_worker,
            args=(phi, query, queue),
        )
        process.start()
        process.join(timeout=DEFAULT_TIMEOUT_SEC)

        if process.is_alive():
            print("Timeout reached for query:", str(query))
            process.terminate()  # kill worker
            process.join()  # clean up
            return -1, DEFAULT_TIMEOUT_SEC
    except Exception:
        print("Unhandled error with", query)
        return -2, -2

    assert not queue.empty(), "Queue should not be empty"
    result, total_time = queue.get()

    # print("allsmt", result, total_time)

    return result, total_time


def tddnnf_mc_ua(
    ddnnf: Ddnnf, cube: FNode, mapping: dict[FNode, int]
) -> tuple[int, float]:
    literals = list(cube.args()) if cube.is_and() else [cube]
    assumptions = []
    for lit in literals:
        is_neg = lit.is_not()
        atom = lit.arg(0) if is_neg else lit

        if atom in mapping:
            var = mapping[atom]
        else:
            var = str(atom)[1:]  # Removes the initial v
            assert int(var) in mapping.values()

        query_lit = -int(var) if is_neg else int(var)
        assumptions.append(query_lit)

    start = time.time()
    count = ddnnf.as_mut().count_multiple(assumptions=assumptions, variables=[])
    end = time.time()
    # print("ddnnf", count, end - start)

    return int(str(count[0])), end - start


def tbdd_mc_ua(
    phi: FNode, query: FNode, tbdd_base_path: str, normalizer: SMTEnumerator
) -> tuple[int, float]:
    tbdd = TheoryBDD(phi, folder_name=tbdd_base_path, solver=normalizer)

    start = time.time()
    literals = list(query.args()) if query.is_and() else [query]
    for lit in literals:
        is_neg = lit.is_not()
        atom = lit.arg(0) if is_neg else lit

        abstr_atom = tbdd.abstraction[atom]
        label = f"-{abstr_atom}" if is_neg else abstr_atom

        tbdd.condition(label)

    count = tbdd.count_models()
    end = time.time()

    return int(count), end - start


def tsdd_mc_ua(
    phi: FNode, query: FNode, tsdd_base_path: str, normalizer: SMTEnumerator
) -> tuple[int, float]:
    tsdd = TheorySDD(phi, folder_name=tsdd_base_path, solver=normalizer)

    start = time.time()
    literals = list(query.args()) if query.is_and() else [query]
    for lit in literals:
        is_neg = lit.is_not()
        atom = lit.arg(0) if is_neg else lit

        abstr_atom = tsdd.abstraction[atom]
        label = -abstr_atom if is_neg else abstr_atom

        tsdd.condition(label)

    count = tsdd.count_models()
    end = time.time()

    return int(count), end - start


def main():
    if len(sys.argv) != 6:
        print(
            "Usage: python3 scripts/tasks/query_ce_task.py <input formula> "
            "<base output path> <nnf_path> <tsdd_path> <tbdd_path>"
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
    ddnnf = Ddnnf.from_file(nnf_path, len(phi.get_atoms()))

    mapping_path = os.path.dirname(nnf_path)
    mapping_path = os.path.join(mapping_path, "mapping", "mapping.json")
    abstraction, _ = load_mapping(normalizer_solver, mapping_path)

    # Get DDs paths
    tsdd_base_path = sys.argv[4]
    use_tsdds = tsdd_base_path != "none"
    tbdd_base_path = sys.argv[5]
    use_tbdds = tbdd_base_path != "none"

    logs = {"MC": {}}

    # Generate 50 queries so that it's unlikely to find more than 40 unsat cases
    queries = generate_ce_cubes(solver=normalizer_solver, phi=phi, desired_cub_num=50)
    sat_queries = 0
    next_query = 0

    while sat_queries < QUERY_NUMBER_LIMIT and next_query < len(queries):
        query = queries[next_query]
        query = get_normalized(query, normalizer_converter)

        # Extract SMT result
        smt_count, smt_time = smt_mc_ua_builder(phi, query)

        # Compute t-d-DNNF result
        nnf_count, nnf_time = tddnnf_mc_ua(ddnnf, query, abstraction)

        tbdd_result, tbdd_time = None, None
        if use_tbdds:
            tbdd_result, tbdd_time = tbdd_mc_ua(
                phi, query, tbdd_base_path, normalizer_solver
            )
            print("T-BDD:", tbdd_result, tbdd_time)
            assert nnf_count == tbdd_result, "d-DNNF and OBDD counts should match"

        tsdd_result, tsdd_time = None, None
        if use_tsdds:
            tsdd_result, tsdd_time = tsdd_mc_ua(
                phi, query, tsdd_base_path, normalizer_solver
            )
            print("T-SDD:", tsdd_result, tsdd_time)
            assert nnf_count == tsdd_result, "d-DNNF and OBDD counts should match"

        assert (
            smt_count < 0 or smt_count == nnf_count
        ), f"Counts should match: {smt_count} vs {nnf_count}"

        # Adding None's for DDs to keep compatibility
        if nnf_count > 0:
            log = {
                "AllSMT time": smt_time,
                "AllSMT count": smt_count,
                "d-DNNF time": nnf_time,
                "d-DNNF count": nnf_count,
                "T-BDD time": tbdd_time,
                "T-BDD count": tbdd_result,
                "T-SDD time": tsdd_time,
                "T-SDD count": tsdd_result,
                "AllSMT timeout": smt_count < 0,
            }
            logs["MC"][str(query)] = log
            sat_queries += 1

        next_query += 1

    logs_path = os.path.join(base_out_path, "logs.json")
    with open(logs_path, "w+") as out_f:
        json.dump(logs, out_f, indent=4)


if __name__ == "__main__":
    main()
