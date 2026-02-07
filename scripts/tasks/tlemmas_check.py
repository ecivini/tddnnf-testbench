from pysmt.shortcuts import read_smtlib, Solver, Iff, And
from pysmt.fnode import FNode
from theorydd.solvers.mathsat_total import MathSATTotalEnumerator
from theorydd.solvers.mathsat_partial_extended import MathSATExtendedPartialEnumerator
from theorydd.solvers.with_partitioning import WithPartitioningWrapper
from theorydd.formula import get_normalized
from typing import Iterable
from pysmt.oracles import get_logic
import pysmt

from theorydd.walkers.walker_bool_abstraction import BooleanAbstractionWalker
from theorydd.walkers.walker_refinement import RefinementWalker

import sys
import os
import json
import time


def assert_models_are_tsat(phi: FNode, models: list[Iterable[FNode]]) -> None:
    with Solver() as check_solver:
        check_solver.add_assertion(phi)
        for model in models:
            check_solver.push()
            check_solver.add_assertions(model)
            sat = check_solver.solve()
            assert sat, "T-UNSAT model found: {}".format(model)
            check_solver.pop()


def assert_lemmas_are_tvalid(lemmas: list[FNode]):
    with Solver("msat") as check_solver:
        for lemma in lemmas:
            check_solver.push()
            assert check_solver.is_valid(lemma), "Lemma {} is not valid".format(
                lemma.serialize()
            )
            check_solver.pop()


def assert_phi_equiv_phi_and_lemmas(phi: FNode, phi_and_lemmas):
    with Solver("msat") as check_solver:
        assert check_solver.is_valid(
            Iff(phi, phi_and_lemmas)
        ), "Phi and Phi & lemmas are not theory-equivalent"


def process_raw_tlemmas(raw_tlemmas: FNode) -> list[FNode]:
    if raw_tlemmas.is_and():
        return list(raw_tlemmas.args())
    elif raw_tlemmas.is_or():
        return [raw_tlemmas]
    else:
        raise ValueError("Unexpected T-lemmas format")


def gt_model_count(logs: dict) -> int:
    return logs["T-DDNNF"]["Total models"]


def main():
    if len(sys.argv) < 7:
        print(
            "Usage: python3 scripts/tasks/tlemmas_check.py <input formula> "
            "<base output path> <solver_type> <use_projection> <use_partition> "
            "<tlemmas to check path> <ground truth logs path | optional>"
        )
        sys.exit(1)

    # Check base output path exists, otherwise create it
    if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2])

    phi = read_smtlib(sys.argv[1])
    solver_type = sys.argv[3]
    use_projection = True if sys.argv[4].lower() == "true" else False
    use_partition = True if sys.argv[5].lower() == "true" else False

    # Process T-lemmas to check
    tlemmas_path = sys.argv[6]
    if not os.path.isfile(tlemmas_path):
        print("[-] Invalid tlemmas path")
        sys.exit(1)
    tlemmas = process_raw_tlemmas(read_smtlib(tlemmas_path))

    # Read logs file
    gt_logs = None
    if len(sys.argv) >= 8 and os.path.isfile(sys.argv[7]):
        with open(sys.argv[7], "r") as logs_file:
            gt_logs = json.load(logs_file)

    # mc = tlemmas_logs["T-DDNNF"]["Total models"]
    solver = None
    logger = {}
    if solver_type == "sequential":
        solver = MathSATTotalEnumerator(
            project_on_theory_atoms=use_projection,
            computation_logger=logger,
        )
    elif solver_type == "parallel":
        solver = MathSATExtendedPartialEnumerator(
            project_on_theory_atoms=use_projection,
            computation_logger=logger,
            parallel_procs=6,
        )
    else:
        raise ValueError("Unexpected solver type")

    # mc_logger = {}
    # solver_mc = MathSATTotalEnumerator(
    #     project_on_theory_atoms=use_projection,
    #     computation_logger=mc_logger,
    # )

    if use_partition:
        solver = WithPartitioningWrapper(solver, computation_logger=logger)

    start_time = time.time()

    normalize_solver = Solver("msat")
    phi = get_normalized(phi, normalize_solver.converter)

    # if not use_partition:
    #     solver_mc.check_all_sat(
    #         phi,
    #         atoms=list(phi.get_atoms()),
    #         store_models=False,
    #     )

    # ---- Generate lemmas ----
    print("Generating T-lemmas...")
    phi_atoms = list(phi.get_atoms())
    # phi_sat = solver.check_all_sat(phi, atoms=phi_atoms, store_models=True)
    # assert gt_logs is not None or solver.get_models_count() == gt_model_count(
    #     gt_logs
    # ), "Model count should match expected: {}".format(solver.get_models())

    # print("Asserting models are T-sat...")
    # assert_models_are_tsat(phi, solver.get_models())

    # ---- Build Boolean abstraction of phi & lemmas ----
    print("Normalizing T-lemmas...")
    lemmas = [get_normalized(lemma, normalize_solver.converter) for lemma in tlemmas]

    print("building boolean abstraction...")
    phi_and_lemmas = And(phi, And(lemmas))
    phi_and_lemmas_atoms = phi_and_lemmas.get_atoms()
    assert set(phi_atoms) <= phi_and_lemmas_atoms
    bool_walker = BooleanAbstractionWalker(atoms=phi_and_lemmas_atoms)
    phi_and_lemmas_abstr = bool_walker.walk(phi_and_lemmas)
    phi_abstr = bool_walker.walk(phi)
    assert len(phi_abstr.get_atoms()) == len(
        phi_atoms
    ), "Abstraction should preserve atoms of phi"

    # NOTE: Some lemmas introduce fresh Skolem variables, which should be existentially quantified for the lemma to
    # be t-valid.
    # However, MathSAT does not support quantifiers, and will flag these lemmas as non t-valid.
    # Anyway, these new variables only appear in fresh atoms, which are later existentially quantified, so that
    # correctness is preserved.
    # It seems the only case this happens is with arrays (e.g. extensionality lemma), so we skip the following checks
    # in that case.
    if not get_logic(phi).theory.arrays:
        assert_lemmas_are_tvalid(lemmas)
        # assert_phi_equiv_phi_and_lemmas(phi, phi_and_lemmas)

    print("Running AllSAT on Boolean abstraction ...")
    solver_abstr = MathSATTotalEnumerator(project_on_theory_atoms=False)
    abstr_sat = solver_abstr.check_all_sat(
        phi_and_lemmas_abstr,
        atoms=list(phi_abstr.get_atoms()),
        store_models=True,
    )
    print("Checking models number match ...")
    assert abstr_sat, "Abstracted formula with lemmas should be satisfiable"
    # assert (
    #     use_partition or solver_abstr.get_models_count() == solver_mc.get_models_count()
    # ), f"Model count should match expected: {solver_abstr.get_models_count()} vs {solver_mc.get_models_count()}"

    # Check phi_and_lemmas is t-reduced
    print("Refining Boolean abstraction ...")
    refinement_walker = RefinementWalker(abstraction=bool_walker.abstraction)
    refined_models = [
        [refinement_walker.walk(lit) for lit in model]
        for model in solver_abstr.get_models()
    ]

    assert_models_are_tsat(phi, refined_models)

    if gt_logs is not None:
        assert len(refined_models) == gt_model_count(
            gt_logs
        ), "Refined models number should match ground truth"

    logger["T-LEMMAS CHECK"] = {}
    logger["T-LEMMAS CHECK"]["Total time"] = time.time() - start_time

    log_path = os.path.join(sys.argv[2], "logs.json")
    with open(log_path, "w") as log_file:
        json.dump(logger, log_file, indent=4)


if __name__ == "__main__":
    main()
