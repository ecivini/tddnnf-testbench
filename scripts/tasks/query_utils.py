from pysmt.fnode import FNode
from pysmt.shortcuts import Not, Or
from theorydd.solvers.solver import SMTEnumerator
from theorydd.formula import get_normalized
import random


def generate_ce_clauses(
    solver: SMTEnumerator, phi: FNode, seed: int = 42
) -> list[FNode]:
    random.seed(seed)

    clauses = set()

    # Compute the number of clauses to generate
    atoms = list(phi.get_atoms())
    atoms_num = len(atoms)
    clauses_num = _compute_clauses_num(atoms_num)

    while len(clauses) < clauses_num:
        clause = _compute_clause(solver, atoms)
        if clause in clauses:
            continue
        clauses.add(clause)

    clauses = list(clauses)

    # Sort clauses based on the number of literals
    clauses = sorted(clauses, key=lambda clause: len(clause.args()))

    return clauses


# TODO: Find appropriate function to compute the number of clauses
def _compute_clauses_num(atoms_num: int) -> int:
    return 10 * atoms_num


def _compute_clause(solver: SMTEnumerator, atoms: list[FNode]) -> FNode:
    converter = solver.get_converter()

    atoms_num = len(atoms)
    clause_size = random.randint(1, _max_clause_size(atoms_num))

    literals = set()
    used_atoms = set()
    while len(literals) < clause_size:
        literal, atom_index = _new_literal(atoms)
        if atom_index in used_atoms:
            continue
        used_atoms.add(atom_index)
        literals.add(literal)

    clause = Or(literals)
    clause = get_normalized(clause, converter)

    return clause


def _max_clause_size(atoms_num: int) -> int:
    return atoms_num // 3


def _new_literal(atoms: list[FNode]) -> tuple[FNode, int]:
    # Generate polarity randomly
    polarity_prob = random.randint(1, 100)
    positive = polarity_prob > 50

    # Pick random atom
    index = random.randint(0, len(atoms) - 1)
    literal = atoms[index] if positive else Not(atoms[index])

    return literal, index
