from pysmt.fnode import FNode
from pysmt.shortcuts import Not, And
from theorydd.solvers.solver import SMTEnumerator
from theorydd.formula import get_normalized
import random


def generate_ce_cubes(solver: SMTEnumerator, phi: FNode, seed: int = 42) -> list[FNode]:
    random.seed(seed)

    cubes = set()

    # Compute the number of cubes to generate
    atoms = list(phi.get_atoms())
    atoms_num = len(atoms)
    cubes_num = _compute_cubes_num(atoms_num)

    while len(cubes) < cubes_num:
        cube = _compute_cube(solver, atoms)
        if cube in cubes:
            continue
        cubes.add(cube)

    cubes = list(cubes)

    # Sort cubes based on the number of literals
    cubes = sorted(cubes, key=lambda cube: len(cube.args()))

    return cubes


# TODO: Find appropriate function to compute the number of cubes
def _compute_cubes_num(atoms_num: int) -> int:
    return 10 * atoms_num


def _compute_cube(solver: SMTEnumerator, atoms: list[FNode]) -> FNode:
    converter = solver.get_converter()

    atoms_num = len(atoms)
    cube_size = random.randint(_min_cube_size(atoms_num), _max_cube_size(atoms_num))

    literals = set()
    used_atoms = set()
    while len(literals) < cube_size:
        literal, atom_index = _new_literal(atoms)
        if atom_index in used_atoms:
            continue
        used_atoms.add(atom_index)
        literals.add(literal)

    cube = And(literals)
    cube = get_normalized(cube, converter)

    return cube


def _max_cube_size(atoms_num: int) -> int:
    return atoms_num // 3


def _min_cube_size(atoms_num: int) -> int:
    return min(atoms_num, 3)


def _new_literal(atoms: list[FNode]) -> tuple[FNode, int]:
    # Generate polarity randomly
    polarity_prob = random.randint(1, 100)
    positive = polarity_prob > 50

    # Pick random atom
    index = random.randint(0, len(atoms) - 1)
    literal = atoms[index] if positive else Not(atoms[index])

    return literal, index
