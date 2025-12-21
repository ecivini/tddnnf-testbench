from theorydd.solvers.solver import SMTEnumerator
from theorydd.formula import get_normalized
from pysmt.smtlib.parser import SmtLibParser
from io import StringIO

import json


def load_mapping(
    normalizer_solver: SMTEnumerator, mapping_path: str
) -> tuple[dict, dict]:
    data = None
    with open(mapping_path, "r") as f:
        data = json.load(f)

    parser = SmtLibParser()
    converter = normalizer_solver.get_converter()

    # Record = [id, atom]
    abstraction = {}
    refinement = {}
    for record in data:
        script = parser.get_script(StringIO(record[1]))
        atom = script.get_last_formula()
        atom = get_normalized(atom, converter)
        abstraction[atom] = record[0]
        refinement[record[0]] = atom

    return abstraction, refinement
