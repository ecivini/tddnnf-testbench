""" Extract benchmark files based on the ones used by Massimo Michelutti """

import os 
import json
import shutil

from typing import List

DATA_TARGET_OUTPUT_FOLDER = "output_tbdd_total_new"
DATA_ROOT_PATHS = [
    os.path.join("data/michelutti_tdds/ldd_randgen", DATA_TARGET_OUTPUT_FOLDER),
    os.path.join("data/michelutti_tdds/randgen", DATA_TARGET_OUTPUT_FOLDER),
]
BENCHMARK_OUTPUT_PATH = "data/benchmark/"
TIMEOUT_KEY = "timeout"
ALL_SMT_COMPUTATION_TIME_KEY = "All-SMT computation time"

def is_candidate_benchmark(json_file_path: str) -> bool:
    """
    Return True if the given JSON result file should be included as a
    benchmark candidate.

    The function checks:
    - the file extension is `.json`;
    - the JSON does not contain the `TIMEOUT_KEY` (timed-out runs are
        excluded);
    - the JSON contains the `ALL_SMT_COMPUTATION_TIME_KEY` which indicates
        the test was executed successfully.
    """
    if not json_file_path.endswith(".json"):
        return False

    with open(json_file_path, "r") as f:
        data = json.load(f)

    if TIMEOUT_KEY in data:
        print("Skipping timed out:", json_file_path)
        return False

    if ALL_SMT_COMPUTATION_TIME_KEY not in data:
        print("Skipping unrecognized json:", json_file_path)
        return False

    return True

def retrieve_smt2_path(json_file_path: str) -> str:
    """
    Map a JSON result path back to its original `.smt2` file.

    The previous experiments placed JSON results under the
    `DATA_TARGET_OUTPUT_FOLDER` directory while the original SMT2 files are
    kept under a parallel `data` directory. This function performs the
    required string replacements to derive the `.smt2` path from the JSON
    result path.

    Raises:
        ValueError: if the expected output folder name is not present in the
            provided path.
    """
    if DATA_TARGET_OUTPUT_FOLDER not in json_file_path:
        raise ValueError(f"Unexpected json file path: {json_file_path}")

    smt2_file_path = (
        json_file_path
            .replace(DATA_TARGET_OUTPUT_FOLDER, "data")
            .replace(".json", ".smt2")
    )

    return smt2_file_path

def get_candidate_benchmark_files() -> List[str]:
    """
    Walk the output directories of the previous tests and collect `.smt2`
    file paths for JSON results that pass the candidate filter.
    """
    candidates: List[str] = []

    for starting_dir in DATA_ROOT_PATHS:
        for root, _, files in os.walk(starting_dir):
            for file in files:
                json_path = os.path.join(root, file)
                if is_candidate_benchmark(json_path):
                        smt2_file_path = retrieve_smt2_path(json_path)
                        candidates.append(smt2_file_path)

    return candidates

def copy_candidates_to_benchmark(candidates: List[str]) -> None:
    """
    Copy candidate `.smt2` files into the benchmark output folder.
    """
    for candidate in candidates:
        target_dst_path = os.path.join(
            BENCHMARK_OUTPUT_PATH, candidate.replace("data/michelutti_tdds/", "")
        )
        os.makedirs(os.path.dirname(target_dst_path), exist_ok=True)
        shutil.copy(candidate, target_dst_path)

if __name__ == "__main__":
    print("Looking for candidate files...")
    candidate_benchmark_files = get_candidate_benchmark_files()

    print(f"Found {len(candidate_benchmark_files)} candidate files.")
    print("Copying to target output directory ...")

    copy_candidates_to_benchmark(candidate_benchmark_files)

    print("Done.")