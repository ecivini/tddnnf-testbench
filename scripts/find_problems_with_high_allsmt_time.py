"""Extract benchmark files based on the ones used by Massimo Michelutti"""

import os
import json
import sys
import shutil


TLEMMAS_DIR = "data/results/merged_all_tlemmas_projected"
PROBLEMS_DIR = "data/michelutti_tdds"
RESULT_DIR = "data/high_all_smt_time_problems"
DEFAULT_MIN_TIME = 100.0
DEFAULT_TIMEOUT = 3600.0


def scan_tlemmas(min_allsmt_time: float) -> list:
    problems = []

    for root, _, files in os.walk(TLEMMAS_DIR):
        for file in files:
            if file != "logs.json":
                continue

            file_path = os.path.join(root, file)
            with open(file_path, "r") as logs_f:
                data = json.load(logs_f)
                allsmt_time = data["T-DDNNF"]["All-SMT computation time"]
                if allsmt_time >= min_allsmt_time and allsmt_time < DEFAULT_TIMEOUT:
                    problems.append(file_path)

    return problems


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        print(
            "usage: python3 scripts/find_problems_with_high_allsmt_time.py <optional: min time>"
        )
        sys.exit(0)

    min_time = DEFAULT_MIN_TIME
    if len(sys.argv) == 2:
        min_time = float(sys.argv[1])

    for lemmas_file in scan_tlemmas(min_time):
        # Recreate phi file path
        lemmas_file = lemmas_file.replace(TLEMMAS_DIR, PROBLEMS_DIR)

        pieces = lemmas_file.split("/")[:-1]
        phi_path = "/".join(pieces) + ".smt2"

        target_path = phi_path.replace(PROBLEMS_DIR, RESULT_DIR)

        target_dir = os.path.dirname(target_path)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        shutil.copyfile(phi_path, target_path)
