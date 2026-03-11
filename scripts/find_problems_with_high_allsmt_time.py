"""Extract benchmark files based on the ones used by Massimo Michelutti"""

import os
import json
import sys
import shutil


TLEMMAS_DIR = "data/results/merged_all_tlemmas_projected"
PROBLEMS_DIR = "data/michelutti_tdds"
# RESULT_DIR = "data/high_all_smt_time_problems_test"
RESULT_DIR = "data/high_all_smt_time_problems_proj_0_373s_to_1s{0}"
# RESULT_DIR = "test{0}"
DEFAULT_MIN_TIME = 0.373
DEFAULT_TIMEOUT = 1.0
SERVER_NAMES = ["_qui", "_quo", "_qua"]


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

    print(f"Found {len(problems)} problems")

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

    server_index = 0
    for lemmas_file in scan_tlemmas(min_time):
        # Recreate phi file path
        lemmas_file = lemmas_file.replace(TLEMMAS_DIR, PROBLEMS_DIR)

        pieces = lemmas_file.split("/")[:-1]
        phi_path = "/".join(pieces) + ".smt2"

        result_path = RESULT_DIR.format(SERVER_NAMES[server_index])
        target_path = phi_path.replace(PROBLEMS_DIR, result_path)

        target_dir = os.path.dirname(target_path)
        print(target_dir)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        server_index = (server_index + 1) % len(SERVER_NAMES)

        shutil.copyfile(phi_path, target_path)
