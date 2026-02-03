import json
import os
import shutil


def build_data_path(server: str) -> str:
    # base = f"data/results/{server}_all_tlemmas/{server}_tlemmas_all_1Prob_45Procs/data/benchmark" # noqa
    base = f"data/results/{server}_tlemmas_rand_sequential/{server}_tlemmas_rand_total_sequential/data/benchmark"  # noqa
    return os.path.join(base, server)


DATA_FOLDERS = [
    build_data_path("qui"),
    build_data_path("quo"),
    build_data_path("qua"),
]

ERROR_FILES = [
    "data/results/qua_tlemmas_rand_sequential/qua_tlemmas_rand_total_sequential/errors.json",  # noqa
    "data/results/quo_tlemmas_rand_sequential/quo_tlemmas_rand_total_sequential/errors.json",  # noqa
    "data/results/qui_tlemmas_rand_sequential/qui_tlemmas_rand_total_sequential/errors.json",  # noqa
]


BASE_OUT_PATH = "data/results/merged_all_tlemmas_sequential"


def copy_to_folder(file_path: str, target_path: str) -> None:
    file_direcory = os.path.dirname(target_path)
    os.makedirs(file_direcory, exist_ok=True)
    shutil.copy(file_path, target_path)


def merge_tlemmas() -> int:
    smt_file_counter = 0
    for starting_dir in DATA_FOLDERS:
        print("Merging from:", starting_dir)
        for root, _, files in os.walk(starting_dir):
            for file in files:
                file_path = os.path.join(root, file)
                target_path = file_path.replace(starting_dir, BASE_OUT_PATH)
                copy_to_folder(file_path, target_path)

                if file.endswith(".smt2"):
                    smt_file_counter += 1
    return smt_file_counter


def merge_errors(out_error_file_path: str) -> int:
    merged_errors = {}

    for error_file in ERROR_FILES:
        with open(error_file, "r") as f:
            errors = json.load(f)
            merged_errors.update(errors)

    if len(merged_errors.keys()):
        with open(out_error_file_path, "w+") as f:
            json.dump(merged_errors, f, indent=4)

    return len(merged_errors.keys())


def main():
    merged_lemmas = merge_tlemmas()
    print("Merged lemmas:", merged_lemmas)

    error_path = os.path.join(BASE_OUT_PATH, "errors.json")
    merged_errors = merge_errors(error_path)

    print("Merged errors:", merged_errors)
    print("Moved files:", merged_lemmas + merged_errors)


if __name__ == "__main__":
    main()
