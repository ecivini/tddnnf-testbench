from typing import Generator, Tuple
import yaml
import os
import sys
from multiprocessing import Pool
import random
import time
import subprocess
import resource
import json
import psutil

###############################################################################

CONFIG_FILE = "config.yaml"

TASK_TLEMMAS = "tlemmas"
TASK_TLEMMAS_WITH_PROJECTION = "tlemmas_proj"
TASK_COMPILE = "compile"
TASK_TDDNNF = "tddnnfonly"
TASK_QUERY = "query"
ALLOWED_TASKS = [
    TASK_TLEMMAS,
    TASK_TLEMMAS_WITH_PROJECTION,
    TASK_COMPILE,
    TASK_QUERY,
    TASK_TDDNNF,
]
TLEMMAS_RELATED_TASKS = [TASK_TLEMMAS, TASK_TLEMMAS_WITH_PROJECTION]

###############################################################################


def get_config():
    """
    Reads and loads the current configuration from the default
    configuration file.
    """
    config = None

    with open(CONFIG_FILE, "r") as config_file:
        config = yaml.safe_load(config_file)

    return config


def set_memory_limit(memory_limit: int):
    """
    Sets the memory limit for a subprocess.
    The memory limit is given in MB.
    """
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    memory_limit_bytes = memory_limit * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, hard))


def check_ext(file_path: str, ext: str = ".smt2") -> bool:
    """
    Checks if the given file has the specified extension.
    """
    _, fext = os.path.splitext(file_path)
    return fext == ext


def computed_selector_for_compilation(files: list[str], task: str) -> bool:
    """
    Selector function to determine if a benchmark has already been computed
    for compilation task. A benchmark is considered computed if there is a
    file named "logs.json" in its result directory.
    """
    if task not in [TASK_COMPILE, TASK_TLEMMAS]:
        return False

    return "logs.json" in files


###############################################################################


def get_test_cases(
    paths: list[str], already_computed: list[str]
) -> Generator[str, None, None]:
    """
    Yields all test cases found in the given paths that have not been
    already computed.
    """
    for path in paths:
        if os.path.isfile(path):
            name = path.split(os.sep)[-1]
            if check_ext(path) and name not in already_computed:
                yield path
            else:
                print("[-] Skipping test case:", path)

        for root, _, files in os.walk(path):
            for file_path in files:
                test_case = os.path.join(root, file_path)

                if check_ext(test_case) and file_path not in already_computed:
                    yield test_case
                else:
                    print("[-] Skipping test case:", test_case)


def get_computed_tlemmas(path: str) -> dict[str, str]:
    tlemmas = {}
    for root, _, files in os.walk(path):
        for file_path in files:
            tlemma = os.path.join(root, file_path)

            if check_ext(tlemma):
                key = (
                    root.replace(path, "")
                    .replace("data/benchmark/", "")
                    .replace("/randgen", "")
                    .replace("/ldd_randgen", "")
                )
                tlemmas[key] = tlemma
            else:
                print("[-] Skipping tlemma:", tlemma)
    return tlemmas


def get_already_computed_benchmarks(base_path: str, task: str) -> list[str]:
    # TODO: Adapt this function to different tasks
    """
    Returns a list of already computed benchmarks in the given base path.
    """
    computed = []
    for root, _, files in os.walk(base_path):
        if computed_selector_for_compilation(files, task):
            # Remove the result path and add .smt2 extension
            problem_path = root.split(os.sep)[-1] + ".smt2"
            computed.append(problem_path)

    print(f"[+] Found {len(computed)} already computed benchmarks")

    return computed


def run_with_timeout_and_kill_children(
    command: list[str], timeout: float, memory_limit: int
) -> Tuple[int, str]:
    proc = subprocess.Popen(
        command,
        preexec_fn=set_memory_limit(memory_limit),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        proc.wait(timeout=timeout + 2)
        return proc.returncode, proc.stderr.read().decode("utf-8")
    except subprocess.TimeoutExpired as e:
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                continue
        try:
            parent.kill()
            parent.wait()
        except psutil.NoSuchProcess:
            pass

        raise e


def compile_task(data: dict) -> tuple:
    """
    Compiles a given SMT formula using the compile_task script.
    Returns a tuple (succeeded: bool, test_case: str, error_message: str)
    """
    compilation_succeeded = True
    error_message = ""
    try:
        print(f"[+] Compiling formula {data["formula_path"]}...")
        command = (
            f"python3 scripts/tasks/compile_task.py {data["formula_path"]} "
            f"{data["base_output_path"]} {data["allsmt_processes"]} {data["generate_tlemmas_only"]} "
            f"{data["solver"]} {data["project_atoms"]}"
        )
        command = command.split(" ")
        return_code, error = run_with_timeout_and_kill_children(
            command, data["timeout"], data["memory_limit"]
        )
        if return_code != 0:
            print(f"[-] Error during compilation of {data['formula_path']}:", error)
            compilation_succeeded = False
            error_message = error
        else:
            print(f"[+] Successfully compiled {data['formula_path']}")
    except subprocess.TimeoutExpired:
        print(f"\t[-] Timeout during compilation of {data['formula_path']}")
        compilation_succeeded = False
        error_message = "timeout"
    except Exception as e:
        print(f"[-] Exception during compilation of {data['formula_path']}: {e}")
        compilation_succeeded = False
        error_message = str(e)

    return compilation_succeeded, data["formula_path"], error_message


def tddnnf_task(data: dict) -> tuple:
    """
    Compiles a given SMT formula using the tddnf_task script.
    Returns a tuple (succeeded: bool, test_case: str, error_message: str)
    """
    if data["tlemmas_path"] is None:
        return False, data["formula_path"], "Missing tlemmas"

    compilation_succeeded = True
    error_message = ""
    try:
        print(f"[+] Compiling formula {data["formula_path"]}...")
        command = (
            f"python3 scripts/tasks/tddnnf_task.py {data["formula_path"]} "
            f"{data["base_output_path"]} {data["tlemmas_path"]}"
        )
        command = command.split(" ")
        return_code, error = run_with_timeout_and_kill_children(
            command, data["timeout"], data["memory_limit"]
        )
        if return_code != 0:
            print(f"[-] Error during compilation of {data['formula_path']}:", error)
            compilation_succeeded = False
            error_message = error
        else:
            print(f"[+] Successfully compiled {data['formula_path']}")
    except subprocess.TimeoutExpired:
        print(f"\t[-] Timeout during compilation of {data['formula_path']}")
        compilation_succeeded = False
        error_message = "timeout"
    except Exception as e:
        print(f"[-] Exception during compilation of {data['formula_path']}: {e}")
        compilation_succeeded = False
        error_message = str(e)

    return compilation_succeeded, data["formula_path"], error_message


def find_associated_tlemmas_from_phi(
    benchmark: str, tlemmas: dict[str, str]
) -> str | None:
    for key in tlemmas.keys():
        if key in benchmark:
            return tlemmas[key]
    return None


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ALLOWED_TASKS:
        print(
            "Usage: python3 scripts/benchmark_controller.py "
            "<tlemmas|compile|query> <test_name> <optional:sequential|parallel>"
        )
        sys.exit(1)
    selected_task = sys.argv[1]
    test_name = sys.argv[2]

    config = get_config()

    # Get solver type
    solver = "parallel"  # parallel by default
    if len(sys.argv) >= 4 and sys.argv[3].lower().strip() == "sequential":
        solver = "sequential"

    # Create output file
    base_path = config["results"]
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    output_base_path = os.path.join(base_path, test_name)
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    already_computed = get_already_computed_benchmarks(output_base_path, selected_task)

    processes = int(config["processes"])
    allsmt_processes = int(config["allsmt_processes"])

    computed_tlemmas = {}
    if selected_task == TASK_TDDNNF:
        tlemmas_base_path = config["tlemmas_dir"]
        computed_tlemmas = get_computed_tlemmas(tlemmas_base_path)

    # Run tests
    datas = []
    for test_case in get_test_cases(config["benchmarks"], already_computed):
        # remove .smt2
        output_path = os.path.join(output_base_path, test_case)[:-5]
        data = {
            "timeout": int(config["timeout"]),
            "memory_limit": config["memory"],
            "formula_path": test_case,
            "base_output_path": output_path,
            "allsmt_processes": allsmt_processes,
            "generate_tlemmas_only": selected_task in TLEMMAS_RELATED_TASKS,
            "tlemmas_path": find_associated_tlemmas_from_phi(
                test_case, computed_tlemmas
            ),
            "project_atoms": selected_task == TASK_TLEMMAS_WITH_PROJECTION,
            "solver": solver,
        }
        datas.append(data)

    task_fn = None
    if selected_task in [TASK_COMPILE, TASK_TLEMMAS, TASK_TLEMMAS_WITH_PROJECTION]:
        task_fn = compile_task
    elif selected_task == TASK_TDDNNF:
        task_fn = tddnnf_task
    elif selected_task == TASK_QUERY:
        print("[-] Query task not implemented yet")
        sys.exit(1)

    random.shuffle(datas)

    start_ts = time.time()
    errored_cases = {}
    with Pool(processes=processes) as pool:
        for succeeded, test_case, error in pool.imap_unordered(task_fn, datas):
            if not succeeded:
                errored_cases[test_case] = error

    if errored_cases:
        errors_file_path = os.path.join(output_base_path, "errors.json")
        with open(errors_file_path, "w+") as errors_file:
            json.dump(errored_cases, errors_file, indent=4)

    total_time = time.time() - start_ts
    print(f"[+] Benchmark completed in {total_time:.2f} seconds")


###############################################################################


if __name__ == "__main__":
    main()
