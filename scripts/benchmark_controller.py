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
TASK_TLEMMAS_CHECK = "tlemmas_check"
TASK_COMPILE = "compile"
TASK_TDDNNF = "tddnnfonly"
TASK_QUERY = "query"
TASK_TBDD = "tbdd"
TASK_TSDD = "tsdd"
TASK_QUERY_MC = "query_mc"
TASK_QUERY_CE = "query_ce"
ALLOWED_TASKS = [
    TASK_TLEMMAS,
    TASK_TLEMMAS_WITH_PROJECTION,
    TASK_COMPILE,
    TASK_QUERY,
    TASK_TDDNNF,
    TASK_TBDD,
    TASK_TSDD,
    TASK_QUERY_MC,
    TASK_QUERY_CE,
    TASK_TLEMMAS_CHECK,
]
TLEMMAS_RELATED_TASKS = [TASK_TLEMMAS, TASK_TLEMMAS_WITH_PROJECTION, TASK_TLEMMAS_CHECK]

TBDDS_RESULTS_BASE_PATHS = [
    "data/results/tbdd_par_proj/tbdd_parallel_proj/data/serialized_tdds/ldd_randgen/data",
    "data/results/tbdd_par_proj/tbdd_parallel_proj/data/serialized_tdds/randgen/data",
]

TSDDS_RESULTS_BASE_PATHS = [
    "data/results/tsdd_proj/tsdd_parallel_proj/data/serialized_tdds/ldd_randgen/data",
    "data/results/tsdd_proj/tsdd_parallel_proj/data/serialized_tdds/randgen/data",
]

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


def get_computed_dds(path: str) -> dict[str, str]:
    dds = {}
    for root, _, files in os.walk(path):
        for file_path in files:
            file_path = os.path.join(root, file_path)

            if file_path.endswith("abstraction.json"):
                key = (
                    root.replace(path, "")
                    .replace("data/benchmark/", "")
                    .replace("/randgen", "")
                    .replace("/ldd_randgen", "")
                    .replace("_tsdd", "")
                    .replace("_tbdd", "")
                )
                dds[key] = os.path.dirname(file_path)
            # else:
            #     print("[-] Skipping dd file:", file_path)
    return dds


def get_computed_nnfs(path: str) -> dict[str, str]:
    nnfs = {}
    for root, _, files in os.walk(path):
        for file_path in files:
            file_path = os.path.join(root, file_path)

            if file_path.endswith("compilation_output.nnf"):
                key = (
                    root.replace(path, "")
                    .replace("data/benchmark/", "")
                    .replace("/randgen", "")
                    .replace("/ldd_randgen", "")
                )
                nnfs[key] = file_path
    return nnfs


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
        print(f"[+] Compiling formula {data['formula_path']}...")
        command = (
            f"python3 scripts/tasks/compile_task.py {data['formula_path']} "
            f"{data['base_output_path']} {data['allsmt_processes']} {data['generate_tlemmas_only']} "
            f"{data['solver']} {data['project_atoms']}"
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
        print(f"[+] Compiling formula {data['formula_path']}...")
        command = (
            f"python3 scripts/tasks/tddnnf_task.py {data['formula_path']} "
            f"{data['base_output_path']} {data['tlemmas_path']}"
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


def dd_task(data: dict) -> tuple:
    """
    Compiles a given SMT formula using the tbdd_task script.
    Returns a tuple (succeeded: bool, test_case: str, error_message: str)
    """
    if data["tlemmas_path"] is None:
        return False, data["formula_path"], "Missing tlemmas"

    if "task" not in data:
        return False, data["formula_path"], "Missing task"
    selected_task = data["task"]
    if selected_task == TASK_TBDD:
        task_file = "tbdd_task.py"
        dd_type = "T-BDD"
    elif selected_task == TASK_TSDD:
        task_file = "tsdd_task.py"
        dd_type = "T-SDD"
    else:
        raise ValueError("Cannot handle task " + selected_task + " as DD.")

    compilation_succeeded = True
    error_message = ""
    try:
        print(f"[+] Compiling {dd_type} of {data['formula_path']}...")
        command = (
            f"python3 scripts/tasks/{task_file} {data['formula_path']} "
            f"{data['base_output_path']} {data['tlemmas_path']}"
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


def query_mc_task(data: dict) -> tuple:
    """
    Run MC queries on all inputs for plain SMT, d-DNNF, BDD and SDD.
    Returns a tuple (succeeded: bool, test_case: str, error_message: str)
    """
    if data["tlemmas_path"] is None:
        return False, data["formula_path"], "Missing tlemmas"

    if data["bdd_path"] is None:
        data["bdd_path"] = "none"

    if data["sdd_path"] is None:
        data["sdd_path"] = "none"

    if data["nnf_path"] is None:
        data["nnf_path"] = "none"

    compilation_succeeded = True
    error_message = ""
    try:
        print(f"[+] [MC] Querying formula {data['formula_path']}...")
        command = (
            f"python3 scripts/tasks/query_mc_task.py {data['formula_path']} "
            f"{data['base_output_path']} {data['nnf_path']} {data['tlemmas_path']} "
            f"{data['bdd_path']} {data['sdd_path']}"
        )
        command = command.split(" ")
        return_code, error = run_with_timeout_and_kill_children(
            command, data["timeout"], data["memory_limit"]
        )
        if return_code != 0:
            print(f"[-] Error during query of {data['formula_path']}:", error)
            compilation_succeeded = False
            error_message = error
        else:
            print(f"[+] Successfully queried {data['formula_path']}")
    except subprocess.TimeoutExpired:
        print(f"\t[-] Timeout during query of {data['formula_path']}")
        compilation_succeeded = False
        error_message = "timeout"
    except Exception as e:
        print(f"[-] Exception during query of {data['formula_path']}: {e}")
        compilation_succeeded = False
        error_message = str(e)

    return compilation_succeeded, data["formula_path"], error_message


def query_ce_task(data: dict) -> tuple:
    """
    Run CE queries on all inputs for plain SMT, d-DNNF, BDD and SDD.
    Returns a tuple (succeeded: bool, test_case: str, error_message: str)
    """
    # if data["tlemmas_path"] is None:
    #     return False, data["formula_path"], "Missing tlemmas"

    if data["bdd_path"] is None:
        data["bdd_path"] = "none"

    if data["sdd_path"] is None:
        data["sdd_path"] = "none"

    if data["nnf_path"] is None:
        return False, data["formula_path"], "Missing NNF"

    compilation_succeeded = True
    error_message = ""
    try:
        print(f"[+] [CE] Querying formula {data['formula_path']}...")
        command = (
            f"python3 scripts/tasks/query_ce_task.py {data['formula_path']} "
            f"{data['base_output_path']} {data['nnf_path']} {data['sdd_path']} "
            f"{data['bdd_path']}"
        )
        command = command.split(" ")
        return_code, error = run_with_timeout_and_kill_children(
            command, data["timeout"], data["memory_limit"]
        )
        if return_code != 0:
            print(f"[-] Error during query of {data['formula_path']}:", error)
            compilation_succeeded = False
            error_message = error
        else:
            print(f"[+] Successfully queried {data['formula_path']}")
    except subprocess.TimeoutExpired:
        print(f"\t[-] Timeout during query of {data['formula_path']}")
        compilation_succeeded = False
        error_message = "timeout"
    except Exception as e:
        print(f"[-] Exception during query of {data['formula_path']}: {e}")
        compilation_succeeded = False
        error_message = str(e)

    return compilation_succeeded, data["formula_path"], error_message


def tlemmas_check_task(data: dict) -> tuple:
    """
    Checks the correctness of the generated tlemmas for a given set of formulas.
    Returns a tuple (succeeded: bool, test_case: str, error_message: str)
    """
    test_succeeded = True
    error_message = ""
    try:
        print(f"[+] Testing t-lemmas for formula {data['formula_path']}...")
        solver = "parallel" if data["solver"] == "partition" else solver
        partition = "true" if data["solver"] == "partition" else "false"
        command = (
            f"python3 scripts/tasks/tlemmas_check.py {data['formula_path']} "
            f"{data['base_output_path']} {solver} {data['project_atoms']} "
            f"{partition}"
        )
        command = command.split(" ")
        return_code, error = run_with_timeout_and_kill_children(
            command, data["timeout"], data["memory_limit"]
        )
        if return_code != 0:
            print(f"[-] Error during tlemmas check of {data['formula_path']}:", error)
            test_succeeded = False
            error_message = error
        else:
            print(f"[+] Successfully checked {data['formula_path']}")
    except subprocess.TimeoutExpired:
        print(f"\t[-] Timeout during tlemmas check of {data['formula_path']}")
        test_succeeded = False
        error_message = "timeout"
    except Exception as e:
        print(f"[-] Exception during tlemmas check of {data['formula_path']}: {e}")
        test_succeeded = False
        error_message = str(e)

    return test_succeeded, data["formula_path"], error_message


def find_associated(
    benchmark: str, map: dict[str, str], for_nnf: bool = False
) -> str | None:
    if for_nnf:
        for key in map.keys():
            pieces = key.split("/")
            # sanity
            if "ldd_" in benchmark and "ldd_" not in key:
                continue
            if pieces[-1] in benchmark:
                return map[key]
    else:
        for key in map.keys():
            if key in benchmark:
                return map[key]
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
    if len(sys.argv) >= 4:
        name = sys.argv[3].lower().strip()
        if name == "sequential":
            solver = "sequential"
        elif name == "partition":
            solver = "partition"
        elif name == "parallel":
            solver = "parallel"
        else:
            raise ValueError("Unknown solver type")

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
    computed_bdds = {}
    computed_sdds = {}
    computed_nnfs = {}
    if selected_task in [
        TASK_TDDNNF,
        TASK_TBDD,
        TASK_TSDD,
        TASK_QUERY_MC,
        TASK_QUERY_CE,
    ]:
        tlemmas_base_path = config["tlemmas_dir"]
        computed_tlemmas = get_computed_tlemmas(tlemmas_base_path)

        if "tddnnf_dir" not in config:
            raise RuntimeError("Missing tddnnf_dir in config.yaml")

        tddnnf_base_path = config["tddnnf_dir"]
        computed_nnfs = get_computed_nnfs(tddnnf_base_path)

        for base_path in TBDDS_RESULTS_BASE_PATHS:
            computed_bdds.update(get_computed_dds(base_path))

        for base_path in TSDDS_RESULTS_BASE_PATHS:
            computed_sdds.update(get_computed_dds(base_path))

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
            "tlemmas_path": find_associated(test_case, computed_tlemmas),
            "project_atoms": selected_task == TASK_TLEMMAS_WITH_PROJECTION,
            "solver": solver,
            "nnf_path": find_associated(test_case, computed_nnfs, for_nnf=True),
            "bdd_path": find_associated(test_case, computed_bdds, for_nnf=True),
            "sdd_path": find_associated(test_case, computed_sdds, for_nnf=True),
            "task": selected_task,
        }
        datas.append(data)

        # nnf_path = find_associated(test_case, computed_nnfs, for_nnf=True)
        # print("PHI:", test_case)
        # print("NNF:", nnf_path)

    task_fn = None
    if selected_task in [TASK_COMPILE, TASK_TLEMMAS, TASK_TLEMMAS_WITH_PROJECTION]:
        task_fn = compile_task
    elif selected_task == TASK_TDDNNF:
        task_fn = tddnnf_task
    elif selected_task in [TASK_TBDD, TASK_TSDD]:
        task_fn = dd_task
    elif selected_task == TASK_QUERY_MC:
        task_fn = query_mc_task
    elif selected_task == TASK_QUERY_CE:
        task_fn = query_ce_task
    elif selected_task == TASK_TLEMMAS_CHECK:
        task_fn = tlemmas_check_task
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
