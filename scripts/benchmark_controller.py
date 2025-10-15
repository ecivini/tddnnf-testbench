from typing import Generator
import yaml
import os
from multiprocessing import Pool
import random
import time
import subprocess
import resource

###############################################################################

CONFIG_FILE = "config.yaml"

###############################################################################

"""
Reads and loads the current configuration from the default configuration file.
"""
def get_config():
    config = None

    with open(CONFIG_FILE, "r") as config_file:
        config = yaml.safe_load(config_file)
        
    return config

###############################################################################

def set_memory_limit(memory_limit: int):
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    memory_limit_bytes = memory_limit * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, hard))

def check_ext(file_path: str, ext: str = ".smt2") -> bool:
    _, fext = os.path.splitext(file_path)
    return fext == ext

def get_test_cases(paths: list[str]) -> Generator[str, None, None]:
    for path in paths:
        if os.path.isfile(path):
            if check_ext(path):
                yield path
            else:
                print("[-] Skipping test case: invalid file name", path)

        for root, _, files in os.walk(path):
            for file_path in files:
                test_case = os.path.join(root, file_path)

                if check_ext(test_case):
                    yield test_case
                else:
                    print("[-] Skipping test case: invalid file name", test_case)

def compile_task(data: dict) -> None:
    try:
        print(f"[+] Compiling formula {data["formula_path"]}...")
        command = f"python3 scripts/tasks/compile_task.py {data["formula_path"]} {data["base_output_path"]}".split(" ")
        _ = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=data["timeout"] + 5,
            preexec_fn=set_memory_limit(int(data["memory_limit"]))
        )
    except subprocess.TimeoutExpired:
        print(f"[-] Timeout expired during compilation of {data['formula_path']}")
    except Exception as e:
        print(f"[-] Exception during compilation of {data['formula_path']}: {e}")

def main():
    config = get_config()

    # Create output file
    base_path = config["results"]
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    timestamp = str(int(time.time()))
    output_base_path = os.path.join(base_path, timestamp)

    processes = int(config["processes"])

    # Run tests
    datas = []
    for test_case in get_test_cases(config["benchmarks"]):
        output_path = os.path.join(output_base_path, test_case)[:-5] # remove .smt2
        data = {
            "timeout": int(config["timeout"]),
            "memory_limit": config["memory"],
            "formula_path": test_case,
            "base_output_path": output_path
        }
        datas.append(data)

    random.shuffle(datas)
    with Pool(processes=processes) as pool:
        for _ in pool.imap_unordered(compile_task, datas):
            continue
    
###############################################################################

if __name__ == "__main__":
    main()
