import os 
import json
import shutil
from collections import defaultdict

from typing import List, Dict

DATA_ROOT_PATHS = [
    "data/michelutti_tdds/ldd_randgen",
    "data/michelutti_tdds/randgen"
]
INVESTIGATION_OUTPUT_PATH = "data/investigation.json"
TIMEOUT_KEY = "timeout"

def scan_benchmark() -> None:
    counts: Dict = defaultdict(int)

    for starting_dir in DATA_ROOT_PATHS:
        for root, _, files in os.walk(starting_dir):
            for file in files:
                if not file.endswith(".json"):
                    continue

                json_path = os.path.join(root, file)
                with open(json_path, "r") as json_file:
                    json_data = json.load(json_file)

                if TIMEOUT_KEY in json_data:    
                    value = json_data[TIMEOUT_KEY]
                    counts[value] += 1

    with open(INVESTIGATION_OUTPUT_PATH, "w+") as investigation_file:
        json.dump(counts, investigation_file)

if __name__ == "__main__":
    scan_benchmark()