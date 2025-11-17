import json
import os

import matplotlib.pyplot as plt
import numpy as np

PREVIOUS_RESULTS_PATHS = [
    "data/michelutti_tdds/randgen/output_tddnnf_d4_total_new",
    "data/michelutti_tdds/ldd_randgen/output_tddnnf_d4_total_new",
]

CURRENT_RESULTS_PATHS = [
    # "data/results/merged_all_tlemmas/ldd_randgen",
    # "data/results/merged_all_tlemmas/randgen",
    "data/results/planning_h3_parallel/planning_h3_1Prob_45Procs/data/benchmark/planning/h3",
]

CURRENT_RESULTS_ERROR_FILE = "data/results/planning_h3_parallel/planning_h3_1Prob_45Procs/errors.json"  # "data/results/merged_all_tlemmas/errors.json"

DEFAULT_TIMEOUT = 3600.0  # seconds
PREVIOUS_RESULTS_TIME_KEY = "All-SMT computation time"


def get_previous_results_times() -> dict:
    times = {}
    for base_dir in PREVIOUS_RESULTS_PATHS:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if not file.endswith(".json") or file == "abstraction.json":
                    continue

                file_path = os.path.join(root, file)
                key_name = file.replace(".json", "")
                with open(file_path, "r") as f:
                    data = json.load(f)
                    if "timeout" in data and data["timeout"] == "ALL SMT":
                        times[key_name] = DEFAULT_TIMEOUT
                    elif PREVIOUS_RESULTS_TIME_KEY in data:
                        times[key_name] = data[PREVIOUS_RESULTS_TIME_KEY]
                    else:
                        times[key_name] = DEFAULT_TIMEOUT

    return times


def get_current_results_times(err_file: str, paths: list[str]) -> dict:
    times = {}

    with open(err_file, "r") as f:
        errors = json.load(f)
        for problem, reason in errors.items():
            if reason == "timeout":
                key_name = problem.split(os.sep)[-1].replace(".smt2", "")
                times[key_name] = DEFAULT_TIMEOUT
            else:
                print("Error reason:", reason)

    for base_dir in paths:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file != "logs.json":
                    continue

                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    data = json.load(f)

                problem_name = os.path.dirname(file_path).split(os.sep)[-1]
                times[problem_name] = data["T-DDNNF"][PREVIOUS_RESULTS_TIME_KEY]

    return times


def create_bar_plot(previous: dict, current: dict):
    problems = list(current.keys())
    previous_times = [previous[problem] for problem in problems]
    current_times = [current[problem] for problem in problems]

    x = range(len(problems))

    plt.figure(figsize=(36, 6))
    plt.bar(x, previous_times, width=0.4, label='Sequential AllSMT', align='center')
    plt.bar([i + 0.4 for i in x], current_times, width=0.4, label='Parallel AllSMT', align='center')

    plt.xlabel('Problems')
    plt.ylabel('Generation Time (seconds)')
    plt.title('Comparison of T-Lemmas Generation Time')
    plt.xticks([i + 0.2 for i in x], problems, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig('tlemmas_bar.png')


def create_cactus_plot(previous: dict, current: dict):
    cutoff = 2 * DEFAULT_TIMEOUT

    previous_times = []
    current_times = []
    vbs_times = []
    for problem in current:
        prev_time = previous[problem] if previous[problem] <= DEFAULT_TIMEOUT else DEFAULT_TIMEOUT
        current_time = current[problem] if current[problem] <= DEFAULT_TIMEOUT else DEFAULT_TIMEOUT
        vbs_time = min(prev_time, current_time)

        previous_times.append(prev_time)
        current_times.append(current_time)
        vbs_times.append(vbs_time)

    previous_times.sort()
    current_times.sort()
    vbs_times.sort()

    x1 = np.arange(1, len(previous_times) + 1)
    x2 = np.arange(1, len(current_times) + 1)
    x3 = np.arange(1, len(vbs_times) + 1)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x1, previous_times, label="Sequential AllSMT", marker='o', markersize=2)
    plt.plot(x2, current_times, label="Parallel AllSMT", marker='^', markersize=2)
    # plt.plot(x3, vbs_times, label="Virtual Best", marker='s', markersize=1)

    plt.xlabel("Number of problems solved")
    plt.ylabel("Time (s)")
    plt.title("Solvers comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("tlemmas_cactus.png")


def create_scatter_plot(previous: dict, current: dict):
    previous_times = []
    current_times = []
    previous_timeouts = 0
    current_timeouts = 0

    for problem in current.keys():
        previous_times.append(previous[problem])
        current_times.append(current[problem])

        if previous[problem] >= DEFAULT_TIMEOUT:
            previous_timeouts += 1

        if current[problem] >= DEFAULT_TIMEOUT:
            current_timeouts += 1

    timeout = DEFAULT_TIMEOUT
    linthresh = 1  # Linear region until 1

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 7))

    # Scatter plot
    ax.scatter(x=current_times, y=previous_times,
               color='darkgreen', edgecolors='black',
               s=25, zorder=4, alpha=0.2)

    # Reference line y = x
    ax.plot([1e-2, timeout], [1e-2, timeout], 'k--', label="y = x", zorder=2)

    # Timeout lines (dashed)
    ax.axvline(timeout, linestyle='--', color='gray', label=f"parallel timeouts: {current_timeouts}")
    ax.axhline(timeout, linestyle='--', color='gray', label=f"sequential timeouts: {previous_timeouts}")

    # Set symlog scale
    ax.set_xscale('symlog', linthresh=linthresh)
    ax.set_yscale('symlog', linthresh=linthresh)
    ax.set_aspect('equal')

    # Set limits
    ax.set_xlim(left=1e-2, right=timeout * 1.1)
    ax.set_ylim(bottom=1e-2, top=timeout * 1.1)

    # Labels
    ax.set_xlabel('Parallelized AllSMT', fontsize=12)
    ax.set_ylabel('Sequential AllSMT', fontsize=12)

    # Grid
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    # Legend
    ax.legend()

    # Show plot
    plt.tight_layout()
    plt.savefig('tlemmas_scatter.png')


if __name__ == "__main__":
    # previous_times = get_previous_results_times()
    previous_times = get_current_results_times(
        "data/results/planning_h3_sequential/planning_h3_1Proc_SequentialAllSmt/errors.json", 
        ["data/results/planning_h3_sequential/planning_h3_1Proc_SequentialAllSmt/data/benchmark/planning/h3"]
    )

    current_times = get_current_results_times(CURRENT_RESULTS_ERROR_FILE, CURRENT_RESULTS_PATHS)

    # create_bar_plot(previous_times, current_times)
    create_scatter_plot(previous_times, current_times)
    create_cactus_plot(previous_times, current_times)
