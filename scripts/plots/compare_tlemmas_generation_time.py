import json
import os

import matplotlib.pyplot as plt

PREVIOUS_RESULTS_PATHS = [
    "data/michelutti_tdds/randgen/output_tbdd_total_new",
    "data/michelutti_tdds/ldd_randgen/output_tbdd_total_new"
]

CURRENT_RESULTS_PATHS = [
    "data/results/merged_tlemmas/ldd_randgen",
    "data/results/merged_tlemmas/randgen"
]

CURRENT_RESULTS_ERROR_FILE = "data/results/merged_tlemmas/errors.json"

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


def get_current_results_times() -> dict:
    times = {}

    with open(CURRENT_RESULTS_ERROR_FILE, "r") as f:
        errors = json.load(f)
        for problem, reason in errors.items():
            if reason == "timeout":
                key_name = problem.split(os.sep)[-1].replace(".smt2", "")
                times[key_name] = DEFAULT_TIMEOUT

    for base_dir in CURRENT_RESULTS_PATHS:
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

    plt.figure(figsize=(18, 6))
    plt.bar(x, previous_times, width=0.4, label='Sequential AllSMT', align='center')
    plt.bar([i + 0.4 for i in x], current_times, width=0.4, label='Parallel AllSMT', align='center')

    plt.xlabel('Problems')
    plt.ylabel('Generation Time (seconds)')
    plt.title('Comparison of T-Lemmas Generation Time')
    plt.xticks([i + 0.2 for i in x], problems, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig('tlemmas_bar.png')


def create_scatter_plot(previous: dict, current: dict):
    previous_times = []
    current_times = []
    for problem in current.keys():
        previous_times.append(previous[problem])
        current_times.append(current[problem])
    timeout = DEFAULT_TIMEOUT
    # linthresh = 1  # Linear region until 1

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 7))

    # Scatter plot
    ax.scatter(current_times, previous_times,
               color='darkgreen', edgecolors='black',
               s=25, zorder=4, alpha=0.2)

    # Reference line y = x
    ax.plot([1e-2, timeout], [1e-2, timeout], 'k--', label="y = x", zorder=2)

    # Timeout lines (dashed)
    ax.axvline(timeout, linestyle='--', color='gray')
    ax.axhline(timeout, linestyle='--', color='gray')

    # Set symlog scale
    # ax.set_xscale('symlog', linthresh=linthresh)
    # ax.set_yscale('symlog', linthresh=linthresh)
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
    previous_times = get_previous_results_times()
    current_times = get_current_results_times()

    create_bar_plot(previous_times, current_times)
    create_scatter_plot(previous_times, current_times)
