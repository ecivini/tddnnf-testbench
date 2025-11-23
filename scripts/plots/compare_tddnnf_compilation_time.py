import json
import os

import matplotlib.pyplot as plt
import numpy as np

PREVIOUS_RESULTS_PATHS = [
    "data/michelutti_tdds/randgen/output_tddnnf_d4_total_new",
    "data/michelutti_tdds/ldd_randgen/output_tddnnf_d4_total_new",
]

CURRENT_RESULTS_PATHS = [
    "results/tddnnf_all_rand_parallel/data/michelutti_tdds/randgen/data",
    "results/tddnnf_all_rand_parallel/data/michelutti_tdds/ldd_randgen/data",
    # "data/results/tddnnf_all_rand_parallel/data/serialized_tdds/randgen",
    # "data/results/planning_h3_parallel/planning_h3_1Prob_45Procs/data/benchmark/planning/h3",
]

CURRENT_RESULTS_ERROR_FILE = "results/tddnnf_all_rand_parallel/errors.json"  # data/results/merged_all_tlemmas/errors.json"

DEFAULT_TIMEOUT = 2.0  # seconds
DDNNF_TIME_KEY = "dDNNF compilation time"
DDNNF_EDGES_KEY = "DD edges"
DDNNF_NODES_KEY = "DD nodes"

JSONS_TO_EXCLUDE = [
    "important_labels.json",
    "mapping.json"
]


def get_previous_results_times() -> tuple[dict, int, dict, dict]:
    times = {}
    edges = {}
    nodes = {}
    timeouts = 0
    for base_dir in PREVIOUS_RESULTS_PATHS:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if not file.endswith(".json") or file in JSONS_TO_EXCLUDE:
                    continue

                file_path = os.path.join(root, file)
                key_name = file.replace(".json", "")
                with open(file_path, "r") as f:
                    data = json.load(f)
                    if "timeout" in data:
                        timeouts += 1
                        times[key_name] = DEFAULT_TIMEOUT
                    elif DDNNF_TIME_KEY in data["T-dDNNF"]:
                        times[key_name] = data["T-dDNNF"][DDNNF_TIME_KEY]
                        edges[key_name] = data["T-dDNNF"][DDNNF_EDGES_KEY]
                        nodes[key_name] = data["T-dDNNF"][DDNNF_NODES_KEY]
                    else:
                        raise ValueError("Unhandled case for " + str(file_path))

    return times, timeouts, nodes, edges


def get_current_results_times(err_file: str, paths: list[str]) -> tuple[dict, int, dict, dict]:
    times = {}
    edges = {}
    nodes = {}

    missing_tlemmas = 0
    if err_file:
        with open(err_file, "r") as f:
            errors = json.load(f)
            for problem, reason in errors.items():
                if reason == "Missing tlemmas":
                    missing_tlemmas += 1
                    problem_name = problem.split(os.sep)[-1].replace(".smt2", "")
                    times[problem_name] = DEFAULT_TIMEOUT
                else:
                    print("Error reason:", reason)
                    raise ValueError("Unexpected error for:", problem)

    for base_dir in paths:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file != "logs.json":
                    continue

                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    data = json.load(f)

                problem_name = os.path.dirname(file_path).split(os.sep)[-1]
                times[problem_name] = data["T-DDNNF"][DDNNF_TIME_KEY]
                edges[problem_name] = data["T-DDNNF"][DDNNF_EDGES_KEY]
                nodes[problem_name] = data["T-DDNNF"][DDNNF_NODES_KEY]

    return times, missing_tlemmas, nodes, edges


def create_cactus_plot(previous: dict, current: dict):
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

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x1, previous_times, label="Sequential AllSMT", marker='o', markersize=2)
    plt.plot(x2, current_times, label="Parallel AllSMT", marker='^', markersize=2)

    plt.xlabel("Number of problems compiled")
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
    ax.axvline(timeout, linestyle='--', color='gray', label=f"parallel allsmt timeouts: {current_timeouts}")
    ax.axhline(timeout, linestyle='--', color='gray', label=f"sequential allsmt timeouts: {previous_timeouts}")

    # Set symlog scale
    # ax.set_xscale('symlog', linthresh=linthresh)
    # ax.set_yscale('symlog', linthresh=linthresh)
    ax.set_aspect('equal')

    # Set limits
    ax.set_xlim(left=1e-2, right=timeout * 1.1)
    ax.set_ylim(bottom=1e-2, top=timeout * 1.1)

    # Labels
    ax.set_xlabel('Parallelized AllSMT compilation time', fontsize=12)
    ax.set_ylabel('Sequential AllSMT compilation time', fontsize=12)

    # Grid
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    # Legend
    ax.legend()

    # Show plot
    plt.tight_layout()
    plt.savefig('tlemmas_scatter.png')


def create_counter_scatter_plot(
    previous: dict,
    current: dict,
    previous_label: str,
    current_label: str,
    plot_file_name: str
):
    previous_edges = []
    current_edges = []

    for problem in current.keys():
        if problem not in previous:
            continue

        previous_edges.append(previous[problem])
        current_edges.append(current[problem])

    timeout = max(max(previous_edges), max(current_edges))
    linthresh = 1  # Linear region until 1

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 7))

    # Scatter plot
    ax.scatter(x=current_edges, y=previous_edges,
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
    ax.set_xlabel(current_label, fontsize=12)
    ax.set_ylabel(previous_label, fontsize=12)

    # Grid
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    # Legend
    ax.legend()

    # Show plot
    plt.tight_layout()
    plt.savefig(plot_file_name)


if __name__ == "__main__":
    # previous_times, prev_timeouts, prev_nodes, prev_edges = get_previous_results_times()
    previous_times, prev_timeouts, prev_nodes, prev_edges = get_current_results_times(
        "results/tddnnf_all_rand_sequential/errors.json", 
        ["results/tddnnf_all_rand_sequential/data/michelutti_tdds/randgen/data",
         "results/tddnnf_all_rand_sequential/data/michelutti_tdds/ldd_randgen/data"]
    )

    current_times, curr_timeouts, curr_nodes, curr_edges = get_current_results_times(
        CURRENT_RESULTS_ERROR_FILE, CURRENT_RESULTS_PATHS)

    print("Previous timeouts", prev_timeouts)
    print("Current timeouts", curr_timeouts)

    prev_keys = set(previous_times.keys())
    curr_keys = set(current_times.keys())
    print("Missing keys:", prev_keys - curr_keys)

    # create_bar_plot(previous_times, current_times)
    create_scatter_plot(previous_times, current_times)
    create_cactus_plot(previous_times, current_times)

    create_counter_scatter_plot(
        prev_edges, curr_edges,
        "Sequential AllSMT - d-DNNF edges",
        "Parallel AllSMT - d-DNNF edges",
        "edges_scatter.png"
    )

    create_counter_scatter_plot(
        prev_nodes, curr_nodes,
        "Sequential AllSMT - d-DNNF nodes",
        "Parallel AllSMT - d-DNNF nodes",
        "nodes_scatter.png"
    )
