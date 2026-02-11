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

JSONS_TO_EXCLUDE = ["important_labels.json", "mapping.json"]


def get_previous_results_times() -> tuple[dict, int, dict, dict, dict]:
    times = {}
    edges = {}
    nodes = {}
    partial_models = {}
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
                        # partial_models[key_name] = data["T-dDNNF"][PARTIAL_MODELS_KEY]
                    else:
                        raise ValueError("Unhandled case for " + str(file_path))

    return times, timeouts, nodes, edges, partial_models


def get_current_results_times(
    err_file: str, paths: list[str]
) -> tuple[dict, int, dict, dict, dict]:
    times = {}
    edges = {}
    nodes = {}
    partial_models = {}

    missing_tlemmas = 0
    if err_file:
        with open(err_file, "r") as f:
            errors = json.load(f)
            for problem, reason in errors.items():
                if reason in ["Missing tlemmas", "timeout"]:
                    missing_tlemmas += 1
                    # problem_name = problem.split(os.sep)[-1].replace(".smt2", "")
                    # times[problem_name] = DEFAULT_TIMEOUT
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

    return times, missing_tlemmas, nodes, edges, partial_models


def create_cactus_plot(
    previous: dict,
    current: dict,
    prev_label: str,
    curr_label: str,
    third: dict | None = None,
    third_label: str | None = None,
    out_path: str = "cactus.pdf",
):
    previous_times = []
    current_times = []
    third_times = []
    vbs_times = []
    for problem in current:
        if (
            problem not in previous
            or problem not in current
            or (third is not None and problem not in third)
        ):
            continue

        prev_time = previous[problem]
        current_time = current[problem]

        vbs_time = min(prev_time, current_time)
        if third is not None:
            third_time = third[problem]

            vbs_time = min(vbs_time, third_time)
            third_times.append(third_time)

        previous_times.append(prev_time)
        current_times.append(current_time)
        vbs_times.append(vbs_time)

    previous_times.sort()
    current_times.sort()
    third_times.sort()
    vbs_times.sort()

    x1 = np.arange(1, len(previous_times) + 1)
    x2 = np.arange(1, len(current_times) + 1)
    x3 = np.arange(1, len(third_times) + 1)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x1, previous_times, label=f"{prev_label}", marker="o", markersize=2)
    plt.plot(x2, current_times, label=f"{curr_label}", marker="^", markersize=2)

    if third is not None:
        plt.plot(x3, third_times, label=f"{third_label}", marker="+", markersize=2)

    plt.xlabel("Number of problems compiled", fontsize=24)
    plt.ylabel("Time (s)", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.title(f"{prev_label} vs {curr_label}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)


def create_scatter_plot(
    previous: dict,
    current: dict,
    prev_label: str,
    curr_label: str,
    out_path: str = "scatter.pdf",
):
    previous_times = []
    current_times = []
    previous_timeouts = 0
    current_timeouts = 0

    for problem in current.keys():
        if problem not in previous or problem not in current:
            continue

        previous_times.append(previous[problem])
        current_times.append(current[problem])

        if previous[problem] >= DEFAULT_TIMEOUT:
            previous_timeouts += 1

        if current[problem] >= DEFAULT_TIMEOUT:
            current_timeouts += 1

    timeout = max(max(previous_times), max(current_times)) * 1.2
    linthresh = 1  # Linear region until 1

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 5))

    # Scatter plot
    ax.scatter(
        x=current_times,
        y=previous_times,
        color="lightskyblue",
        edgecolors="black",
        s=100,
        zorder=4,
        alpha=1,
        marker="X",
    )

    # Reference line y = x
    ax.plot(
        [1e-2, timeout],
        [1e-2, timeout],
        label="y = x",
        zorder=2,
        color="gray",
        linestyle="--",
    )

    # Timeout lines (dashed)
    ax.axvline(
        timeout,
        linestyle="--",
        color="gray",
        # label=f"{curr_label} timeouts: {current_timeouts}",
    )
    ax.axhline(
        timeout,
        linestyle="--",
        color="gray",
        # label=f"{prev_label} timeouts: {previous_timeouts}",
    )
    print(out_path)
    print("curr timeouts:", current_timeouts)
    print("prev timeouts:", prev_timeouts)

    # Set symlog scale
    # ax.set_xscale('symlog', linthresh=linthresh)
    # ax.set_yscale('symlog', linthresh=linthresh)
    ax.set_aspect("equal")

    # Set limits
    ax.set_xlim(left=1e-2, right=timeout * 1.1)
    ax.set_ylim(bottom=1e-2, top=timeout * 1.1)

    # Labels
    ax.set_xlabel(f"{curr_label}", fontsize=24)
    ax.set_ylabel(f"{prev_label}", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Grid
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    # Legend
    # ax.legend()

    # Show plot
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)


def create_counter_scatter_plot(
    previous: dict,
    current: dict,
    previous_label: str,
    current_label: str,
    out_path: str,
):
    previous_edges = []
    current_edges = []

    for problem in current.keys():
        if problem not in previous:
            continue

        previous_edges.append(
            previous[problem]  # if previous[problem] <= timeout else timeout
        )
        current_edges.append(
            current[problem]  # if current[problem] <= timeout else timeout
        )

    timeout = max(max(previous_edges), max(current_edges))
    # linthresh = timeout / 16  # Linear region until 1

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 5))

    # Scatter plot
    ax.scatter(
        x=current_edges,
        y=previous_edges,
        color="lightskyblue",
        edgecolors="black",
        s=100,
        zorder=4,
        alpha=1,
        marker="X",
    )

    # Reference line y = x
    ax.plot(
        [1e-2, timeout],
        [1e-2, timeout],
        label="y = x",
        zorder=2,
        color="gray",
        linestyle="--",
    )

    # Timeout lines (dashed)
    # ax.axvline(timeout, linestyle="--", color="gray")
    # ax.axhline(timeout, linestyle="--", color="gray")

    # Set symlog scale
    # ax.set_xscale('symlog', lprev_solverinthresh=linthresh)
    # ax.set_yscale('symlog', linthresh=linthresh)
    # ax.set_xscale("linear")
    # ax.set_yscale("equal")
    ax.set_aspect("equal")

    # Set limits
    ax.set_xlim(left=1e-2, right=timeout * 1.1)
    ax.set_ylim(bottom=1e-2, top=timeout * 1.1)

    # Labelsprev_solver
    ax.set_xlabel(current_label, fontsize=24)
    ax.set_ylabel(previous_label, fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Grid
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    # Legend
    # ax.legend()

    # Show plot
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    # previous_times, prev_timeouts, prev_nodes, prev_edges, prev_par_models = (
    #     get_previous_results_times()
    # )
    # previous_times, prev_timeouts, prev_nodes, prev_edges, prev_par_models = (
    #     get_current_results_times(
    #         "results/tddnnf_compilation_rand_sequential_v2/errors.json",
    #         [
    #             "results/tddnnf_compilation_rand_sequential_v2/data/michelutti_tdds/ldd_randgen/data",  # noqa
    #             "results/tddnnf_compilation_rand_sequential_v2/data/michelutti_tdds/randgen/data",  # noqa
    #         ],
    #     )
    # )

    # current_times, curr_timeouts, curr_nodes, curr_edges, curr_par_models = (
    #     get_current_results_times(
    #         "results/tddnnf_compilation_rand_parallel/errors.json",
    #         [
    #             "results/tddnnf_compilation_rand_parallel/data/michelutti_tdds/ldd_randgen/data",  # noqa
    #             "results/tddnnf_compilation_rand_parallel/data/michelutti_tdds/randgen/data",  # noqa
    #         ],
    #     )
    # )

    # x3_times, x3_timeouts, x3_nodes, x3_edges, x3_par_models = (
    #     get_current_results_times(
    #         "results/tddnnf_compilation_rand_projected_atoms/errors.json",
    #         [
    #             "results/tddnnf_compilation_rand_projected_atoms/data/michelutti_tdds/ldd_randgen/data",  # noqa
    #             "results/tddnnf_compilation_rand_projected_atoms/data/michelutti_tdds/randgen/data",  # noqa
    #         ],
    #     )
    # )

    # PLANNING
    previous_times, prev_timeouts, prev_nodes, prev_edges, prev_par_models = (
        get_current_results_times(
            "results/tddnnf_compilation_planning_h3_sequential/errors.json",
            [
                "results/tddnnf_compilation_planning_h3_sequential/data/benchmark/planning/h3/Painter",  # noqa
            ],
        )
    )

    current_times, curr_timeouts, curr_nodes, curr_edges, curr_par_models = (
        get_current_results_times(
            "results/tddnnf_compilation_planning_h3_parallel/errors.json",
            [
                "results/tddnnf_compilation_planning_h3_parallel/data/benchmark/planning/h3/Painter",  # noqa
            ],
        )
    )

    x3_times, x3_timeouts, x3_nodes, x3_edges, x3_par_models = (
        get_current_results_times(
            "results/tddnnf_compilation_planning_h3_parallel_proj/errors.json",
            [
                "results/tddnnf_compilation_planning_h3_parallel_proj/data/benchmark/planning/h3/Painter",  # noqa
            ],
        )
    )

    print("Previous timeouts", prev_timeouts)
    print("Current timeouts", curr_timeouts)

    prev_keys = set(previous_times.keys())
    curr_keys = set(current_times.keys())
    print("Missing keys:", prev_keys - curr_keys)

    # create_bar_plot(previous_times, current_times)
    prev_solver = "Baseline"
    curr_solver = "D&C"
    x3_solver = "D&C+Proj"

    # Cactus - compilation times
    create_cactus_plot(
        previous_times,
        current_times,
        prev_solver,
        curr_solver,
        third=x3_times,
        third_label=x3_solver,
        out_path="cactus_seq_vs_par45_vs_par45_proj_atoms_comp_time.pdf",
    )

    # Scatter - Compilation time
    create_scatter_plot(
        current_times,
        previous_times,
        curr_solver,
        prev_solver,
        out_path="seq_vs_par45_comp_time.pdf",
    )

    create_scatter_plot(
        x3_times,
        current_times,
        x3_solver,
        curr_solver,
        out_path="par45_vs_par45_proj_atoms_comp_time.pdf",
    )

    create_scatter_plot(
        x3_times,
        previous_times,
        x3_solver,
        prev_solver,
        out_path="seq_vs_par45_proj_atoms_comp_time.pdf",
    )

    # Scatter - dDNNF edges
    create_counter_scatter_plot(
        curr_edges,
        prev_edges,
        curr_solver,
        prev_solver,
        "seq_vs_par45_ddnnf_edges.pdf",
    )

    create_counter_scatter_plot(
        x3_edges,
        prev_edges,
        x3_solver,
        prev_solver,
        "seq_vs_par45_proj_atoms_ddnnf_edges.pdf",
    )

    create_counter_scatter_plot(
        x3_edges,
        curr_edges,
        x3_solver,
        curr_solver,
        "par45_vs_par45_proj_atoms_ddnnf_edges.pdf",
    )

    # Scatter - dDNNF nodes
    create_counter_scatter_plot(
        curr_nodes,
        prev_nodes,
        curr_solver,
        prev_solver,
        "seq_vs_par45_ddnnf_nodes.pdf",
    )

    create_counter_scatter_plot(
        x3_nodes,
        prev_nodes,
        x3_solver,
        prev_solver,
        "seq_vs_par45_proj_atoms_ddnnf_nodes.pdf",
    )

    create_counter_scatter_plot(
        x3_nodes,
        curr_nodes,
        x3_solver,
        curr_solver,
        "par45_vs_par45_proj_atoms_ddnnf_nodes.pdf",
    )
