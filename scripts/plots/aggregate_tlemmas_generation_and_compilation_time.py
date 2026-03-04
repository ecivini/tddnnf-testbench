import json
import os

import matplotlib.pyplot as plt
import numpy as np

PREVIOUS_RESULTS_PATHS = [
    # "data/michelutti_tdds/randgen/output_tddnnf_d4_total_new",
    # "data/michelutti_tdds/ldd_randgen/output_tddnnf_d4_total_new",
    "data/michelutti_tdds/randgen/output_tbdd_total_new",
    "data/michelutti_tdds/ldd_randgen/output_tbdd_total_new",
]

# CURRENT_RESULTS_PATHS = [
#     "results/tddnnf_all_rand_parallel/data/michelutti_tdds/randgen/data",
#     "results/tddnnf_all_rand_parallel/data/michelutti_tdds/ldd_randgen/data",
#     # "data/results/tddnnf_all_rand_parallel/data/serialized_tdds/randgen",
#     # "data/results/planning_h3_parallel/planning_h3_1Prob_45Procs/data/benchmark/planning/h3",
# ]

CURRENT_RESULTS_ERROR_FILE = "results/tddnnf_all_rand_parallel/errors.json"  # data/results/merged_all_tlemmas/errors.json"

DEFAULT_TIMEOUT = 3600.0  # seconds
# DDNNF_TIME_KEY = "dDNNF compilation time"
DDNNF_TIME_KEY = "dDNNF compilation time"
CURRENT_RESULTS_TIME_KEY = "Total time"
PREVIOUS_RESULTS_TIME_KEY = "total computation time"
DDNNF_EDGES_KEY = "DD edges"
DDNNF_NODES_KEY = "DD nodes"

JSONS_TO_EXCLUDE = ["important_labels.json", "mapping.json", "abstraction.json"]


def get_previous_results_times() -> tuple[dict, int]:
    times = {}
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
                    elif PREVIOUS_RESULTS_TIME_KEY in data:
                        times[key_name] = data[PREVIOUS_RESULTS_TIME_KEY]
                    else:
                        raise ValueError("Unhandled case for " + str(file_path))

    return times, timeouts


def _get_current_results_times(
    err_file: str | None, paths: list[str], target_key: str
) -> tuple[dict, int]:
    times = {}

    missing_tlemmas = 0
    if err_file:
        with open(err_file, "r") as f:
            errors = json.load(f)
            for problem, reason in errors.items():
                if reason in ["Missing tlemmas", "timeout"]:
                    missing_tlemmas += 1
                    problem_name = problem.split(os.sep)[-1].replace(".smt2", "")
                    times[problem_name] = DEFAULT_TIMEOUT
                else:
                    print("Error reason:", reason)
                    raise ValueError("Unexpected error for:", problem)

    for base_dir in paths:
        print(base_dir)
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file != "logs.json":
                    continue

                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    data = json.load(f)

                problem_name = "".join(os.path.dirname(file_path).split(os.sep)[-4:])
                times[problem_name] = data["T-DDNNF"][target_key]

    return times, missing_tlemmas


def get_current_results_times(
    tddnnf_err_file: str | None, tlemmas_path: list[str], tddnnf_path: list[str]
):
    tlemmas_times, tlemmas_timeouts = _get_current_results_times(
        None, tlemmas_path, CURRENT_RESULTS_TIME_KEY
    )

    tddnnf_times, tddnnf_timeouts = _get_current_results_times(
        tddnnf_err_file, tddnnf_path, DDNNF_TIME_KEY
    )

    times = {}
    for problem in tddnnf_times:
        if problem not in tlemmas_times:
            times[problem] = DEFAULT_TIMEOUT
        else:
            times[problem] = tddnnf_times[problem] + tlemmas_times[problem]
            if times[problem] >= DEFAULT_TIMEOUT:
                times[problem] = DEFAULT_TIMEOUT

    print(len(tlemmas_times))

    return times, tlemmas_timeouts + tddnnf_timeouts


def create_cactus_plot(
    previous: dict,
    current: dict,
    prev_label: str,
    curr_label: str,
    third: dict | None = None,
    third_label: str = "",
    fourth: dict | None = None,
    fourth_label: str = "",
    out_path: str = "cactus.pdf",
):
    previous_times = []
    current_times = []
    third_times = []
    fourth_times = []
    # vbs_times = []
    # for problem in previous:
    #     prev_time = previous[problem]
    #     current_time = current[problem]
    #     vbs_time = min(prev_time, current_time)

    #     if third is not None:
    #         third_time = third[problem]
    #         vbs_time = min(vbs_time, third_time)
    #         third_times.append(third_time)

    #     previous_times.append(prev_time)
    #     current_times.append(current_time)
    #     vbs_times.append(vbs_time)

    #     if prev_time > DEFAULT_TIMEOUT:
    #         print(problem)

    max_num_of_probs = 0
    if fourth is not None:
        for problem in fourth:
            fourth_times.append(fourth[problem])
        max_num_of_probs = len(fourth_times)

    if third is not None:
        for problem in third:
            third_times.append(third[problem])
        if max_num_of_probs > 0:
            while len(third_times) < max_num_of_probs:
                third_times.append(DEFAULT_TIMEOUT)
        else:
            max_num_of_probs = len(third_times)

    for problem in current:
        current_times.append(current[problem])
    if max_num_of_probs > 0:
        while len(current_times) < max_num_of_probs:
            current_times.append(DEFAULT_TIMEOUT)
    else:
        max_num_of_probs = len(previous_times)

    for problem in previous:
        previous_times.append(previous[problem])
    while len(previous_times) < max_num_of_probs:
        previous_times.append(DEFAULT_TIMEOUT)

    previous_times.sort()
    current_times.sort()
    third_times.sort()
    fourth_times.sort()
    # vbs_times.sort()

    x1 = np.arange(1, len(previous_times) + 1)
    x2 = np.arange(1, len(current_times) + 1)
    x3 = np.arange(1, len(third_times) + 1)
    x4 = np.arange(1, len(fourth_times) + 1)

    # Plot
    plt.figure(figsize=(9, 6))
    plt.plot(x1, previous_times, label=f"{prev_label}", marker="o", markersize=2)
    plt.plot(x2, current_times, label=f"{curr_label}", marker="^", markersize=2)

    if third is not None:
        plt.plot(x3, third_times, label=f"{third_label}", marker="+", markersize=2)

    if fourth is not None:
        plt.plot(x4, fourth_times, label=f"{fourth_label}", marker="+", markersize=2)

    plt.xlabel("Number of problems transformed", fontsize=24)
    plt.ylabel("Time (s)", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.title(
    #     f"Solvers comparison: {prev_label} vs {curr_label}" + f" vs {third_label}"
    #     if third is not None
    #     else ""
    # )
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(out_path)


def create_scatter_plot(
    previous: dict,
    current: dict,
    prev_label: str,
    curr_label: str,
    out_path: str = "cactus.pdf",
):
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
    linthresh = 10  # Linear region until 1

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

    print(
        f"\n{out_path}\n"
        f"{prev_label} timeouts: {previous_timeouts}\n"
        f"{curr_label} timeouts: {current_timeouts}"
    )

    # Set symlog scale
    ax.set_xscale("symlog", linthresh=linthresh)
    ax.set_yscale("symlog", linthresh=linthresh)
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
    # ax.legend(loc="lower right")

    # Show plot
    plt.tight_layout()
    plt.savefig(out_path)


if __name__ == "__main__":
    # ###########################################################################
    # # RAND - SEQUENTIAL V2
    # tddnnf_err_file = "results/tddnnf_compilation_rand_seq/errors.json"
    # tlemmas_paths = [
    #     "data/results/merged_all_tlemmas_sequential/ldd_randgen/data",
    #     "data/results/merged_all_tlemmas_sequential/randgen/data",
    # ]

    # tddnnf_paths = [
    #     "results/tddnnf_compilation_rand_seq/data/michelutti_tdds/ldd_randgen/data",
    #     "results/tddnnf_compilation_rand_seq/data/michelutti_tdds/randgen/data",
    # ]
    # previous_times, prev_timeouts = get_current_results_times(
    #     tddnnf_err_file, tlemmas_paths, tddnnf_paths
    # )

    # # ###########################################################################
    # # # RAND - PARALLEL (WITHOUT PROJECTION ON T-ATOMS)
    # tddnnf_err_file = "results/tddnnf_compilation_rand_par/errors.json"
    # tlemmas_paths = [
    #     "data/results/merged_all_tlemmas/ldd_randgen/data",
    #     "data/results/merged_all_tlemmas/randgen/data",
    # ]

    # tddnnf_paths = [
    #     "results/tddnnf_compilation_rand_par/data/michelutti_tdds/ldd_randgen/data",
    #     "results/tddnnf_compilation_rand_par/data/michelutti_tdds/randgen/data",
    # ]
    # current_times, curr_timeouts = get_current_results_times(
    #     tddnnf_err_file, tlemmas_paths, tddnnf_paths
    # )

    # # ###########################################################################
    # # # RAND - PARALLEL WITH PROJECTION ON T-ATOMS
    # tddnnf_err_file = "results/tddnnf_compilation_rand_proj/errors.json"
    # tlemmas_paths = [
    #     "data/results/merged_all_tlemmas_projected/ldd_randgen/data",
    #     "data/results/merged_all_tlemmas_projected/randgen/data",
    # ]

    # tddnnf_paths = [
    #     "results/tddnnf_compilation_rand_proj/data/michelutti_tdds/ldd_randgen/data",
    #     "results/tddnnf_compilation_rand_proj/data/michelutti_tdds/randgen/data",
    # ]
    # x3_times, x3_timeouts = get_current_results_times(
    #     tddnnf_err_file, tlemmas_paths, tddnnf_paths
    # )

    # # ###########################################################################
    # # # RAND - PARALLEL WITH PROJECTION ON T-ATOMS
    # tddnnf_err_file = "results/tddnnf_compilation_rand_part/errors.json"
    # tlemmas_paths = [
    #     "data/results/tlemmas_rand_partition/ldd_randgen/data",
    #     "data/results/tlemmas_rand_partition/randgen/data",
    # ]

    # tddnnf_paths = [
    #     "results/tddnnf_compilation_rand_part/data/michelutti_tdds/ldd_randgen/data",
    #     "results/tddnnf_compilation_rand_part/data/michelutti_tdds/randgen/data",
    # ]
    # x4_times, x4_timeouts = get_current_results_times(
    #     tddnnf_err_file, tlemmas_paths, tddnnf_paths
    # )

    ###########################################################################
    ###########################################################################

    ###########################################################################
    # PLANNING - SEQUENTIAL
    tddnnf_err_file = "results/tddnnf_compilation_planning_seq/errors.json"
    tlemmas_paths = [
        "data/results/merged_planning_seq/planning",
    ]

    tddnnf_paths = [
        "results/tddnnf_compilation_planning_seq/data/benchmark",
    ]
    previous_times, prev_timeouts = get_current_results_times(
        tddnnf_err_file, tlemmas_paths, tddnnf_paths
    )

    ###########################################################################
    # PLANNING - PARALLEL (WITHOUT PROJECTION ON T-ATOMS)
    tddnnf_err_file = "results/tddnnf_compilation_planning_par/errors.json"
    tlemmas_paths = [
        "data/results/merged_planning_par/planning",
    ]

    tddnnf_paths = [
        "results/tddnnf_compilation_planning_par/data/benchmark",
    ]
    current_times, curr_timeouts = get_current_results_times(
        tddnnf_err_file, tlemmas_paths, tddnnf_paths
    )

    ###########################################################################
    # PLANNING - PARALLEL WITH PROJECTION ON T-ATOMS
    tddnnf_err_file = "results/tddnnf_compilation_planning_proj/errors.json"
    tlemmas_paths = [
        "data/results/merged_planning_proj/planning",
    ]

    tddnnf_paths = [
        "results/tddnnf_compilation_planning_proj/data/benchmark",
    ]
    x3_times, x3_timeouts = get_current_results_times(
        tddnnf_err_file, tlemmas_paths, tddnnf_paths
    )

    ###########################################################################
    # PLANNING - PARALLEL WITH PARTITIONING
    tddnnf_err_file = None  # "results/tddnnf_compilation_planning_part/errors.json"
    tlemmas_paths = [
        "data/results/merged_planning_part/planning",
    ]

    tddnnf_paths = [
        "results/tddnnf_compilation_planning_part/data/benchmark",
    ]
    x4_times, x4_timeouts = get_current_results_times(
        tddnnf_err_file, tlemmas_paths, tddnnf_paths
    )

    ###########################################################################
    ###########################################################################

    # prev_keys = set(previous_times.keys())
    curr_keys = set(current_times.keys())
    x3_keys = set(current_times.keys())
    # print("Missing keys:", prev_keys - curr_keys)
    print("Missing keys:", curr_keys - x3_keys)

    prev_solver = "Baseline"
    curr_solver = "D&C"
    x3_solver = "D&C+Proj"
    x4_solver = "D&C+Proj+Part"

    # Scatter
    # create_scatter_plot(
    #     previous_times,
    #     current_times,
    #     prev_solver,
    #     curr_solver,
    #     out_path="aggr_seq_vs_par45_tlemmas_gen_and_comp.pdf",
    # )
    # create_scatter_plot(
    #     previous_times,
    #     x3_times,
    #     prev_solver,
    #     x3_solver,
    #     out_path="aggr_seq_vs_par45_proj_atoms_tlemmas_gen_and_comp.pdf",
    # )
    # create_scatter_plot(
    #     current_times,
    #     x3_times,
    #     curr_solver,
    #     x3_solver,
    #     out_path="aggr_par45_vs_par45_proj_atoms_tlemmas_gen_and_comp.pdf",
    # )

    # Cactus
    create_cactus_plot(
        previous_times,
        current_times,
        prev_solver,
        curr_solver,
        third=x3_times,
        third_label=x3_solver,
        fourth=x4_times,
        fourth_label=x4_solver,
        out_path="cactus_aggr_seq_vs_par45_vs_par45_proj_atoms_tlemmas_gen_and_comp.pdf",
    )

    # create_scatter_plot(current_times, x3_times, curr_solver, x3_solver)
    # create_cactus_plot(current_times, x3_times, curr_solver, x3_solver)
