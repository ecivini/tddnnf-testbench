import json
import os

from theorydd.tdd.theory_bdd import TheoryBDD
import pysmt
from pysmt.shortcuts import read_smtlib
from pysmt.oracles import SizeOracle


import matplotlib.pyplot as plt
import numpy as np

PREVIOUS_RESULTS_PATHS = [
    "data/michelutti_tdds/ldd_randgen/output_tbdd_total_new",
    "data/michelutti_tdds/randgen/output_tbdd_total_new",
]

DEFAULT_TIMEOUT = 3600  # seconds

PREVIOUS_TOTAL_TIME_KEY = "total computation time"
CURRENT_TOTAL_TIME_KEY = "Total time"
CURRENT_TLEMMAS_TIME_KEY = "All-SMT computation time"

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
                    elif PREVIOUS_TOTAL_TIME_KEY in data:
                        times[key_name] = data[PREVIOUS_TOTAL_TIME_KEY]
                    else:
                        raise ValueError("Unhandled case for " + str(file_path))

    return times, timeouts


def compute_sizes(phi_path: str, base_path: str) -> tuple:
    pysmt.environment.reset_env()
    phi = read_smtlib(phi_path)
    tbdd = TheoryBDD(phi, folder_name=base_path)

    dd_nodes = tbdd.count_nodes()
    phi_nodes = SizeOracle().get_size(phi)

    return (dd_nodes, phi_nodes)


def get_current_results_times(
    err_file: str, paths: list[str], lemmas_err_file: str, lemmas_paths: list[str]
) -> tuple[dict, int, int, list]:
    times = {}

    sizes = []

    missing_tlemmas = 0
    timeouts = 0

    tlemmas_times = get_current_tlemmas_times(lemmas_err_file, lemmas_paths)

    if err_file:
        with open(err_file, "r") as f:
            errors = json.load(f)
            for problem, reason in errors.items():
                problem_name = (problem.split("/")[-1]).replace(".smt2", "")
                if reason == "Missing tlemmas":
                    missing_tlemmas += 1
                    times[problem_name] = DEFAULT_TIMEOUT
                elif reason == "timeout":
                    timeouts += 1
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
                times[problem_name] = (
                    data["T-BDD"][CURRENT_TOTAL_TIME_KEY] + tlemmas_times[problem_name]
                )
                assert tlemmas_times[problem_name] != 0
                assert times[problem_name] > data["T-BDD"][CURRENT_TOTAL_TIME_KEY]

                phi_path = "/".join(file_path.split(os.sep)[4:])
                phi_path = phi_path.replace("/logs.json", ".smt2").replace(
                    "serialized_tdds", "michelutti_tdds"
                )
                sizes.append(compute_sizes(phi_path, root))

    return times, missing_tlemmas, timeouts, sizes


def get_current_tlemmas_times(err_file: str, paths: list[str]) -> dict:
    times = {}

    with open(err_file, "r") as f:
        errors = json.load(f)
        for problem, reason in errors.items():
            if reason == "timeout":
                key_name = problem.split(os.sep)[-1].replace(".smt2", "")
                times[key_name] = DEFAULT_TIMEOUT
            else:
                print("Error reason:", reason)
                raise ValueError(
                    "Unhandled case while reading tlemmas for" + str(problem)
                )

    for base_dir in paths:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file != "logs.json":
                    continue

                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    data = json.load(f)

                problem_name = os.path.dirname(file_path).split(os.sep)[-1]
                times[problem_name] = data["T-DDNNF"][CURRENT_TLEMMAS_TIME_KEY]

    return times


def create_cactus_plot(
    # x1_data: dict,
    x2_data: dict,
    # x1_label: str,
    x2_label: str,
    x3_data: dict | None = None,
    x4_data: dict | None = None,
    x3_label: str | None = None,
    x4_label: str | None = None,
    out_path: str = "cactus.pdf",
    filter_timeouts: bool = True,
):
    # x1_times = []
    x2_times = []
    x3_times = []
    x4_times = []
    for problem in x2_data:
        if (
            problem not in x2_data
            or (x3_data is not None and problem not in x3_data)
            or (x4_data is not None and problem not in x4_data)
        ):
            continue

        # x1_time = x1_data[problem]
        # if filter_timeouts and x1_time > DEFAULT_TIMEOUT:
        #     x1_time = DEFAULT_TIMEOUT
        # x1_times.append(x1_time)

        x2_time = x2_data[problem]
        if filter_timeouts and x2_time > DEFAULT_TIMEOUT:
            x2_time = DEFAULT_TIMEOUT
        x2_times.append(x2_time)

        if x3_data is not None:
            x3_time = x3_data[problem]
            if filter_timeouts and x3_time > DEFAULT_TIMEOUT:
                x3_time = DEFAULT_TIMEOUT
            x3_times.append(x3_time)

        if x4_data is not None:
            x4_time = x4_data[problem]
            if filter_timeouts and x4_time > DEFAULT_TIMEOUT:
                x4_time = DEFAULT_TIMEOUT
            x4_times.append(x4_time)

    # x1_times.sort()
    x2_times.sort()
    x3_times.sort()
    x4_times.sort()

    # x1 = np.arange(1, len(x1_times) + 1)
    x2 = np.arange(1, len(x2_times) + 1)
    x3 = np.arange(1, len(x3_times) + 1)
    x4 = np.arange(1, len(x4_times) + 1)

    # Plot
    plt.figure(figsize=(9, 6))
    # plt.plot(x1, x1_times, label=x1_label, marker="o", markersize=2)
    plt.plot(x2, x2_times, label=x2_label, marker="^", markersize=2)

    if x3_data is not None:
        plt.plot(x3, x3_times, label=x3_label, marker="+", markersize=2)

    if x4_data is not None:
        plt.plot(x4, x4_times, label=x4_label, marker="*", markersize=2)

    plt.xlabel("Number of problems compiled", fontsize=24)
    plt.ylabel("Time (s)", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # title = f"T-BDD creation with different lemmas: {x1_label} vs {x2_label}"
    # if x3_data is not None:
    #     title += f" vs {x3_label}"
    #     if x4_data is not None:
    #         title += f" vs {x4_label}"

    # plt.title(title)
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(out_path)


def create_scatter_plot(
    x_data: list,
    y_data: list,
    x_label: str,
    y_label: str,
    out_path: str = "scatter.pdf",
    log_scale: bool = False,
    upperbound: float = 0.0,
    alpha: float = 1.0,
):
    timeout = max(max(x_data), max(y_data)) if upperbound == 0.0 else upperbound
    # Create figure
    fig, ax = plt.subplots(figsize=(5, 5))

    # Scatter plot
    ax.scatter(
        x=x_data,
        y=y_data,
        color="lightskyblue",
        edgecolors="black",
        s=100,
        zorder=4,
        alpha=alpha,
        marker="X",
    )

    # Reference line y = x
    ax.plot(
        [0, timeout],
        [0, timeout],
        label="y = x",
        zorder=2,
        color="gray",
        linestyle="--",
    )

    # Set symlog scale
    if log_scale:
        ax.set_xscale("symlog")
        ax.set_yscale("symlog")
    ax.set_aspect("equal")

    # Set limits
    ax.set_xlim(left=1e-8, right=timeout * 1.1)
    ax.set_ylim(bottom=1e-8, top=timeout * 1.1)

    # Labelsout_path
    ax.set_xlabel(f"{x_label}", fontsize=24)
    ax.set_ylabel(f"{y_label}", fontsize=24)

    # Grid
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    # Legend
    # ax.legend()

    # plt.title(f"Query comparison: {x_label} vs {y_label}")

    # Show plot
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    previous_times, prev_timeouts = get_previous_results_times()

    # seq_times, seq_missing_lemmas, seq_timeouts = get_current_results_times(
    #     "data/results/tbdd_seq/tbdd_sequential/errors.json",
    #     [
    #         "data/results/tbdd_seq/tbdd_sequential/data/serialized_tdds/ldd_randgen/data",  # noqa
    #         "data/results/tbdd_seq/tbdd_sequential/data/serialized_tdds/randgen/data",  # noqa
    #     ],
    #     "data/results/merged_all_tlemmas_sequential/errors.json",
    #     [
    #         "data/results/merged_all_tlemmas_sequential/ldd_randgen/data",
    #         "data/results/merged_all_tlemmas_sequential/randgen/data",
    #     ],
    # )

    # par_times, par_missing_lemmas, par_timeouts = get_current_results_times(
    #     "data/results/tbdd_par/tbdd_parallel/errors.json",
    #     [
    #         "data/results/tbdd_par/tbdd_parallel/data/serialized_tdds/ldd_randgen/data",  # noqa
    #         "data/results/tbdd_par/tbdd_parallel/data/serialized_tdds/randgen/data",  # noqa
    #     ],
    #     "data/results/merged_all_tlemmas/errors.json",
    #     [
    #         "data/results/merged_all_tlemmas/ldd_randgen/data",
    #         "data/results/merged_all_tlemmas/randgen/data",
    #     ],
    # )

    # par_proj_times, par_proj_missing_lemmas, par_proj_timeouts = (
    #     get_current_results_times(
    #         "data/results/tbdd_par_proj/tbdd_parallel_proj/errors.json",
    #         [
    #             "data/results/tbdd_par_proj/tbdd_parallel_proj/data/serialized_tdds/ldd_randgen/data",  # noqa
    #             "data/results/tbdd_par_proj/tbdd_parallel_proj/data/serialized_tdds/randgen/data",  # noqa
    #         ],
    #         "data/results/merged_all_tlemmas_projected/errors.json",
    #         [
    #             "data/results/merged_all_tlemmas_projected/ldd_randgen/data",
    #             "data/results/merged_all_tlemmas_projected/randgen/data",
    #         ],
    #     )
    # )

    # print("Previous timeouts", prev_timeouts)
    # print("Sequential timeouts:", seq_missing_lemmas + seq_timeouts)
    # print("Parallel[45] timeouts:", par_missing_lemmas + par_timeouts)
    # print(
    #     "Parallel[45] with projected atoms timeouts:",
    #     par_proj_missing_lemmas + par_proj_timeouts,
    # )

    # # create_bar_plot(previous_times, current_times)
    # # prev_solver = "AllSMT v1"
    # seq_solver = "AllSMT"
    # par_solver = "D&C"
    # par_proj_solver = "D&C+Proj"

    # assert (
    #     len(previous_times) == len(seq_times)
    #     and len(seq_times) == len(par_times)
    #     and len(par_times) == len(par_proj_times)
    # )

    # # Cactus - compilation times
    # create_cactus_plot(
    #     # x1_data=previous_times,
    #     x2_data=seq_times,
    #     x3_data=par_times,
    #     x4_data=par_proj_times,
    #     # x1_label=prev_solver,
    #     x2_label=seq_solver,
    #     x3_label=par_solver,
    #     x4_label=par_proj_solver,
    #     filter_timeouts=True,  # TODO: fix this when set to False
    #     out_path="tbdd_creation_only_all.pdf",
    # )

    par_proj_times, par_proj_missing_lemmas, par_proj_timeouts, par_proj_sizes = (
        get_current_results_times(
            "data/results/tbdd_par_proj/tbdd_parallel_proj/errors.json",
            [
                "data/results/tbdd_par_proj/tbdd_parallel_proj/data/serialized_tdds/ldd_randgen/data",  # noqa
                "data/results/tbdd_par_proj/tbdd_parallel_proj/data/serialized_tdds/randgen/data",  # noqa
            ],
            "data/results/merged_all_tlemmas_projected/errors.json",
            [
                "data/results/merged_all_tlemmas_projected/ldd_randgen/data",
                "data/results/merged_all_tlemmas_projected/randgen/data",
            ],
        )
    )

    dd_sizes = []
    phi_sizes = []
    for size in par_proj_sizes:
        dd_sizes.append(size[0])
        phi_sizes.append(size[1])

    create_scatter_plot(
        x_data=phi_sizes,
        y_data=dd_sizes,
        x_label="PHI DAG nodes",
        y_label="T-OBDD",
        out_path="phi_nodes_vs_tobdd_nodes.pdf",
        log_scale=True,
        upperbound=10**5 * 2,
        alpha=0.4,
    )
