import json
import os

from pysmt.shortcuts import read_smtlib

import matplotlib.pyplot as plt
import numpy as np

PREVIOUS_RESULTS_PATHS = [
    # "data/michelutti_tdds/randgen/output_tddnnf_d4_total_new",
    # "data/michelutti_tdds/ldd_randgen/output_tddnnf_d4_total_new",
    "data/michelutti_tdds/ldd_randgen/tmp_total_new",
    "data/michelutti_tdds/randgen/tmp_total_new",
]

CURRENT_RESULTS_PATHS = [
    "data/results/merged_all_tlemmas/ldd_randgen",
    "data/results/merged_all_tlemmas/randgen",
    # "data/results/planning_h3_parallel/planning_h3_1Prob_45Procs/data/benchmark/planning/h3",
]

CURRENT_RESULTS_ERROR_FILE = "data/results/merged_all_tlemmas/errors.json"

DEFAULT_TIMEOUT = 3600.0  # seconds
PREVIOUS_RESULTS_TIME_KEY = "All-SMT computation time"
CURRENT_RESULTS_TLEMMAS_NUM_KEY = "T-lemmas amount"


def extract_tlemmas_from_smt2(path: str) -> int:
    try:
        lemmas = read_smtlib(path)
        if lemmas.is_or():
            return 1

        assert lemmas.is_and()

        return len(lemmas.args())
    except:
        return None


def get_previous_results_times() -> tuple[dict, dict]:
    times = {}
    tlemmas = {}
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
                        tlemmas[key_name] = data["total lemmas"]

                        # smt_file = file_path.replace(".json", ".smt2")
                        # lemmas_num = extract_tlemmas_from_smt2(smt_file)
                        # if lemmas_num is not None:
                        #     tlemmas[key_name] = lemmas_num
                    else:
                        times[key_name] = DEFAULT_TIMEOUT

    return times, tlemmas


def get_current_results_times(err_file: str, paths: list[str]) -> tuple[dict, dict]:
    times = {}
    tlemmas = {}

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
                tlemmas[problem_name] = data["T-DDNNF"][CURRENT_RESULTS_TLEMMAS_NUM_KEY]

    return times, tlemmas


def create_bar_plot(previous: dict, current: dict):
    problems = list(current.keys())
    previous_times = [previous[problem] for problem in problems]
    current_times = [current[problem] for problem in problems]

    x = range(len(problems))

    plt.figure(figsize=(36, 6))
    plt.bar(x, previous_times, width=0.4, label="Sequential AllSMT", align="center")
    plt.bar(
        [i + 0.4 for i in x],
        current_times,
        width=0.4,
        label="Parallel AllSMT",
        align="center",
    )

    plt.xlabel("Problems")
    plt.ylabel("Generation Time (seconds)")
    plt.title("Comparison of T-Lemmas Generation Time")
    plt.xticks([i + 0.2 for i in x], problems, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig("tlemmas_bar.png")


def create_cactus_plot(
    previous: dict,
    current: dict,
    x1_label: str,
    x2_label: str,
    third: dict | None = None,  # Eventual third solver data
    x3_label: str | None = None,  # Eventual third solver label
    show_vbs: bool = False,
    out_path: str = "cactus.png",
) -> None:
    previous_times = []
    current_times = []
    third_times = []
    vbs_times = []
    for problem in current:
        prev_time = (
            previous[problem]
            if previous[problem] <= DEFAULT_TIMEOUT
            else DEFAULT_TIMEOUT
        )
        current_time = (
            current[problem] if current[problem] <= DEFAULT_TIMEOUT else DEFAULT_TIMEOUT
        )

        third_time = None
        if third is not None:
            third_time = (
                third[problem] if third[problem] <= DEFAULT_TIMEOUT else DEFAULT_TIMEOUT
            )

        vbs_time = min(prev_time, current_time)
        if third_time:
            vbs_time = min(vbs_time, third_time)

        previous_times.append(prev_time)
        current_times.append(current_time)
        if third_time:
            third_times.append(third_time)
        vbs_times.append(vbs_time)

    previous_times.sort()
    current_times.sort()
    third_times.sort()
    vbs_times.sort()

    x1 = np.arange(1, len(previous_times) + 1)
    x2 = np.arange(1, len(current_times) + 1)
    x3 = np.arange(1, len(third_times) + 1)
    x4 = np.arange(1, len(vbs_times) + 1)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x1, previous_times, label=x1_label, marker="o", markersize=2)
    plt.plot(x2, current_times, label=x2_label, marker="^", markersize=2)
    if len(x3) > 0:
        plt.plot(x3, third_times, label=x3_label, marker="+", markersize=2)
    if show_vbs:
        plt.plot(x4, vbs_times, label="Virtual Best", marker="s", markersize=1)

    plt.xlabel("Number of problems solved")
    plt.ylabel("Time (s)")
    plt.title(
        f"Solvers comparison: {x1_label} vs {x2_label}"
        + (f" vs {x3_label}" if x3_label else "")
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)


def create_scatter_plot(
    previous: dict,
    current: dict,
    x_label: str,
    y_label: str,
    lower_threshold: float = 1.0,
    out_path: str = "scatter.png",
):
    previous_times = []
    current_times = []
    previous_timeouts = 0
    current_timeouts = 0

    previous_under_lower_threshold = 0
    current_under_lower_threshold = 0

    for problem in current.keys():
        previous_times.append(previous[problem])
        current_times.append(current[problem])

        if previous[problem] >= DEFAULT_TIMEOUT:
            previous_timeouts += 1
        elif previous[problem] <= lower_threshold:
            previous_under_lower_threshold += 1

        if current[problem] >= DEFAULT_TIMEOUT:
            current_timeouts += 1
        elif current[problem] <= lower_threshold:
            current_under_lower_threshold += 1

    timeout = DEFAULT_TIMEOUT
    linthresh = 1  # Linear region until 1

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 7))

    # Scatter plot
    ax.scatter(
        x=current_times,
        y=previous_times,
        color="darkgreen",
        edgecolors="black",
        s=25,
        zorder=4,
        alpha=0.2,
    )

    # Reference line y = x
    ax.plot([1e-2, timeout], [1e-2, timeout], "k--", label="y = x", zorder=2)

    # Timeout lines (dashed)
    ax.axvline(
        timeout,
        linestyle="--",
        color="gray",
        label=(
            f"{x_label} timeouts: {current_timeouts} "  # noqa
            f"| below {lower_threshold} sec: {current_under_lower_threshold}"
        ),
    )
    ax.axhline(
        timeout,
        linestyle="--",
        color="gray",
        label=(
            f"{y_label} timeouts: {previous_timeouts} "  # noqa
            f"| below {lower_threshold} sec: {previous_under_lower_threshold}"
        ),
    )

    # Set symlog scale
    ax.set_xscale("symlog", linthresh=linthresh)
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_aspect("equal")

    # Set limits
    ax.set_xlim(left=1e-2, right=timeout * 1.1)
    ax.set_ylim(bottom=1e-2, top=timeout * 1.1)

    # Labels
    ax.set_xlabel(f"{x_label} times", fontsize=12)
    ax.set_ylabel(f"{y_label} times", fontsize=12)

    # Grid
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    # Legend
    ax.legend(loc="lower right")

    # Show plot
    plt.tight_layout()
    plt.savefig(out_path)


def create_tlemmas_scatter_plot(
    previous: dict,
    current: dict,
    prev_label: str,
    curr_label: str,
    out_path: str = "scatter_num.png",
):
    previous_times = []
    current_times = []

    for problem in current.keys():
        if problem not in previous:
            continue

        previous_times.append(previous[problem])
        current_times.append(current[problem])

    timeout = max(max(previous_times), max(current_times))
    # linthresh = 1  # Linear region until 1

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 7))

    # Scatter plot
    ax.scatter(
        x=current_times,
        y=previous_times,
        color="darkgreen",
        edgecolors="black",
        s=25,
        zorder=4,
        alpha=0.2,
    )

    # Reference line y = x
    ax.plot([1e-2, timeout], [1e-2, timeout], "k--", label="y = x", zorder=2)

    # Timeout lines (dashed)
    ax.axvline(timeout, linestyle="--", color="gray")
    ax.axhline(timeout, linestyle="--", color="gray")

    # Set symlog scale
    # ax.set_xscale('symlog', linthresh=linthresh)
    # ax.set_yscale('symlog', linthresh=linthresh)
    ax.set_aspect("equal")

    # Set limits
    ax.set_xlim(left=1e-2, right=timeout * 1.1)
    ax.set_ylim(bottom=1e-2, top=timeout * 1.1)

    # Labels
    ax.set_xlabel(f"{curr_label} T-lemmas number", fontsize=12)
    ax.set_ylabel(f"{prev_label} T-lemmas number", fontsize=12)

    # Grid
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    # Legend
    ax.legend()

    # Show plot
    plt.tight_layout()
    plt.savefig(out_path)


if __name__ == "__main__":
    # previous_times, prev_tlemmas = get_previous_results_times()

    ###########################################################################
    # RAND PROBLEMS
    previous_times, prev_tlemmas = get_current_results_times(
        "data/results/merged_all_tlemmas_sequential/errors.json",
        [
            "data/results/merged_all_tlemmas_sequential/ldd_randgen/data",  # noqa
            "data/results/merged_all_tlemmas_sequential/randgen/data",
        ],
    )

    current_times, curr_tlemmas = get_current_results_times(
        "data/results/merged_all_tlemmas/errors.json",
        [
            "data/results/merged_all_tlemmas/ldd_randgen/data",
            "data/results/merged_all_tlemmas/randgen/data",
        ],
    )

    x3_times, x3_tlemmas = get_current_results_times(
        "data/results/merged_all_tlemmas_projected/errors.json",
        [
            "data/results/merged_all_tlemmas_projected/ldd_randgen/data",  # noqa
            "data/results/merged_all_tlemmas_projected/randgen/data",
        ],
    )

    ###########################################################################
    # PLANNING PROBLEMS
    # previous_times, prev_tlemmas = get_current_results_times(
    #     "/home/ecivini/Projects/MSc/Thesis/Thesis/data/results/planning_sequential/qui_tlemmas_planning_h3_1Prob_Sequential/errors.json",
    #     [
    #         "/home/ecivini/Projects/MSc/Thesis/Thesis/data/results/planning_sequential/qui_tlemmas_planning_h3_1Prob_Sequential/data/benchmark/planning/h3/Painter",  # noqa
    #     ],
    # )

    # current_times, curr_tlemmas = get_current_results_times(
    #     "/home/ecivini/Projects/MSc/Thesis/Thesis/data/results/planning_parallel45/quo_tlemmas_planning_h3_1Prob_45Procs/errors.json",
    #     [
    #         "/home/ecivini/Projects/MSc/Thesis/Thesis/data/results/planning_parallel45/quo_tlemmas_planning_h3_1Prob_45Procs/data/benchmark/planning/h3/Painter",  # noqa
    #     ],
    # )

    # x3_times, x3_tlemmas = get_current_results_times(
    #     "/home/ecivini/Projects/MSc/Thesis/Thesis/data/results/planning_parallel45_proj/qua_tlemmas_planning_h3_1Prob_45Procs_ProjectedAtoms/errors.json",
    #     [
    #         "/home/ecivini/Projects/MSc/Thesis/Thesis/data/results/planning_parallel45_proj/qua_tlemmas_planning_h3_1Prob_45Procs_ProjectedAtoms/data/benchmark/planning/h3/Painter",  # noqa
    #     ],
    # )

    prev_problems = set(previous_times.keys())
    curr_problems = set(current_times.keys())

    prob_diff = curr_problems - prev_problems
    if len(prob_diff) > 0:
        for problem in prob_diff:
            previous_times[problem] = DEFAULT_TIMEOUT
        prev_problems = set(previous_times.keys())

    assert prev_problems == curr_problems, "Problems don't match"

    x3_problems = set(x3_times.keys())
    assert prev_problems == x3_problems, "Problems don't match with x3_times"

    # create_bar_plot(previous_times, current_times)
    solver_prev = "Sequential AllSMT"
    solver_curr = "Parallel[45]"
    solver_x3 = "Parallel[45] with projected T-atoms"

    # Scatter plots
    create_scatter_plot(
        previous_times,
        current_times,
        x_label=solver_curr,
        y_label=solver_prev,
        out_path="seq_vs_par45_tlemmas_gen_time.png",
    )
    create_scatter_plot(
        previous_times,
        x3_times,
        x_label=solver_x3,
        y_label=solver_curr,
        out_path="seq_vs_par45_proj_atoms_tlemmas_gen_time.png",
    )
    create_scatter_plot(
        current_times,
        x3_times,
        x_label=solver_x3,
        y_label=solver_curr,
        out_path="par45_vs_par45_proj_atoms_tlemmas_gen_time.png",
    )

    create_tlemmas_scatter_plot(
        prev_tlemmas,
        curr_tlemmas,
        solver_prev,
        solver_curr,
        out_path="seq_vs_par45_tlemmas_num.png",
    )

    create_tlemmas_scatter_plot(
        prev_tlemmas,
        x3_tlemmas,
        solver_prev,
        solver_x3,
        out_path="seq_vs_par45_proj_atoms_tlemmas_num.png",
    )

    create_tlemmas_scatter_plot(
        curr_tlemmas,
        x3_tlemmas,
        solver_curr,
        solver_x3,
        out_path="par45_vs_par45_proj_atoms_tlemmas_num.png",
    )

    # Cactus plots
    create_cactus_plot(
        previous_times,
        current_times,
        x1_label=solver_prev,
        x2_label=solver_curr,
        third=x3_times,
        x3_label=solver_x3,
        out_path="cactus_seq_vs_par45_vs_par45_proj_atoms_tlemmas_gen_time.png",
    )
