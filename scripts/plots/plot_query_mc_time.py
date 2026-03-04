import json
import os

import matplotlib.pyplot as plt
import numpy as np


def get_results(paths: list[str], err_file: str | None = None) -> tuple:
    smt_times = {}
    smt_counts = {}

    tddnnf_times = {}
    tddnnf_counts = {}

    tbdd_times = {}
    tbdd_counts = {}

    tsdd_times = {}
    tsdd_counts = {}

    query_errors = {}

    if err_file:
        with open(err_file, "r") as f:
            errors = json.load(f)
            for problem, reason in errors.items():
                problem_name = "".join(
                    problem.split(os.sep)[-4:]
                )  # problem.split(os.sep)[-1].replace(".smt2", "")
                query_errors[problem_name] = reason

    for base_dir in paths:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file != "logs.json":
                    continue

                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    data = json.load(f)

                problem_name = "".join(file_path.split(os.sep)[-4:])
                smt_times[problem_name] = data["MC"]["AllSMT time"]
                smt_counts[problem_name] = data["MC"]["AllSMT count"]

                tddnnf_times[problem_name] = data["MC"]["d-DNNF time"]
                tddnnf_counts[problem_name] = data["MC"]["d-DNNF count"]

                tbdd_times[problem_name] = data["MC"]["T-BDD time"]
                tbdd_counts[problem_name] = data["MC"]["T-BDD count"]

                tsdd_times[problem_name] = data["MC"]["T-SDD time"]
                tsdd_counts[problem_name] = data["MC"]["T-SDD count"]

                # Do not check the model count against AllSMT as, when
                # we use projection or partitioning, the result is not the one we expect
                # in this context
                # if smt_counts[problem_name] is not None:
                #     assert smt_counts[problem_name] == tddnnf_counts[problem_name]

                if tbdd_times[problem_name] is not None:
                    assert tddnnf_counts[problem_name] == tddnnf_counts[problem_name]

                if tsdd_times[problem_name] is not None:
                    tddnnf_counts[problem_name] == tddnnf_counts[problem_name]

    return (
        smt_times,
        smt_counts,
        tddnnf_times,
        tddnnf_counts,
        tbdd_times,
        tbdd_counts,
        tsdd_times,
        tsdd_counts,
        query_errors,
    )


def extract_time_from_log(data: dict) -> list:
    times = []
    for problem in data:
        time = data[problem]
        if time:
            times.append(time)
    return times


def create_cactus_plot(
    x1: dict,  # AllSMT
    x2: dict,  # T-d-DNNF
    x3: dict,  # T-d-DNNF
    x4: dict,  # T-d-DNNF
    x5: dict,  # T-d-DNNF
    x1_label: str,
    x2_label: str,
    x3_label: str,
    x4_label: str,
    x5_label: str,
    x6: dict | None = None,  # T-OBDD
    x7: dict | None = None,  # T-SDD
    x6_label: str | None = None,
    x7_label: str | None = None,
    exclude_allsmt: bool = False,
    out_path: str = "cactus.pdf",
):
    x1_times = extract_time_from_log(x1)
    x2_times = extract_time_from_log(x2)
    x3_times = extract_time_from_log(x3)
    x4_times = extract_time_from_log(x4)
    x5_times = extract_time_from_log(x5)
    x6_times = []
    if x6:
        x6_times = extract_time_from_log(x6)
    x7_times = []
    if x7:
        x7_times = extract_time_from_log(x7)

    x1_times.sort()
    x2_times.sort()
    x3_times.sort()
    x4_times.sort()
    x5_times.sort()
    x6_times.sort()
    x7_times.sort()

    x1_arr = np.arange(1, len(x1_times) + 1)
    x2_arr = np.arange(1, len(x2_times) + 1)
    x3_arr = np.arange(1, len(x3_times) + 1)
    x4_arr = np.arange(1, len(x4_times) + 1)
    x5_arr = np.arange(1, len(x5_times) + 1)
    x6_arr = np.arange(1, len(x6_times) + 1)
    x7_arr = np.arange(1, len(x7_times) + 1)

    # Plot
    plt.figure(figsize=(9, 6))
    if not exclude_allsmt:
        plt.plot(x1_arr, x1_times, label=x1_label, marker="o", markersize=2)
    plt.plot(x2_arr, x2_times, label=x2_label, marker="o", markersize=2)
    plt.plot(x3_arr, x3_times, label=x3_label, marker="o", markersize=2)
    plt.plot(x4_arr, x4_times, label=x4_label, marker="o", markersize=2)
    plt.plot(x5_arr, x5_times, label=x5_label, marker="o", markersize=2)
    if x6:
        plt.plot(x6_arr, x6_times, label=x6_label, marker="o", markersize=2)
    if x7:
        plt.plot(x7_arr, x7_times, label=x7_label, marker="o", markersize=2)

    plt.xlabel("Queried problems (MC)", fontsize=24)
    plt.ylabel("Time (s)", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def create_scatter_plot(
    x_data: dict,
    y_data: dict,
    x_label: str,
    y_label: str,
    out_path: str = "scatter.pdf",
):
    x_times = []
    y_times = []

    problems = (
        x_data.keys() if len(x_data.keys()) <= len(y_data.keys()) else y_data.keys()
    )

    for problem in problems:
        if x_data[problem] is None or y_data[problem] is None:
            continue
        x_times.append(x_data[problem])
        y_times.append(y_data[problem])

    if not x_times or not y_times:
        print("No data for:", out_path)
        return

    timeout = max(max(x_times), max(y_times))

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 5))

    # Scatter plot
    ax.scatter(
        x=x_times,
        y=y_times,
        color="lightskyblue",
        edgecolors="black",
        s=100,
        zorder=4,
        alpha=1,
        marker="X",
    )

    # Reference line y = x
    ax.plot(
        [0, timeout],
        [0, timeout],
        zorder=2,
        color="gray",
        linestyle="--",
    )

    # Set symlog scale
    ax.set_aspect("equal")

    # Set limits
    ax.set_xlim(left=1e-8, right=timeout * 1.1)
    ax.set_ylim(bottom=1e-8, top=timeout * 1.1)

    # Labelsout_path
    ax.set_xlabel(f"{x_label}", fontsize=24)
    ax.set_ylabel(f"{y_label}", fontsize=24)

    # Grid
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    # Show plot
    plt.xticks(fontsize=18, rotation=45)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    allsmt_type = "AllSMT"
    nnf_seq_type = "T-d-DNNF (Baseline)"
    nnf_par_type = "T-d-DNNF (D&C)"
    nnf_proj_type = "T-d-DNNF (Proj)"
    nnf_part_type = "T-d-DNNF (Part)"
    tbdd_type = "T-OBDD"
    tsdd_type = "T-SDD"

    ###########################################################################

    # RAND PROBLEMS
    # Baseline
    x1_err_file = "results/query_mc_rand_seq_above100s/errors.json"
    x1_logs = ["results/query_mc_rand_seq_above100s/data/michelutti_tdds"]

    # D&C
    x2_err_file = "results/query_mc_rand_par_above100s/errors.json"
    x2_logs = ["results/query_mc_rand_par_above100s/data/michelutti_tdds"]

    # D&C+Proj
    x3_err_file = "results/query_mc_rand_proj_above100s/errors.json"
    x3_logs = ["results/query_mc_rand_proj_above100s/data/michelutti_tdds"]

    # D&C+Proj+Part
    x4_err_file = "results/query_mc_rand_proj_above100s/errors.json"
    x4_logs = ["results/query_mc_rand_proj_above100s/data/michelutti_tdds"]

    ###########################################################################

    # # PLANNING PROBLEMS
    # # Baseline
    # x1_err_file = "results/query_mc_planning_seq_incr_smt/errors.json"
    # x1_logs = ["results/query_mc_planning_seq_incr_smt/data/benchmark"]

    # # D&C
    # x2_err_file = "results/query_mc_planning_par_incr_smt/errors.json"
    # x2_logs = ["results/query_mc_planning_par_incr_smt/data/benchmark"]

    # # D&C+Proj
    # x3_err_file = "results/query_mc_planning_proj_incr_smt/errors.json"
    # x3_logs = ["results/query_mc_planning_proj_incr_smt/data/benchmark"]

    # # D&C+Proj+Part
    # x4_err_file = None
    # x4_logs = ["results/query_mc_planning_part_incr_smt/data/benchmark"]

    ###########################################################################
    # Load data
    (
        _,
        _,
        x1_nnf_times,
        x1_nnf_counts,
        _,
        _,
        _,
        _,
        x1_errors,
    ) = get_results(x1_logs, x1_err_file)

    (
        _,
        _,
        x2_nnf_times,
        x2_nnf_counts,
        _,
        _,
        _,
        _,
        x2_errors,
    ) = get_results(x2_logs, x2_err_file)

    (
        _,
        _,
        x3_nnf_times,
        x3_nnf_counts,
        _,
        _,
        _,
        _,
        x3_errors,
    ) = get_results(x3_logs, x3_err_file)

    (
        x4_smt_times,
        x4_smt_counts,
        x4_nnf_times,
        x4_nnf_counts,
        x4_tbdd_times,
        x4_tbdd_counts,
        x4_tsdd_times,
        x4_tsdd_counts,
        x4_errors,
    ) = get_results(x4_logs, x4_err_file)

    ###########################################################################

    # Scatter plots vs AllSMT
    create_scatter_plot(
        x4_smt_times,
        x1_nnf_times,
        x_label=allsmt_type,
        y_label=nnf_seq_type,
        out_path="query_mc_allsmt_vs_seq.pdf",
    )
    create_scatter_plot(
        x4_smt_times,
        x2_nnf_times,
        x_label=allsmt_type,
        y_label=nnf_par_type,
        out_path="query_mc_allsmt_vs_par.pdf",
    )

    create_scatter_plot(
        x4_smt_times,
        x3_nnf_times,
        x_label=allsmt_type,
        y_label=nnf_proj_type,
        out_path="query_mc_allsmt_vs_proj.pdf",
    )

    create_scatter_plot(
        x4_smt_times,
        x4_nnf_times,
        x_label=allsmt_type,
        y_label=nnf_part_type,
        out_path="query_mc_allsmt_vs_part.pdf",
    )

    # Scatter plots NNF against each other
    create_scatter_plot(
        x1_nnf_times,
        x2_nnf_times,
        x_label=nnf_seq_type,
        y_label=nnf_par_type,
        out_path="query_mc_nnf_seq_vs_nnf_par.pdf",
    )

    create_scatter_plot(
        x1_nnf_times,
        x3_nnf_times,
        x_label=nnf_seq_type,
        y_label=nnf_proj_type,
        out_path="query_mc_nnf_seq_vs_nnf_proj.pdf",
    )

    create_scatter_plot(
        x1_nnf_times,
        x4_nnf_times,
        x_label=nnf_seq_type,
        y_label=nnf_part_type,
        out_path="query_mc_nnf_seq_vs_nnf_part.pdf",
    )

    create_scatter_plot(
        x2_nnf_times,
        x3_nnf_times,
        x_label=nnf_par_type,
        y_label=nnf_proj_type,
        out_path="query_mc_nnf_par_vs_nnf_proj.pdf",
    )

    create_scatter_plot(
        x2_nnf_times,
        x4_nnf_times,
        x_label=nnf_par_type,
        y_label=nnf_part_type,
        out_path="query_mc_nnf_par_vs_nnf_part.pdf",
    )

    create_scatter_plot(
        x3_nnf_times,
        x4_nnf_times,
        x_label=nnf_proj_type,
        y_label=nnf_part_type,
        out_path="query_mc_nnf_proj_vs_nnf_part.pdf",
    )

    # Scatter plots vs T-OBDD
    create_scatter_plot(
        x4_tbdd_times,
        x1_nnf_times,
        x_label=tbdd_type,
        y_label=nnf_seq_type,
        out_path="query_mc_tbdd_vs_seq.pdf",
    )
    create_scatter_plot(
        x4_tbdd_times,
        x2_nnf_times,
        x_label=tbdd_type,
        y_label=nnf_par_type,
        out_path="query_mc_tbdd_vs_par.pdf",
    )

    create_scatter_plot(
        x4_tbdd_times,
        x3_nnf_times,
        x_label=tbdd_type,
        y_label=nnf_proj_type,
        out_path="query_mc_tbdd_vs_proj.pdf",
    )

    create_scatter_plot(
        x4_tbdd_times,
        x4_nnf_times,
        x_label=tbdd_type,
        y_label=nnf_part_type,
        out_path="query_mc_tbdd_vs_part.pdf",
    )

    # Scatter plots vs T-SDD
    create_scatter_plot(
        x4_tsdd_times,
        x1_nnf_times,
        x_label=tsdd_type,
        y_label=nnf_seq_type,
        out_path="query_mc_tsdd_vs_seq.pdf",
    )
    create_scatter_plot(
        x4_tsdd_times,
        x2_nnf_times,
        x_label=tsdd_type,
        y_label=nnf_par_type,
        out_path="query_mc_tsdd_vs_par.pdf",
    )

    create_scatter_plot(
        x4_tsdd_times,
        x3_nnf_times,
        x_label=tsdd_type,
        y_label=nnf_proj_type,
        out_path="query_mc_tsdd_vs_proj.pdf",
    )

    create_scatter_plot(
        x4_tsdd_times,
        x4_nnf_times,
        x_label=tsdd_type,
        y_label=nnf_part_type,
        out_path="query_mc_tsdd_vs_part.pdf",
    )

    ###########################################################################

    # CACTUS PLOTS
    create_cactus_plot(
        x1=x4_smt_times,
        x2=x1_nnf_times,
        x3=x2_nnf_times,
        x4=x3_nnf_times,
        x5=x4_nnf_times,
        x6=x4_tbdd_times,
        x7=x4_tsdd_times,
        x1_label=allsmt_type,
        x2_label=nnf_seq_type,
        x3_label=nnf_par_type,
        x4_label=nnf_proj_type,
        x5_label=nnf_part_type,
        x6_label=tbdd_type,
        x7_label=tsdd_type,
        exclude_allsmt=False,
        out_path="cactus_query_mc_times_all.pdf",
    )

    create_cactus_plot(
        x1=x4_smt_times,
        x2=x1_nnf_times,
        x3=x2_nnf_times,
        x4=x3_nnf_times,
        x5=x4_nnf_times,
        x6=x4_tbdd_times,
        x7=x4_tsdd_times,
        x1_label=allsmt_type,
        x2_label=nnf_seq_type,
        x3_label=nnf_par_type,
        x4_label=nnf_proj_type,
        x5_label=nnf_part_type,
        x6_label=tbdd_type,
        x7_label=tsdd_type,
        exclude_allsmt=True,
        out_path="cactus_query_mc_times_no_allsmt.pdf",
    )

    create_cactus_plot(
        x1=x4_smt_times,
        x2=x1_nnf_times,
        x3=x2_nnf_times,
        x4=x3_nnf_times,
        x5=x4_nnf_times,
        x6=x4_tbdd_times,
        # x7=x4_tsdd_times,
        x1_label=allsmt_type,
        x2_label=nnf_seq_type,
        x3_label=nnf_par_type,
        x4_label=nnf_proj_type,
        x5_label=nnf_part_type,
        x6_label=tbdd_type,
        # x7_label=tsdd_type,
        exclude_allsmt=True,
        out_path="cactus_query_mc_times_no_allsmt_no_tsdd.pdf",
    )

    create_cactus_plot(
        x1=x4_smt_times,
        x2=x1_nnf_times,
        x3=x2_nnf_times,
        x4=x3_nnf_times,
        x5=x4_nnf_times,
        # x6=x4_tbdd_times,
        # x7=x4_tsdd_times,
        x1_label=allsmt_type,
        x2_label=nnf_seq_type,
        x3_label=nnf_par_type,
        x4_label=nnf_proj_type,
        x5_label=nnf_part_type,
        # x6_label=tbdd_type,
        # x7_label=tsdd_type,
        exclude_allsmt=True,
        out_path="cactus_query_mc_times_only_nnf.pdf",
    )

    create_cactus_plot(
        x1=x4_smt_times,
        x2=x1_nnf_times,
        x3=x2_nnf_times,
        x4=x3_nnf_times,
        x5=x4_nnf_times,
        # x6=x4_tbdd_times,
        # x7=x4_tsdd_times,
        x1_label=allsmt_type,
        x2_label=nnf_seq_type,
        x3_label=nnf_par_type,
        x4_label=nnf_proj_type,
        x5_label=nnf_part_type,
        # x6_label=tbdd_type,
        # x7_label=tsdd_type,
        exclude_allsmt=False,
        out_path="cactus_query_mc_times_nnf_and_allsmt.pdf",
    )
