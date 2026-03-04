import json
import os

import matplotlib.pyplot as plt
import numpy as np


def get_results(path: str) -> tuple:
    smt_times = {}
    tddnnf_times = {}
    tbdd_times = {}
    tsdd_times = {}

    with open(path, "r") as f:
        data = json.load(f)

    for cube in data:
        row = data[cube]

        smt_times[cube] = (row["SMT time"], row["SMT result"])
        tddnnf_times[cube] = (row["d-DNNF time"], row["d-DNNF result"])

        if "T-BDD result" in row:
            tbdd_times[cube] = (row["T-BDD time"], row["T-BDD result"])
        else:
            tbdd_times[cube] = (None, None)

        if "T-SDD result" in row:
            tsdd_times[cube] = (row["T-SDD time"], row["T-SDD result"])
        else:
            tsdd_times[cube] = (None, None)

        # tbdd_times[problem_name] = data["CE"]["T-BDD time"]
        # tbdd_counts[problem_name] = data["CE"]["T-BDD count"]

        # tsdd_times[problem_name] = data["CE"]["T-SDD time"]
        # tsdd_counts[problem_name] = data["CE"]["T-SDD count"]

        # assert smt_counts[problem_name] == tddnnf_counts[problem_name]

        # if tbdd_times[problem_name] is not None:
        #     assert smt_counts[problem_name] == tddnnf_counts[problem_name]

        # if tsdd_times[problem_name] is not None:
        #     smt_counts[problem_name] == tddnnf_counts[problem_name]

    return (
        smt_times,
        tddnnf_times,
        tbdd_times,
        tsdd_times,
    )


def data_to_times_list(data: dict) -> list:
    times = []
    for problem in data:
        for cube in data[problem]:
            time = data[problem][cube][0]
            if time:
                times.append(data[problem][cube][0])
    return times


# TODO: Cleanup
def create_cactus_plot(
    x1_times: dict,  # Incremental SMT
    x2_times: dict,  # T-DDNNF 1
    x3_times: dict,  # T-DDNNF 2
    x4_times: dict,  # T-DDNNF 3
    x5_times: dict,  # T-DDNNF 4
    x1_label: str,
    x2_label: str,
    x3_label: str,
    x4_label: str,
    x5_label: str,
    x6_times: dict | None = None,  # Optional - T-OBDD
    x7_times: dict | None = None,  # Optional - T-SDD
    x6_label: str | None = None,
    x7_label: str | None = None,
    out_path: str = "cactus.pdf",
):
    # Remove Nones
    x1_cleaned = data_to_times_list(x1_times)
    x2_cleaned = data_to_times_list(x2_times)
    x3_cleaned = data_to_times_list(x3_times)
    x4_cleaned = data_to_times_list(x4_times)
    x5_cleaned = data_to_times_list(x1_times)

    x6_cleaned = []
    if x6_times:
        x6_cleaned = data_to_times_list(x6_times)

    x7_cleaned = []
    if x7_times:
        x7_cleaned = data_to_times_list(x7_times)

    x1_cleaned.sort()
    x2_cleaned.sort()
    x3_cleaned.sort()
    x4_cleaned.sort()
    x5_cleaned.sort()
    x6_cleaned.sort()
    x7_cleaned.sort()

    x1 = np.arange(1, len(x1_cleaned) + 1)
    x2 = np.arange(1, len(x2_cleaned) + 1)
    x3 = np.arange(1, len(x3_cleaned) + 1)
    x4 = np.arange(1, len(x4_cleaned) + 1)
    x5 = np.arange(1, len(x5_cleaned) + 1)
    x6 = np.arange(1, len(x6_cleaned) + 1)
    x7 = np.arange(1, len(x7_cleaned) + 1)

    # Plot
    plt.figure(figsize=(9, 6))
    # if include_allsmt:
    plt.plot(x1, x1_cleaned, label=x1_label, marker="o", markersize=1)
    plt.plot(x2, x2_cleaned, label=x2_label, marker="o", markersize=1)
    plt.plot(x3, x3_cleaned, label=x3_label, marker="o", markersize=1)
    plt.plot(x4, x4_cleaned, label=x4_label, marker="o", markersize=1)
    plt.plot(x5, x5_cleaned, label=x5_label, marker="o", markersize=1)
    if x6_times:
        plt.plot(x6, x6_cleaned, label=x6_label, marker="o", markersize=1)
    if x7_times:
        plt.plot(x7, x7_cleaned, label=x7_label, marker="o", markersize=1)

    plt.xlabel("Queried problems (CE)", fontsize=24)
    plt.ylabel("Time (s)", fontsize=24)
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(out_path)


def create_scatter_plot(
    x_data: dict,
    y_data: dict,
    x_label: str,
    y_label: str,
    out_path: str = "scatter.pdf",
):
    x_sat_times = []
    y_sat_times = []
    x_unsat_times = []
    y_unsat_times = []

    current_max = 0

    problems = (
        x_data.keys() if len(x_data.keys()) <= len(y_data.keys()) else y_data.keys()
    )

    for problem in problems:
        x_cubes = set(x_data[problem])
        y_cubes = set(y_data[problem])
        assert x_cubes == y_cubes, "Not same cubes for " + out_path

        for cube in x_cubes:
            x_record = x_data[problem][cube]
            y_record = y_data[problem][cube]
            x_sat = x_record[1]
            y_sat = y_record[1]

            if x_sat is None or y_sat is None:
                continue

            assert x_sat == y_sat, (
                "Incongruent results "
                + str(x_sat)
                + " "
                + str(y_sat)
                + " - "
                + out_path
            )

            if x_sat:
                x_sat_times.append(x_record[0])
                y_sat_times.append(y_record[0])
            else:
                x_unsat_times.append(x_record[0])
                y_unsat_times.append(y_record[0])
            current_max = max(current_max, x_record[0], y_record[0])

    timeout = current_max

    print("\n\n", out_path)
    print("CE holds:", len(x_sat_times))
    print("CE doesn't hold:", len(x_unsat_times))

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 5))

    # Scatter plot
    ax.scatter(
        x=x_sat_times,
        y=y_sat_times,
        color="green",
        edgecolors="black",
        s=100,
        zorder=5,
        alpha=1,
        marker="X",
    )

    ax.scatter(
        x=x_unsat_times,
        y=y_unsat_times,
        color="red",
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
        label="y = x",
        zorder=2,
        color="gray",
        linestyle="--",
    )

    # Timeout lines (dashed)
    # ax.axvline(
    #     timeout,
    #     linestyle="--",
    #     color="gray",
    # )
    # ax.axhline(
    #     timeout,
    #     linestyle="--",
    #     color="gray",
    # )

    # Set symlog scale
    # ax.set_xscale("symlog", linthresh=1)
    # ax.set_yscale("symlog", linthresh=1)
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
    plt.xticks(fontsize=18, rotation=45)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(out_path)


def get_all_logs(base_path: str) -> tuple:
    all_smt_times = {}
    all_tddnnf_times = {}
    all_tbdd_times = {}
    all_tsdd_times = {}

    for root, _, files in os.walk(base_path):
        for file in files:
            if file != "logs.json":
                continue

            file_path = os.path.join(root, file)
            key = "".join(file_path.split("/")[2:-2])

            (
                smt_times,
                tddnnf_times,
                tbdd_times,
                tsdd_times,
            ) = get_results(file_path)
            all_smt_times[key] = smt_times
            all_tddnnf_times[key] = tddnnf_times
            all_tbdd_times[key] = tbdd_times
            all_tsdd_times[key] = tsdd_times

    return (
        all_smt_times,
        all_tddnnf_times,
        all_tbdd_times,
        all_tsdd_times,
    )


if __name__ == "__main__":

    ###########################################################################
    # RANDOM

    # tbdd_type = "T-OBDD"
    # tsdd_type = "T-SDD"

    # # Incremental SMT vs T-d-DNNNF (Sequential)
    # x1_path = "results/query_ce_rand_seq_above100s/data"
    # x1_nnf_type = "T-d-DNNF (Baseline)"
    # incr_smt_type = "Incremental SMT"

    # # Incremental SMT vs T-d-DNNNF (Parallel)
    # x2_path = "results/query_ce_rand_par_above100s/data"
    # x2_nnf_type = "T-d-DNNF (D&C)"

    # # Incremental SMT vs T-d-DNNNF (Paralle with Projection)
    # x3_path = "results/query_ce_rand_proj_above100s/data"
    # x3_nnf_type = "T-d-DNNF (Proj)"

    # # Incremental SMT vs T-d-DNNNF (Paralle with Projection and Partitionings)
    # x4_path = "results/query_ce_rand_part_above100s/data"
    # x4_nnf_type = "T-d-DNNF (Part)"

    # SMT vs T-d-DNNNF (Sequential)
    # x4_path = "results/test_query_ce_full_seq/data"
    # x4_nnf_type = "T-d-DNNF (D&C+Proj+Part)"
    # smt_type = "SMT"

    # # # SMT vs T-d-DNNNF (Parallel)
    # x5_path = "results/test_query_ce_full_par/data"
    # x5_nnf_type = "T-d-DNNF (Parallel)"

    # # # SMT vs T-d-DNNNF (Parallel with Projection)
    # x6_path = "results/test_query_ce_full_proj/data"
    # x6_nnf_type = "T-d-DNNF (Parallel with Projection)"

    ###########################################################################
    # PLANNING

    # # Incremental SMT vs T-d-DNNNF (Sequential)
    # x1_path = "results/query_ce_plan_seq/data"
    # x1_nnf_type = "T-d-DNNF (Baseline)"
    # incr_smt_type = "Incremental SMT"

    # # Incremental SMT vs T-d-DNNNF (Parallel)
    # x2_path = "results/query_ce_plan_par/data"
    # x2_nnf_type = "T-d-DNNF (D&C)"

    # # Incremental SMT vs T-d-DNNNF (Paralle with Projection)
    # x3_path = "results/query_ce_plan_proj/data"
    # x3_nnf_type = "T-d-DNNF (Proj)"

    # # Incremental SMT vs T-d-DNNNF (Paralle with Projection and Partitionings)
    # x4_path = "results/query_ce_plan_part/data"
    # x4_nnf_type = "T-d-DNNF (Part)"

    # (
    #     x1_smt_times,
    #     # x1_smt_counts,
    #     x1_tddnnf_times,
    #     # x1_tddnnf_counts,
    #     x1_tbdd_times,
    #     # x1_tbdd_counts,
    #     x1_tsdd_times,
    #     # x1_tsdd_counts,
    # ) = get_all_logs(x1_path)

    # (
    #     x2_smt_times,
    #     # x2_smt_counts,
    #     x2_tddnnf_times,
    #     # x2_tddnnf_counts,
    #     x2_tbdd_times,
    #     # x2_tbdd_counts,
    #     x2_tsdd_times,
    #     # x2_tsdd_counts,
    # ) = get_all_logs(x2_path)

    # (
    #     x3_smt_times,
    #     # x3_smt_counts,
    #     x3_tddnnf_times,
    #     # x3_tddnnf_counts,
    #     x3_tbdd_times,
    #     # x3_tbdd_counts,
    #     x3_tsdd_times,
    #     # x3_tsdd_counts,
    # ) = get_all_logs(x3_path)

    # (
    #     x4_smt_times,
    #     # x4_smt_counts,
    #     x4_tddnnf_times,
    #     # x4_tddnnf_counts,
    #     x4_tbdd_times,
    #     # x4_tbdd_counts,
    #     x4_tsdd_times,
    #     # x4_tsdd_counts,
    # ) = get_all_logs(x4_path)

    # ###########################################################

    # # vs Incremental SMT plots
    # create_scatter_plot(
    #     x_data=x1_smt_times,
    #     y_data=x1_tddnnf_times,
    #     x_label=incr_smt_type,
    #     y_label=x1_nnf_type,
    #     out_path="inc_smt_vs_seq_ce_query.pdf",
    # )

    # create_scatter_plot(
    #     x_data=x2_smt_times,
    #     y_data=x2_tddnnf_times,
    #     x_label=incr_smt_type,
    #     y_label=x2_nnf_type,
    #     out_path="inc_smt_vs_par_ce_query.pdf",
    # )

    # create_scatter_plot(
    #     x_data=x3_smt_times,
    #     y_data=x3_tddnnf_times,
    #     x_label=incr_smt_type,
    #     y_label=x3_nnf_type,
    #     out_path="inc_smt_vs_proj_ce_query.pdf",
    # )

    # create_scatter_plot(
    #     x_data=x4_smt_times,
    #     y_data=x4_tddnnf_times,
    #     x_label=incr_smt_type,
    #     y_label=x4_nnf_type,
    #     out_path="inc_smt_vs_part_ce_query.pdf",
    # )

    # # vs OBDD plots
    # create_scatter_plot(
    #     x_data=x1_tbdd_times,
    #     y_data=x1_tddnnf_times,
    #     x_label=tbdd_type,
    #     y_label=x1_nnf_type,
    #     out_path="tbdd_vs_seq_ce_query.pdf",
    # )

    # create_scatter_plot(
    #     x_data=x2_tbdd_times,
    #     y_data=x2_tddnnf_times,
    #     x_label=tbdd_type,
    #     y_label=x2_nnf_type,
    #     out_path="tbdd_vs_par_ce_query.pdf",
    # )

    # create_scatter_plot(
    #     x_data=x3_tbdd_times,
    #     y_data=x3_tddnnf_times,
    #     x_label=tbdd_type,
    #     y_label=x3_nnf_type,
    #     out_path="tbdd_vs_proj_ce_query.pdf",
    # )

    # create_scatter_plot(
    #     x_data=x4_tbdd_times,
    #     y_data=x4_tddnnf_times,
    #     x_label=tbdd_type,
    #     y_label=x4_nnf_type,
    #     out_path="tbdd_vs_part_ce_query.pdf",
    # )

    # # vs SDD plots
    # create_scatter_plot(
    #     x_data=x1_tsdd_times,
    #     y_data=x1_tddnnf_times,
    #     x_label=tsdd_type,
    #     y_label=x1_nnf_type,
    #     out_path="tsdd_vs_seq_ce_query.pdf",
    # )

    # create_scatter_plot(
    #     x_data=x2_tsdd_times,
    #     y_data=x2_tddnnf_times,
    #     x_label=tsdd_type,
    #     y_label=x2_nnf_type,
    #     out_path="tsdd_vs_par_ce_query.pdf",
    # )

    # create_scatter_plot(
    #     x_data=x3_tsdd_times,
    #     y_data=x3_tddnnf_times,
    #     x_label=tsdd_type,
    #     y_label=x3_nnf_type,
    #     out_path="tsdd_vs_proj_ce_query.pdf",
    # )

    # create_scatter_plot(
    #     x_data=x4_tsdd_times,
    #     y_data=x4_tddnnf_times,
    #     x_label=tsdd_type,
    #     y_label=x4_nnf_type,
    #     out_path="tsdd_vs_part_ce_query.pdf",
    # )

    # # Different T-d-DNNF against each other
    # create_scatter_plot(
    #     x_data=x1_tddnnf_times,
    #     y_data=x2_tddnnf_times,
    #     x_label=x1_nnf_type,
    #     y_label=x2_nnf_type,
    #     out_path="nnf_seq_vs_nnf_par_ce_query.pdf",
    # )

    # create_scatter_plot(
    #     x_data=x1_tddnnf_times,
    #     y_data=x3_tddnnf_times,
    #     x_label=x1_nnf_type,
    #     y_label=x3_nnf_type,
    #     out_path="nnf_seq_vs_nnf_proj_ce_query.pdf",
    # )

    # create_scatter_plot(
    #     x_data=x1_tddnnf_times,
    #     y_data=x4_tddnnf_times,
    #     x_label=x1_nnf_type,
    #     y_label=x4_nnf_type,
    #     out_path="nnf_seq_vs_nnf_part_ce_query.pdf",
    # )

    # create_scatter_plot(
    #     x_data=x2_tddnnf_times,
    #     y_data=x3_tddnnf_times,
    #     x_label=x2_nnf_type,
    #     y_label=x3_nnf_type,
    #     out_path="nnf_par_vs_nnf_proj_ce_query.pdf",
    # )

    # create_scatter_plot(
    #     x_data=x2_tddnnf_times,
    #     y_data=x4_tddnnf_times,
    #     x_label=x2_nnf_type,
    #     y_label=x4_nnf_type,
    #     out_path="nnf_par_vs_nnf_part_ce_query.pdf",
    # )

    # create_scatter_plot(
    #     x_data=x3_tddnnf_times,
    #     y_data=x4_tddnnf_times,
    #     x_label=x3_nnf_type,
    #     y_label=x4_nnf_type,
    #     out_path="nnf_proj_vs_nnf_part_ce_query.pdf",
    # )

    # # CACTUS with all experiments
    # create_cactus_plot(
    #     x1_times=x1_smt_times,
    #     x2_times=x1_tddnnf_times,
    #     x3_times=x2_tddnnf_times,
    #     x4_times=x3_tddnnf_times,
    #     x5_times=x4_tddnnf_times,
    #     x6_times=x4_tbdd_times,
    #     x7_times=x4_tsdd_times,
    #     x1_label=incr_smt_type,
    #     x2_label=x1_nnf_type,
    #     x3_label=x2_nnf_type,
    #     x4_label=x3_nnf_type,
    #     x5_label=x4_nnf_type,
    #     x6_label=tbdd_type,
    #     x7_label=tsdd_type,
    #     out_path="cactus_query_ce_all.pdf",
    # )

    #########################################################
    #########################################################

    # Plots for SAT paper
    # Incremental SMT vs T-d-DNNNF (Sequential)
    x1_path = "data/results/query_ce_rand_proj_all_incr_smt/data"
    x1_nnf_type = "T-d-DNNF"
    incr_smt_type = "Incremental SMT"

    # Incremental SMT vs T-d-DNNNF (Parallel)
    x2_path = "data/results/query_ce_rand_proj_all_NO_INCR_SMT/data"
    x2_nnf_type = "T-d-DNNF"
    smt_type = "SMT"

    (
        x1_incr_smt_times,
        x1_tddnnf_times,
        x1_tbdd_times,
        x1_tsdd_times,
    ) = get_all_logs(x1_path)

    (
        x2_smt_times,
        x2_tddnnf_times,
        x2_tbdd_times,
        x2_tsdd_times,
    ) = get_all_logs(x2_path)

    create_scatter_plot(
        x_data=x1_incr_smt_times,
        y_data=x1_tddnnf_times,
        x_label=incr_smt_type,
        y_label=x1_nnf_type,
        out_path="incr_smt_vs_nnf_ce_query.pdf",
    )

    create_scatter_plot(
        x_data=x2_smt_times,
        y_data=x2_tddnnf_times,
        x_label=smt_type,
        y_label=x2_nnf_type,
        out_path="smt_vs_nnf_ce_query.pdf",
    )
