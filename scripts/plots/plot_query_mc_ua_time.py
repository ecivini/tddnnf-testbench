import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


DEFAULT_ALLSMT_TIMEOUT = 600


def get_results(base_path: str) -> tuple:
    smt_times = []
    smt_counts = []

    tddnnf_times = []
    tddnnf_counts = []

    tbdd_times = []
    tsdd_times = []

    allsmt_timeouts = 0
    tddnnf_timeouts = 0
    tbdd_timeouts = 0
    tsdd_timeouts = 0
    for root, _, files in os.walk(base_path):
        for file in files:
            if file != "logs.json":
                continue

            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                data = json.load(f)

            for cube in data["MC"]:
                is_allsmt_timeout = data["MC"][cube]["AllSMT timeout"]
                if is_allsmt_timeout:
                    smt_times.append(DEFAULT_ALLSMT_TIMEOUT)
                    smt_counts.append(-1)
                    allsmt_timeouts += 1
                else:
                    smt_times.append(data["MC"][cube]["AllSMT time"])
                    smt_counts.append(data["MC"][cube]["AllSMT count"])

                if data["MC"][cube]["d-DNNF time"] < DEFAULT_ALLSMT_TIMEOUT:
                    tddnnf_times.append(data["MC"][cube]["d-DNNF time"])
                    tddnnf_counts.append(data["MC"][cube]["d-DNNF count"])
                else:
                    tddnnf_times.append(DEFAULT_ALLSMT_TIMEOUT)
                    tddnnf_counts.append(-1)
                    tddnnf_timeouts += 1

                if data["MC"][cube]["T-BDD time"] is not None:
                    tbdd_times.append(data["MC"][cube]["T-BDD time"])
                    # tbdd_counts.append(data["MC"][cube]["d-DNNF count"])
                else:
                    # tbdd_times.append(DEFAULT_ALLSMT_TIMEOUT)
                    tbdd_timeouts += 1

                if data["MC"][cube]["T-SDD time"] is not None:
                    tsdd_times.append(data["MC"][cube]["T-SDD time"])
                    # tbdd_counts.append(data["MC"][cube]["d-DNNF count"])
                else:
                    # tbdd_times.append(DEFAULT_ALLSMT_TIMEOUT)
                    tsdd_timeouts += 1

    print(f"Number of AllSMT timeouts: {allsmt_timeouts} / {len(smt_times)}")
    print(f"Number of T-d-DNNF timeouts: {tddnnf_timeouts} / {len(tddnnf_times)}")
    # print(f"Number of T-BDD queries: {tbdd_timeouts} / {len(tbdd_times)}")
    # print(f"Number of T-SDD queries: {tsdd_timeouts} / {len(tsdd_times)}")

    return (smt_times, smt_counts, tddnnf_times, tddnnf_counts)


def create_scatter_plot(
    x_data: list,
    y_data: list,
    x_label: str,
    y_label: str,
    out_path: str = "scatter.pdf",
    log_scale: bool = False,
    alpha: float = 1.0,
    upperbound: float = 0.0,
    linthresh: float | None = None,
):
    assert len(x_data) == len(y_data)

    print(min(x_data), min(y_data))

    within_upperbound_x = []
    within_upperbound_y = []
    if upperbound == 0.0:
        within_upperbound_x = x_data
        within_upperbound_y = y_data
        timeout = DEFAULT_ALLSMT_TIMEOUT
    else:
        for i in range(len(x_data)):
            if x_data[i] > upperbound and y_data[i] > upperbound:
                continue
            within_upperbound_x.append(x_data[i])
            within_upperbound_y.append(y_data[i])

        timeout = max(max(within_upperbound_x), max(within_upperbound_y))

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 5))

    # Scatter plot
    ax.scatter(
        x=within_upperbound_x,
        y=within_upperbound_y,
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

    # Timeout lines (dashed)
    ax.axvline(
        timeout,
        linestyle="--",
        color="gray",
    )
    ax.axhline(
        timeout,
        linestyle="--",
        color="gray",
    )

    # Set symlog scale
    if log_scale:
        if linthresh is not None:
            ax.set_xscale("symlog", linthresh=linthresh)
            ax.set_yscale("symlog", linthresh=linthresh)
        else:
            ax.set_xscale("symlog")
            ax.set_yscale("symlog")
    ax.set_aspect("equal")

    # Set limits
    ax.set_xlim(left=1e-20, right=timeout * 1.1)
    ax.set_ylim(bottom=1e-20, top=timeout * 1.1)

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


def create_cactus_plot(
    x1: list,  # AllSMT
    x2: list,  # T-d-DNNF
    x1_label: str,
    x2_label: str,
    out_path: str = "cactus.pdf",
):
    x1.sort()
    x2.sort()

    x1_arr = np.arange(1, len(x1) + 1)
    x2_arr = np.arange(1, len(x2) + 1)

    # Plot
    plt.figure(figsize=(5, 5))
    plt.plot(x1_arr, x1, label=x1_label, marker="o", markersize=2)
    plt.plot(x2_arr, x2, label=x2_label, marker="o", markersize=2, color="green")

    plt.xlabel("Queried problems", fontsize=24)
    plt.ylabel("Time (s)", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=14)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(150))
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


if __name__ == "__main__":
    ###########################################################################

    # RAND PROBLEMS
    # AllSMT vs T-d-DNNF (Proj)
    x1_logs = "data/results/final_query_mcua_25s_to_3600s_with_dds/query_mcua_rand_proj_between_25s_100s_on_proj_10s_timeout_per_query_overall_2h_timeout_with_dds"
    qui_logs = "data/results/mcua_1s_25_qui/query_mcua_rand_proj_between_1s_25s_on_proj_10s_timeout_per_query_overall_2h_timeout_qui"
    quo_logs = "data/results/mcua_1s_25_quo"
    qua_logs = "data/results/mcua_1s_25_qua"

    ###########################################################################
    # Load data
    (x1_smt_times, x1_smt_counts, x1_tddnnf_times, x1_tddnnf_counts) = get_results(
        x1_logs
    )
    (qui_smt_times, qui_smt_counts, qui_tddnnf_times, qui_tddnnf_counts) = get_results(
        qui_logs
    )
    (quo_smt_times, quo_smt_counts, quo_tddnnf_times, quo_tddnnf_counts) = get_results(
        quo_logs
    )
    (qua_smt_times, qua_smt_counts, qua_tddnnf_times, qua_tddnnf_counts) = get_results(
        qua_logs
    )

    smt_times = x1_smt_times + qui_smt_times + quo_smt_times + qua_smt_times
    tddnnf_times = (
        x1_tddnnf_times + qui_tddnnf_times + quo_tddnnf_times + qua_tddnnf_times
    )

    ###########################################################################

    # Scatter plots vs AllSMT
    create_scatter_plot(
        x_data=smt_times,
        y_data=tddnnf_times,
        x_label="AllSMT",
        y_label="T-red d-DNNF",
        out_path="query_mc_allsmt_vs_dnnf_lin_scale.pdf",
        log_scale=False,
        alpha=0.4,
    )

    create_scatter_plot(
        x_data=smt_times,
        y_data=tddnnf_times,
        x_label="AllSMT",
        y_label="T-red d-DNNF",
        out_path="query_mc_allsmt_vs_dnnf_log_scale.pdf",
        log_scale=True,
        alpha=0.4,
    )

    create_scatter_plot(
        x_data=smt_times,
        y_data=tddnnf_times,
        x_label="AllSMT",
        y_label="T-red d-DNNF",
        out_path="query_mc_allsmt_vs_dnnf_log_scale_lin_until_0_1s.pdf",
        log_scale=True,
        alpha=0.4,
        linthresh=0.1,
    )

    create_cactus_plot(
        x1=smt_times,
        x2=tddnnf_times,
        x1_label="AllSMT",
        x2_label="T-red d-DNNF",
        out_path="cactus_query_mc_allsmt_vs_dnnf.pdf",
    )
