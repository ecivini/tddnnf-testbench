import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import json
from matplotlib.ticker import MultipleLocator


TIMEOUT = 3600
CURRENT_TOTAL_TIME_KEY = "Total time"
CURRENT_TLEMMAS_TIME_KEY = "All-SMT computation time"


def get_current_results_times(
    err_file: str, paths: list[str], lemmas_err_file: str, lemmas_paths: list[str]
) -> tuple[dict, dict, int, int]:
    dd_times = {}

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
                    # dd_times[problem_name] = TIMEOUT
                    del tlemmas_times[problem_name]
                elif reason == "timeout" or "malloc" in reason or "calloc" in reason:
                    timeouts += 1
                    # dd_times[problem_name] = TIMEOUT
                    del tlemmas_times[problem_name]
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
                dd_times[problem_name] = data["T-SDD"][CURRENT_TOTAL_TIME_KEY]
                assert tlemmas_times[problem_name] != 0

                # if dd_times[problem_name] + tlemmas_times[problem_name] > TIMEOUT:
                #     del dd_times[problem_name]
                #     del tlemmas_times[problem_name]

    print("Compiled problems:", len(dd_times), len(tlemmas_times))
    print("Missing T-lemmas:", missing_tlemmas)
    print("DD Comp timeout:", timeouts)

    return tlemmas_times, dd_times, missing_tlemmas, timeouts


def get_current_tlemmas_times(err_file: str, paths: list[str]) -> dict:
    times = {}

    with open(err_file, "r") as f:
        errors = json.load(f)
        for problem, reason in errors.items():
            if reason == "timeout":
                key_name = problem.split(os.sep)[-1].replace(".smt2", "")
                times[key_name] = TIMEOUT
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


tlemmas_times_dict, dd_times_dict, missing_tlemmas, dd_timeouts = (
    ###########################################################################
    # FOR T-OBDDs
    # get_current_results_times(
    #     "data/results/tbdd_par_proj/tbdd_parallel_proj/errors.json",
    #     [
    #         "data/results/tbdd_par_proj/tbdd_parallel_proj/data/serialized_tdds/ldd_randgen/data",  # noqa
    #         "data/results/tbdd_par_proj/tbdd_parallel_proj/data/serialized_tdds/randgen/data",  # noqa
    #     ],
    #     "data/results/merged_all_tlemmas_projected/errors.json",
    #     [
    #         "data/results/merged_all_tlemmas_projected/ldd_randgen/data",
    #         "data/results/merged_all_tlemmas_projected/randgen/data",
    #     ],
    # )
    ###########################################################################
    # FOR T-SDDs
    get_current_results_times(
        "data/results/tsdd_proj/tsdd_parallel_proj/errors.json",
        [
            "data/results/tsdd_proj/tsdd_parallel_proj/data/serialized_tdds/ldd_randgen/data",  # noqa
            "data/results/tsdd_proj/tsdd_parallel_proj/data/serialized_tdds/randgen/data",  # noqa
        ],
        "data/results/merged_all_tlemmas_projected/errors.json",
        [
            "data/results/merged_all_tlemmas_projected/ldd_randgen/data",
            "data/results/merged_all_tlemmas_projected/randgen/data",
        ],
    )
)

# --- Generate mock data ---
lemma_time = []
bool_time = []
bool_to = []
for problem in tlemmas_times_dict:
    tlemmas_time = tlemmas_times_dict[problem]

    if problem in dd_times_dict:
        dd_time = dd_times_dict[problem]
        bool_to.append(False)
    else:
        dd_time = 0
        bool_to.append(True)

    lemma_time.append(tlemmas_time)
    bool_time.append(dd_time)

lemma_time = np.array(lemma_time)
bool_time = np.array(bool_time)
lemma_to = lemma_time >= TIMEOUT
bool_to = bool_time >= TIMEOUT

total_time = lemma_time + bool_time

# Sort by total time
order = np.argsort(total_time)
lemma_sorted = lemma_time[order]
bool_sorted = bool_time[order]
lemma_to_sorted = lemma_to[order]
bool_to_sorted = bool_to[order]
total_time_sorted = total_time[order]

# # --- Colorblind-safe palette (Wong 2011) ---
# # Orange for lemma, sky blue for boolean
# C_LEMMA = "#E69F00"  # orange
# C_BOOL = "#56B4E9"  # sky blue
# C_TO = "#D55E00"  # vermillion (timeout marker)

# # --- Plot ---
# fig, ax = plt.subplots(figsize=(5, 5))

# N = 450  # len(lemma_time)
# missing = N - len(lemma_time)
# print(f"Filling {missing} gaps")
# for _ in range(missing):
#     lemma_sorted = np.append(lemma_sorted, [0])
#     bool_sorted = np.append(bool_sorted, [0])
# print("Sorted lemmas:", len(lemma_sorted))

# x = np.arange(N)

# ax.bar(x, lemma_sorted, width=1.0, color=C_LEMMA, linewidth=0, label="Lemma generation")
# ax.bar(
#     x,
#     bool_sorted,
#     width=1.0,
#     color=C_BOOL,
#     linewidth=0,
#     bottom=lemma_sorted,
#     label="Boolean compilation",
# )

# # Timeout ceiling
# # ax.axhline(TIMEOUT, color="#444444", linewidth=1.0, linestyle="--", zorder=5)
# # ax.text(
# #     N * 0.01, TIMEOUT * 1.04, "1 h timeout", fontsize=8, color="#444444", va="bottom"
# # )

# # Vertical line where lemma timeouts begin
# # first_to = np.where(total_time_sorted >= TIMEOUT)[0][0]
# # if total_time_sorted[first_to]:
# #     ax.axvline(first_to, color=C_TO, linewidth=1.2, linestyle=":", zorder=5)
# #     n_lto = len(np.where(total_time_sorted > TIMEOUT))
# #     ax.text(
# #         first_to + 4,
# #         TIMEOUT * 0.55,
# #         f"lemma timeout\n({n_lto} instances)",
# #         fontsize=8,
# #         color=C_TO,
# #         va="center",
# #     )

# # Y axis: log scale
# ax.set_yscale("log")
# ax.set_ylim(0.05, 50000)
# # ax.yaxis.set_major_formatter(
# #     plt.FuncFormatter(
# #         lambda t, _: {1: "1s", 10: "10s", 60: "1m", 600: "10m", 3600: "1h"}.get(
# #             int(t), ""
# #         )
# #     )
# # )
# # ax.set_yticks([1, 10, 60, 600, 3600])

# # X axis
# ax.set_xlim(0, N)
# ax.set_xlabel("Compiled problems", fontsize=24)
# ax.set_ylabel("Time (s)", fontsize=24)

# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)

# # Legend
# patches = [
#     mpatches.Patch(color=C_LEMMA, label=f"Lemma enumeration"),
#     mpatches.Patch(color=C_BOOL, label=f"Boolean compilation"),
# ]
# ax.legend(handles=patches, loc="upper left", fontsize=14, framealpha=0.9)

# # ax.set_title(
# #     "Compilation time breakdown across benchmark instances", fontsize=12, pad=10
# # )

# # Light grid
# ax.yaxis.grid(True, which="major", color="#dddddd", linewidth=0.7, zorder=0)
# ax.set_axisbelow(True)
# ax.spines[["top", "right"]].set_visible(False)

# plt.tight_layout()
# plt.savefig("cactus_tlemmas_and_tsdd_comp_time.pdf", bbox_inches="tight", pad_inches=0)
# print("Saved.")

##########################################################################
##########################################################################
##########################################################################


def create_cactus_plot(
    x1: list,
    x2: list,
    x3: list,
    x1_label: str,
    x2_label: str,
    x3_label: str,
    out_path: str = "cactus.pdf",
    legend_path: str = "legend.pdf",
) -> None:
    x1_arr = np.arange(1, len(x1) + 1)
    x2_arr = np.arange(1, len(x2) + 1)
    x3_arr = np.arange(1, len(x3) + 1)

    fig, ax = plt.subplots(figsize=(5, 5))

    (l1,) = ax.plot(x1_arr, x1, label=x1_label)
    (l2,) = ax.plot(x2_arr, x2, label=x2_label)
    (l3,) = ax.plot(x3_arr, x3, label=x3_label)

    ax.set_yscale("log")

    ax.set_xlabel("Compiled problems", fontsize=24)
    ax.set_ylabel("Time (s)", fontsize=24)

    ax.tick_params(axis="both", labelsize=18)

    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.set_xlim(1e-2, 450)
    ax.set_ylim(1e-2, TIMEOUT * 2.5)

    ax.grid(True)

    # ---- SAVE PLOT WITHOUT LEGEND ----
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)

    # ---- CREATE SEPARATE LEGEND ----
    legend_fig = plt.figure(figsize=(4, 1))
    legend_fig.legend(
        handles=[l1, l2, l3],
        labels=[x1_label, x2_label, x3_label],
        loc="center",
        ncol=3,
        fontsize=14,
        frameon=False,
    )

    legend_fig.savefig(legend_path, bbox_inches="tight", pad_inches=0)


create_cactus_plot(
    x1=list(lemma_sorted),
    x2=list(bool_sorted),
    x3=list(total_time_sorted),
    x1_label="Lemma enumeration",
    x2_label="Bool compilation",
    x3_label="Total time",
    out_path="cactus_tlemmas_and_tsdd_comp_time.pdf",
)
