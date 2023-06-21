import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
import scipy.stats as st
import researchpy as rp


def load_data(data_path=None, predictions_path=None, limit="", data_filter=""):
    print("starting to load data")
    full_data = pd.read_csv(data_path)
    predictions = pd.read_json(r"{}".format(predictions_path))
    full_data["predictions"] = predictions
    print("returning loaded data")
    return full_data


def split_data_in_bins(data, nr_of_bins, bin_labels):
    data.loc[:, "bin"], bin_edges = pd.qcut(data["predictions"],
                                            q=nr_of_bins,
                                            labels=bin_labels,
                                            retbins=True)
    bin_thresholds = pd.DataFrame(bin_edges, columns=["bin_thresholds"])
    # the bin with name "0" has the limits bin_thresholds[0] and bin_thresholds[1]

    if bin_labels == False:
        bin_label_names = np.sort(np.array(data.bin.unique()))
    else:
        bin_label_names = np.sort(np.array(bin_labels))

    return bin_thresholds, bin_label_names


def analyze_bins_by_group(bin_data, group, bin_label_names, add_bootstrap_solution, print_details, ttest):
    average_diffs = {}
    average_diffs_all_data = {}
    for bin_name in bin_label_names:
        bin_rows = bin_data[bin_data["bin"] == bin_name]
        summary_statistics = bin_rows["premium_claims_diff"].describe()
        summary_statistics["sem"] = bin_rows["premium_claims_diff"].sem()
        np.random.seed(0)
        conf_interv_95 = st.norm.interval(
            alpha=0.95, loc=summary_statistics["mean"], scale=summary_statistics["sem"])
        summary_statistics["CI95%"] = conf_interv_95
        summary_statistics["Freq mean"] = bin_rows["Freq"].mean()

        if add_bootstrap_solution:
            summary_statistics["bootstrap_solution"] = my_bootstrap(
                bin_rows["premium_claims_diff"])

        if ttest:
            average_diffs_all_data[bin_name] = bin_rows["premium_claims_diff"]

        average_diffs[bin_name] = summary_statistics

    return average_diffs, average_diffs_all_data


def plot_bin_data_by_group(average_diffs_by_group, errorbars, add_bootstrap_solution, ylim, fig_size, group_by):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'lightcoral', 'burlywood', 'lime', 'lightsteelblue', 'navy',
              'chocolate', 'aquamarine', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'lightcoral', 'burlywood', 'lime', 'lightsteelblue', 'navy', 'chocolate', 'aquamarine']
    counter = 0

    if fig_size == None:
        fig_size = (10, 8)
    plt.figure(figsize=fig_size, dpi=100)
    if ylim is not None:
        plt.ylim(ylim)

    xlim = len(average_diffs_by_group[list(
        average_diffs_by_group.keys())[0]]) - 0.9

    # define x-positions for error bars
    errorbar_x_pos = []
    nr_of_groups = len(average_diffs_by_group.keys())
    for x in range(1, nr_of_groups+1):
        errorbar_x_pos.append((x - (nr_of_groups+1) / 2) * 0.03)

    for group in average_diffs_by_group.keys():
        group_name = "group " + str(group)

        label_mean = group_name
        label_median = group_name + " median"
        label_max = group_name + " max"
        label_min = group_name + " min"

        plt.hlines(y=0, xmin=-0.1, xmax=xlim,
                   linestyles="dashed", color="lightgrey")

        if errorbars == "bootstrap-95%CI":
            print(
                "We plot the means of our sample +/- the 95% confidence interval of the bootstrapped sample means [ group =", str(group), "].")
            plt.errorbar([k + errorbar_x_pos[counter] for k in average_diffs_by_group[group].keys()], [v["mean"] for k, v in average_diffs_by_group[group].items()], [[v["mean"] - v["bootstrap_solution"]["95%CI_of_bootstrapped_means"][0] for k, v in average_diffs_by_group[group].items()],
                         [v["bootstrap_solution"]["95%CI_of_bootstrapped_means"][1] - v["mean"] for k, v in average_diffs_by_group[group].items()]], linestyle=':', marker='o', markersize=6, capsize=3, alpha=0.7, color=colors[counter], label=label_mean)
        elif errorbars == "bootstrap-2SEM":
            print(
                "We plot the means of our sample +/- 2 times the std of the bootstrapped sample means [ group =", str(group), "].")
            plt.errorbar([k + errorbar_x_pos[counter] for k in average_diffs_by_group[group].keys()], [v["mean"] for k, v in average_diffs_by_group[group].items()], [2 * v["bootstrap_solution"]
                         ["std_of_bootstrapped_means"] for k, v in average_diffs_by_group[group].items()], linestyle=':', marker='o', markersize=6, capsize=3, alpha=0.7, color=colors[counter], label=label_mean)
        elif errorbars == "bootstrap-SEM":
            print(
                "We plot the means of our sample +/- the std of the bootstrapped sample means [ group =", str(group), "].")
            plt.errorbar([k + errorbar_x_pos[counter] for k in average_diffs_by_group[group].keys()], [v["mean"] for k, v in average_diffs_by_group[group].items()], [v["bootstrap_solution"]
                         ["std_of_bootstrapped_means"] for k, v in average_diffs_by_group[group].items()], linestyle=':', marker='o', markersize=6, capsize=3, alpha=0.7, color=colors[counter], label=label_mean)
        elif errorbars == "2SEM":
            print(
                "We plot the means of our sample +/- 2 times the estimated SEM [ group =", str(group), "].")
            plt.errorbar([k + errorbar_x_pos[counter] for k in average_diffs_by_group[group].keys()], [v["mean"] for k, v in average_diffs_by_group[group].items()], [2 * v["sem"]
                         for k, v in average_diffs_by_group[group].items()], linestyle=':', marker='o', markersize=6, capsize=3, alpha=0.7, color=colors[counter], label=label_mean)
        elif errorbars == "SEM":
            print(
                "We plot the means of our sample +/- the estimated SEM [ group =", str(group), "].")
            plt.errorbar([k + errorbar_x_pos[counter] for k in average_diffs_by_group[group].keys()], [v["mean"] for k, v in average_diffs_by_group[group].items()], [v["sem"]
                         for k, v in average_diffs_by_group[group].items()], linestyle=':', marker='o', markersize=6, capsize=3, alpha=0.7, color=colors[counter], label=label_mean)
        else:
            print(
                "We plot the means of our sample +/- the estimated 95% conficence interval of the sample mean [ group =", str(group), "].")
            plt.errorbar([k + errorbar_x_pos[counter] for k in average_diffs_by_group[group].keys()], [v["mean"] for k, v in average_diffs_by_group[group].items()], [[v["mean"] - v["CI95%"][0] for k, v in average_diffs_by_group[group].items()],
                         [v["CI95%"][1] - v["mean"] for k, v in average_diffs_by_group[group].items()]], linestyle=':', marker='o', markersize=6, capsize=3, alpha=0.7, color=colors[counter], label=label_mean)

        counter += 1
    plt.xlabel("Bins based on pure premium")
    if errorbars == "SEM":
        plt.ylabel("Mean premium-freq-difference (by group) +/- SEM")
    else:
        plt.ylabel("Mean premium-freq-difference (by group) with 95% CI")
    plt.legend()


def my_bootstrap(data):
    np.random.seed(42)
    alpha = 0.95
    p_lower = ((1.0-alpha)/2.0) * 100
    p_upper = (alpha+((1.0-alpha)/2.0)) * 100
    my_dict = {}
    selection_with_replacement_means = pd.Series([np.random.choice(
        data, replace=True, size=len(data)).mean() for _ in range(1000)])
    my_dict["mean_of_bootstrapped_means"] = selection_with_replacement_means.mean()
    my_dict["std_of_bootstrapped_means"] = selection_with_replacement_means.std()
    my_dict["min_of_bootstrapped_means"] = selection_with_replacement_means.min()
    my_dict["max_of_bootstrapped_means"] = selection_with_replacement_means.max()
    ordered_means = selection_with_replacement_means.sort_values()
    lower = np.percentile(ordered_means, p_lower)
    upper = np.percentile(ordered_means, p_upper)
    my_dict["95%CI_of_bootstrapped_means"] = (lower, upper)

    return my_dict


def analyze_freq_by_premiumBins_and_group(full_data, group_by=None, nr_of_bins=10, subset=None, customized_group_names=None, errorbars="CI95%", add_bootstrap_solution=False, print_details=False, ttest=False, ylim=None, fig_size=None, bin_labels=False):
    # delete data variable from previous calculations to avoid MemoryError
    # del data
    if subset is None:
        data = full_data.copy()
    else:
        data = full_data[subset].copy()

    average_diffs_by_group = {}
    average_diffs_all_data_by_group = {}

    bin_thresholds, bin_label_names = split_data_in_bins(
        data, nr_of_bins, bin_labels)

    if customized_group_names is None:
        customized_group_names = data[group_by].unique()

    for group in customized_group_names:
        if not pd.isnull(group):
            bin_data = data[data[group_by] == group].copy()
            average_diffs_by_group[group], average_diffs_all_data_by_group[group] = analyze_bins_by_group(
                bin_data, group, bin_label_names, add_bootstrap_solution, print_details, ttest)

    plot_bin_data_by_group(average_diffs_by_group, errorbars,
                           add_bootstrap_solution, ylim, fig_size, group_by)

    ttest_results = {}

    if ttest:
        if len(customized_group_names) != 2:
            print("2 groups expected but", str(
                len(customized_group_names)), "groups given.")
        else:
            for bin in bin_label_names:
                summary, results = rp.ttest(group1=average_diffs_all_data_by_group[customized_group_names[0]][bin], group1_name=customized_group_names[0],
                                            group2=average_diffs_all_data_by_group[customized_group_names[1]][bin], group2_name=customized_group_names[1])
                ind_t_test = st.ttest_ind(
                    average_diffs_all_data_by_group[customized_group_names[0]][bin], average_diffs_all_data_by_group[customized_group_names[1]][bin])
                t_value = ind_t_test[0]
                p_value = ind_t_test[1]
                ttest_results[bin] = {
                    "summary": summary, "result": results, "t-value": t_value, "p-value": p_value}

    return bin_thresholds, average_diffs_by_group, ttest_results
