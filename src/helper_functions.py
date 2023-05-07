# helper_functions.py
import os
import time
import numpy as np
import pandas as pd
import hashlib
import json
import scipy.stats as stats
import math

import matplotlib.pyplot as plt
#from plotnine import ggplot, aes, geom_line, geom_ribbon, labs, theme, element_text, scale_color_manual

import eval_algos


def generate_feature_hash(features):
    """
    Takes a list of feature names as input and returns a unique
    hash value that represents the combination of features. 

    Parameters:
    features (list): A list of feature names (strings) to be hashed.

    Returns:
    str: A short, unique hash value representing the input list of features.
    """
    sorted_features = sorted(features)
    concatenated_features = ''.join(sorted_features)
    return hashlib.sha1(concatenated_features.encode('utf-8')).hexdigest()[:8]

def measure_runtime(func):
    """
    Wrapper function to measure the runtime of a function.

    :param func: The function to be measured.

    :return: The wrapper function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' took {end_time - start_time:.5f} seconds to execute.")
        return result
    return wrapper


def sample_and_split(df, frac, seed = None, n = None):
    """
    Randomly subsamples a dataframe and returns the features and labels as numpy arrays.

    :param df: The dataframe to be subsampled.
    :param frac: The fraction of the dataframe to be subsampled.
    :param n: The number of samples to be drawn.
    :param seed: The seed for the random number generator.

    :return: A tuple of numpy arrays containing the features and labels.
    """
    if frac is not None and n is not None:
        raise ValueError("Use either 'frac' or 'n', but not both.")

    if n is None:
        subsample = df.sample(frac = frac, random_state = seed)
    else:
        subsample = df.sample(n, random_state = seed)

    return np.array(subsample.iloc[:,1:]), np.array(subsample.iloc[:,0])


def get_random_baseline(n=1000000):
    """
    Get the random baseline for a list of json files.

    :param json_files: A list of json files.

    :return: A dictionary containing the random baseline for each json file.
    """
    gt = eval_algos.get_ground_truths_ordered()
    gt_first_gen = np.array(gt[1].index).copy()
    k_list = np.arange(1, 101)

    random_scores = np.zeros((n, len(k_list)))

    for i in range(n):
        np.random.shuffle(gt_first_gen)
        for idx, k in enumerate(k_list):
            random_order = gt_first_gen.copy()
            np.random.shuffle(random_order)
            random_scores[i, idx] = eval_algos.jaccard_k(random_order[:k], gt_first_gen[:k], k)

    random_mean_list = np.mean(random_scores, axis=0)
    random_se_list = np.std(random_scores, axis=0, ddof=1) / np.sqrt(n)

    random_ci_low_list = random_mean_list - 1.96 * random_se_list
    random_ci_high_list = random_mean_list + 1.96 * random_se_list

    return random_mean_list, random_ci_low_list, random_ci_high_list


def get_true_baseline(T=100):
    """
    Replaces the old get_random_baseline with an analytical solution.

    :param T: The number of features we are ranking.

    :return: A numpy array containing the expected value for a random permutaiton for each k.
    """
    def probability(i, j, t, T):
        return math.comb(t, i) * math.comb(T - t, j) / math.comb(T, t)

    def expected_score(t, T):
        e_t = 0
        for i in range(t + 1):
            j = t - i
            e_t += probability(i, j, t, T) * (i / (2 * t - i))
        return e_t
    
    baseline = [expected_score(t, T) for t in range(1, T + 1)]

    return np.array(baseline)


def area_under_the_curve(points):
    """
    Computes area under the curve (i, points[i]), i.e., assumes that x-values are at distance 1.
    :param points:
    :return:
    """
    n = len(points) - 1
    a = 0.0
    for i in range(n):
        a += (points[i] + points[i + 1]) / 2
    return a / n


def plot_ranking_results(json_files_list, random_baseline, plot_type='both'):
    k_list = range(1, 101)
    categories = ['first_gen', 'singles']
    if plot_type in categories:
        categories = [plot_type]

    ncols = 1 if plot_type != 'both' else 2
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(12 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    colorblind_colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#999999']
    linestyles = ['-', '--', ':', '-.', (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (5, 1))]

    for ax, category in zip(axes, categories):
        for idx, json_files in enumerate(json_files_list):
            results = []
            for file in json_files:
                with open(file, 'r') as f:
                    data = json.load(f)
                    rank_algo = data["rank_algo"]
                    subsampling_proportion = data["subsampling_proportion"]
                    ranking_results = data["evaluations"][category][0]
                    results.append(ranking_results)

            mean_results = np.mean(results, axis=0)
            std_results = np.std(results, axis=0)

            ax.plot(mean_results, label=f'{rank_algo} ({subsampling_proportion})',
                    color=colorblind_colors[idx % len(colorblind_colors)], linestyle=linestyles[idx % len(linestyles)])
            ax.fill_between(range(100), mean_results - std_results, mean_results + std_results, alpha=0.2,
                            color=colorblind_colors[idx % len(colorblind_colors)])

        if isinstance(random_baseline[0], tuple):
            random_mean_list, random_ci_low_list, random_ci_high_list = random_baseline
        else:
            random_mean_list = random_baseline
            random_ci_low_list = random_baseline
            random_ci_high_list = random_baseline

        ax.plot(random_mean_list, label='random ranking', color='grey')
        x = [k - 1 for k in k_list]
        ax.fill_between(x, random_ci_low_list, random_ci_high_list, alpha=0.2, color='grey')

        ax.set_title(category.replace('_', ' ').title())
        ax.set_xlabel('k')
        ax.set_ylabel('Jaccard score')

        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)

        ax.xaxis.set_ticks(np.arange(0, 101, 20))
        ax.xaxis.set_ticklabels(['1', '20', '40', '60', '80', '100'])

        ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)

    if ncols == 2:
        axes[-1].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0., prop={'size': 15})
    else:
        axes[0].legend(loc='lower right', prop={'size': 15})

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()


def plot_ranking_results_one_run(json_files, random_baseline, plot_type='both'):
    """
    Plots the ranking results for a list of json files.
    Used to compare different ranking algorithms or subsampling proportions in the same plot.

    :param json_files: list of json files containing the ranking results
    :param random_baseline: random baseline obtained from either get_random_baseline (numerical approx.)
                            or get_true_baseline (analytical solution)
    :param plot_type: 'first_gen', 'singles' or 'both'
    :return:
    """

    k_list = range(1, 101)
    categories = ['first_gen', 'singles']
    if plot_type in categories:
        categories = [plot_type]

    ncols = 1 if plot_type != 'both' else 2
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(12 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    colorblind_colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#999999']
    linestyles = ['-', '--', ':', '-.', (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (5, 1))]

    for ax, category in zip(axes, categories):
        for idx, file in enumerate(json_files):
            with open(file, 'r') as f:
                data = json.load(f)
                rank_algo = data["rank_algo"]
                subsampling_proportion = data["subsampling_proportion"]
                ranking_results = data["evaluations"][category][0]

                ax.plot(ranking_results, label=f'{rank_algo} ({subsampling_proportion})',
                        color=colorblind_colors[idx % len(colorblind_colors)], linestyle=linestyles[idx % len(linestyles)])

        if isinstance(random_baseline[0], tuple):
            random_mean_list, random_ci_low_list, random_ci_high_list = random_baseline
        else:
            random_mean_list = random_baseline
            random_ci_low_list = random_baseline
            random_ci_high_list = random_baseline

        ax.plot(random_mean_list, label='random ranking', color='grey')
        x = [k - 1 for k in k_list]
        ax.fill_between(x, random_ci_low_list, random_ci_high_list, alpha=0.2, color='grey')

        ax.set_title(category.replace('_', ' ').title())
        ax.set_xlabel('k')
        ax.set_ylabel('Jaccard score')

        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)

        ax.xaxis.set_ticks(np.arange(0, 101, 20))
        ax.xaxis.set_ticklabels(['1', '20', '40', '60', '80', '100'])

        ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)

    if ncols == 2:
        axes[-1].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0., prop={'size': 15})
    else:
        axes[0].legend(loc='lower right', prop={'size': 15})

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()


def plot_ranking_results_same_algo_one_run(json_files, random_baseline, plot_type='both'):
    """
    Plots the ranking results for a list of json files.
    Used to compare the performance of the same ranking algorithm using different subsampling proportions.

    :param json_files: list of json files containing the ranking results
    :param random_baseline: random baseline obtained from either get_random_baseline (numerical approx.) or
                            get_true_baseline (analytical solution)
    :param plot_type: 'first_gen', 'singles' or 'both'
    :return:
    """

    k_list = range(1, 101)
    categories = ['first_gen', 'singles']
    if plot_type in categories:
        categories = [plot_type]

    ncols = 1 if plot_type != 'both' else 2
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(8 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    colorblind_colors = ['#E69F00', '#56B4E9', '#009E73', '#D55E00', '#F0E442', '#0072B2', '#CC79A7']
    linestyles = ['-', '--', ':', '-.', (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (5, 1))]

    rank_algo_title = None

    for ax, category in zip(axes, categories):
        for idx, file in enumerate(json_files):
            with open(file, 'r') as f:
                data = json.load(f)
                rank_algo = data["rank_algo"]
                subsampling_proportion = data["subsampling_proportion"]
                exec_time = data["exec_time"]
                ranking_results = data["evaluations"][category][0]

                if rank_algo_title is None:
                    rank_algo_title = rank_algo

                ax.plot(ranking_results, label=f'Subsample: {subsampling_proportion}, Time: {exec_time:.2f}s',
                        color=colorblind_colors[idx % len(colorblind_colors)], linestyle=linestyles[idx % len(linestyles)])

        if isinstance(random_baseline[0], tuple):
            random_mean_list, random_ci_low_list, random_ci_high_list = random_baseline
        else:
            random_mean_list = random_baseline
            random_ci_low_list = random_baseline
            random_ci_high_list = random_baseline

        ax.plot(random_mean_list, label='random ranking', color='grey')
        x = [k - 1 for k in k_list]
        ax.fill_between(x, random_ci_low_list, random_ci_high_list, alpha=0.2, color='grey')

        ax.set_title(category.replace('_', ' ').title())
        ax.set_xlabel('k')
        ax.set_ylabel('Jaccard score')

        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)

        ax.xaxis.set_ticks(np.arange(0, 101, 20))
        ax.xaxis.set_ticklabels(['1', '20', '40', '60', '80', '100'])

        ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)

    if ncols == 2:
        axes[-1].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0., prop={'size': 15})
    else:
        axes[0].legend(loc='lower right', prop={'size': 15})

    if rank_algo_title is not None:
        fig.suptitle(rank_algo_title, fontsize=18, y=1.05)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    random_baseline = get_true_baseline()

    '''json_files = ['results/mutual_info_score_seed0_sub0.01_features-all.json',
                  'results/mutual_info_score_seed0_sub0.001_features-all.json',
                  'results/chi2_score_seed0_sub0.01_features-all.json',
                  'results/chi2_score_seed0_sub0.001_features-all.json']

    json_files = ['results/mutual_info_score_seed0_sub0.1_features-all.json',
                  'results/mutual_info_score_seed0_sub0.1_features-all_leon_fail.json']


    plot_ranking_results(json_files, random_baseline)
    #plot_ranking_results_same_algo_one_run(json_files)'''

    # Define the folder paths
    anova_f_0_1_folder = "results/anova_f_0.1_batch"
    anova_f_0_01_folder = "results/anova_f_0.01_batch"

    # Get the JSON file paths for each folder
    anova_f_0_1_files = [os.path.join(anova_f_0_1_folder, f"anova_f_score_seed{i}_sub0.1_features-all.json") for i in
                         range(100)]
    anova_f_0_01_files = [os.path.join(anova_f_0_01_folder, f"anova_f_score_seed{i}_sub0.01_features-all.json") for i in
                          range(100)]

    # Add more file lists for other algorithms here, following the same pattern

    json_files_list = [
        anova_f_0_01_files,
        anova_f_0_1_files,
        # Add other file lists here
    ]

    plot_ranking_results(json_files_list, random_baseline, plot_type='both')
