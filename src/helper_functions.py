# helper_functions.py

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

def get_true_baseline(T):
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

def plot_ranking_results(json_files, random_baseline):
    k_list = range(1, 101)

    fig, ax = plt.subplots(figsize=(12, 6))

    colorblind_colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#999999']
    linestyles = ['-', '--', ':', '-.', (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (5, 1))]

    for idx, file in enumerate(json_files):
        with open(file, 'r') as f:
            data = json.load(f)
            rank_algo = data["rank_algo"]
            subsampling_proportion = data["subsampling_proportion"]
            ranking_results = data["evaluations"]["first_gen"][0]

            ax.plot(ranking_results, label=f'{rank_algo} ({subsampling_proportion})',
                    color=colorblind_colors[idx % len(colorblind_colors)], linestyle=linestyles[idx % len(linestyles)])

    random_mean_list, random_ci_low_list, random_ci_high_list = random_baseline

    ax.plot(random_mean_list, label='random ranking', color='grey')
    x = [k - 1 for k in k_list]
    ax.fill_between(x, random_ci_low_list, random_ci_high_list, alpha=0.2, color='grey')

    ax.set_xlabel('k')
    ax.set_ylabel('Jaccard score')
    #ax.legend()
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0., prop={'size': 15})

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)

    # add grid
    ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 1])
    #plt.tight_layout()
    #plt.savefig('img/report_top_k_scores_first_gen.png', dpi=400, bbox_inches='tight')
    plt.show()


def plot_ranking_results_same_algo(json_files, random_baseline):
    k_list = range(1, 101)

    fig, ax = plt.subplots(figsize=(8, 6))

    colorblind_colors = ['#E69F00', '#56B4E9', '#009E73', '#D55E00', '#F0E442', '#0072B2', '#CC79A7']
    linestyles = ['-', '--', ':', '-.', (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (5, 1))]

    rank_algo_title = None

    for idx, file in enumerate(json_files):
        with open(file, 'r') as f:
            data = json.load(f)
            rank_algo = data["rank_algo"]
            subsampling_proportion = data["subsampling_proportion"]
            exec_time = data["exec_time"]
            ranking_results = data["evaluations"]["first_gen"][0]

            if rank_algo_title is None:
                rank_algo_title = rank_algo

            ax.plot(ranking_results, label=f'Subsample: {subsampling_proportion}, Time: {exec_time:.2f}s',
                    color=colorblind_colors[idx % len(colorblind_colors)], linestyle=linestyles[idx % len(linestyles)])

    random_mean_list, random_ci_low_list, random_ci_high_list = random_baseline

    ax.plot(random_mean_list, label='random ranking', color='grey')
    x = [k - 1 for k in k_list]
    ax.fill_between(x, random_ci_low_list, random_ci_high_list, alpha=0.2, color='grey')

    ax.set_xlabel('k')
    ax.set_ylabel('Jaccard score')
    ax.legend()

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)

    # legend font size
    ax.legend(loc='lower right', prop={'size': 15})

    # add grid
    ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)

    if rank_algo_title is not None:
        plt.title(rank_algo_title, fontsize=18)
    plt.tight_layout()
    #plt.savefig('img/report_top_k_scores_first_gen.png', dpi=400, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    json_files = ['results/mutual_info_score_seed0_sub0.01_features-all.json',
                  'results/mutual_info_score_seed0_sub0.001_features-all.json',
                  'results/chi2_score_seed0_sub0.01_features-all.json',
                  'results/chi2_score_seed0_sub0.001_features-all.json']

    json_files = ['results/mutual_info_score_seed0_sub0.1_features-all.json',
                  'results/mutual_info_score_seed0_sub0.1_features-all_leon_fail.json']

    plot_ranking_results(json_files)
    #plot_ranking_results_same_algo(json_files)


