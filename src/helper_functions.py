# helper_functions.py

import time
import numpy as np
import pandas as pd
import hashlib
import json
import scipy.stats as stats

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


def plot_ranking_results(json_files):
    gt = eval_algos.get_ground_truths_ordered()
    gt_first_gen = np.array(gt[1].index).copy()

    k_list = range(1, 101)

    # Your current implementation for generating random_scores_dict
    random_scores_dict = {}
    for i in range(100):
        np.random.shuffle(gt_first_gen)
        for k in k_list:
            if k not in random_scores_dict:
                random_scores_dict[k] = []
            random_order = gt_first_gen.copy()
            np.random.shuffle(random_order)
            random_scores_dict[k].append(eval_algos.jaccard_k(random_order[:k], gt_first_gen[:k], k))

    # Do bootstrap for each k
    random_bootstrapped = {}
    for k in k_list:
        random_bootstrapped[k] = stats.bootstrap((random_scores_dict[k],), np.mean, confidence_level=0.95,
                                                 n_resamples=100, method='percentile')

    fig, ax = plt.subplots(figsize=(8, 6))

    colorblind_colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628']
    linestyles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1, 1, 1))]

    for idx, file in enumerate(json_files):
        with open(file, 'r') as f:
            data = json.load(f)
            rank_algo = data["rank_algo"]
            subsampling_proportion = data["subsampling_proportion"]
            ranking_results = data["evaluations"]["first_gen"][0]

            print(ranking_results)
            ax.plot(ranking_results, label=f'{rank_algo} ({subsampling_proportion})',
                    color=colorblind_colors[idx % len(colorblind_colors)], linestyle=linestyles[idx % len(linestyles)])

    random_mean_list = []
    random_ci_low_list = []
    random_ci_high_list = []
    for k in k_list:
        random_mean = np.mean(random_scores_dict[k])
        random_ci_low = random_bootstrapped[k].confidence_interval.low
        random_ci_high = random_bootstrapped[k].confidence_interval.high
        random_mean_list.append(random_mean)
        random_ci_low_list.append(random_ci_low)
        random_ci_high_list.append(random_ci_high)

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
    ax.legend(prop={'size': 15})

    # add grid
    ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('img/report_top_k_scores_first_gen.png', dpi=400, bbox_inches='tight')
    plt.show()




if __name__ == '__main__':
    json_files = ['results/mutual_info_score_seed0_sub0.01_features-all.json',
                  'results/mutual_info_score_seed0_sub0.001_features-all.json',
                  'results/chi2_score_seed0_sub0.01_features-all.json',
                  'results/chi2_score_seed0_sub0.001_features-all.json']

    plot_ranking_results(json_files)


