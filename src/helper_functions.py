# helper_functions.py

import time
import numpy as np
import pandas as pd


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