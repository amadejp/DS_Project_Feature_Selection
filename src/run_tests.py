import os
import subprocess
import numpy as np
import time

algo = 'random_forest_score'
batch_file = "batch0.txt"
directory = "random_forest_selected_features/" # results directory is assumed in hpc_app.py
features_file = "remove_features_1.txt"

# List of subsampling proportions to test
#subsample_proportions = np.logspace(-4, -2, 11)[:-1]
subsample_proportions = [0.0001, 0.00016, 0.00025, 0.0004, 0.00063, 0.001, 0.0016, 0.0025, 0.004, 0.0063]

# def output_file_exists(algo, proportion, seed, feature_str='all'):
#     file_name = f'results/random_forest_optimization/{algo}_seed{seed}_sub{proportion}_features-{feature_str}.json'
#     return os.path.exists(file_name)

# Record the start time
start_time = time.time()

# Loop over the algorithms and subsampling proportions and run hpc_app.py with the desired arguments
for proportion in subsample_proportions:
    print(f'Running algorithm: {algo} with subsampling proportion: {proportion}')
    cmd = f'python hpc_app.py --rank-algo {algo} --subsample {proportion} --batch {batch_file} --directory {directory} --drop-features-file {features_file}'
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f'Finished for batch: {batch_file} with subsampling proportion: {proportion}\n')
    except subprocess.CalledProcessError as e:
        print(f'Error with batch: {batch_file} with subsampling proportion: {proportion} - {e}\n')

# Record the end time
end_time = time.time()

# Print out the total runtime
print(f'Total runtime: {end_time - start_time} seconds')
