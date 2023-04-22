import argparse
import json
import pandas as pd
import rank_algos
from rank_eval_pipeline import RankEval
from helper_functions import generate_feature_hash


def main(args):   
    # load data
    data = pd.read_csv('data/full_data.csv')

    # load features to remove from file, if specified
    features_to_remove = set(args.drop_features)
    if args.drop_features_file:
        with open(args.drop_features_file, 'r') as f:
            file_features = [line.strip() for line in f.readlines()]
            features_to_remove.update(file_features)

    # create a dictionary that maps algorithm names to functions
    algos = {
        'mutual_info_score': rank_algos.mutual_info_score,
        'ReliefF_score': rank_algos.ReliefF_score,
        #'ReliefE_score': rank_algos.ReliefE_score,
        'SURF_score': rank_algos.SURF_score,
        'SURFstar_score': rank_algos.SURFstar_score,
        'MultiSURF_score': rank_algos.MultiSURF_score,
        'MultiSURFstar_score': rank_algos.MultiSURFstar_score,
        'xgboost_score': rank_algos.xgboost_score,
        'random_forest_score': rank_algos.random_forest_score,
        'chi2_score': rank_algos.chi2_score,
        'pearson_correlation_score': rank_algos.pearson_correlation_score,
        'anova_f_score': rank_algos.anova_f_score
    }
    # set the rank algorithm based on the input argument
    if args.rank_algo not in algos:
        print(f'Invalid ranking algorithm specified. Available options are: {", ".join(algos.keys())}')
        return
    rank_algo = algos[args.rank_algo]

    # create the RankEval object
    RE = RankEval(data, rank_algo, 
                  seed=args.seed, 
                  subsampling_proportion=args.subsample, 
                  features_to_remove=list(features_to_remove))
    
    
    # get the scores
    results = RE.get_scores()
    evaluations = RE.get_eval_res()

    # create a dictionary to store the results
    output_dict = {
        'rank_algo': args.rank_algo,
        'seed': args.seed,
        'subsampling_proportion': args.subsample,
        'exec_time': RE.exec_time,
        'results': {
            'features': results[0].tolist(),
            'scores': results[1].tolist()
        },
        'evaluations': {
            'eval_method': str(RE.eval_method.__name__),
            'singles': evaluations[0],
            'first_gen': evaluations[1]
        }
    }

    # save results to a JSON file
    if features_to_remove:
        features_hash = generate_feature_hash(features_to_remove)
        feature_str = features_hash
    else:
        feature_str = "all"

    output_file_name = f'results/{args.rank_algo}_seed{args.seed}_sub{args.subsample}_features-{feature_str}.json'
    with open(output_file_name, 'w') as f:
        json.dump(output_dict, f)

    print(f'Results saved to {output_file_name}')


if __name__ == '__main__':
    # define the command line arguments
    parser = argparse.ArgumentParser(description='Rank evaluation pipeline')
    parser.add_argument('--rank-algo', type=str, required=True,
                        help='The name of the ranking algorithm to use (chi2, mutual_info, rf, etc.)')
    parser.add_argument('--seed', type=int, default=0,
                        help='The random seed to use')
    parser.add_argument('--subsample', type=float, default=0.1,
                        help='The proportion of data to subsample')
    parser.add_argument('--drop-features', type=str, nargs='*', default=[],
                        help='List of feature names to remove (e.g., --drop-features feature99 feature98)')
    parser.add_argument('--drop-features-file', type=str, default=None,
                        help='A file containing a list of feature names to remove, one per line')
    args = parser.parse_args()

    # call the main function with the command line arguments
    main(args)
