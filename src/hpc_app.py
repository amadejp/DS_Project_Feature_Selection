import argparse
import json
import pandas as pd
import rank_algos
from rank_eval_pipeline import RankEval


def main(args):
    # load data
    data = pd.read_csv('data/full_data.csv')

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
    RE = RankEval(data, rank_algo, seed=args.seed, subsampling_proportion=args.subsample)

    # get the scores
    results = RE.get_scores()
    evaluations = RE.get_eval_res()

    # create a dictionary to store the results
    output_dict = {
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
    output_file_name = f'results/{args.rank_algo}_seed{args.seed}_sub{args.subsample}.json'
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
    args = parser.parse_args()

    # call the main function with the command line arguments
    main(args)
