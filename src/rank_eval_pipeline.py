import pandas as pd
import numpy as np
import time
from datetime import datetime

import eval_algos
import rank_algos
import helper_functions as hf


class RankEval:
    def __init__(self, data, rank_method,
                 eval_method=eval_algos.jaccard_full_score,
                 seed=0, subsampling_proportion= 1,
                 X=None, y=None,
                 exec_time=None,
                 ranking=None, scores=None,
                 ground_truth_singles=None, ground_truth_first_gen=None,
                 eval_res_singles=None, eval_res_first_gen=None):

        self.data = data
        self.rank_method = rank_method
        self.eval_method = eval_method
        self.seed = seed
        self.subsampling_proportion = subsampling_proportion
        self.X = X
        self.y = y
        self.exec_time = exec_time
        self.ranking = ranking
        self.scores = scores
        self.ground_truth_singles = ground_truth_singles
        self.ground_truth_first_gen = ground_truth_first_gen
        self.eval_res_singles = eval_res_singles
        self.eval_res_first_gen = eval_res_first_gen
        self.sample_indices = None

    def get_sample_indices(self):
        """
        Get indices of a random sample of the data.

        :return: None (Updates its own sample_indices attribute.)
        """
        if self.sample_indices is None:
            self.sample_indices = np.random.choice(
                self.data.index, size=int(self.data.shape[0] * self.subsampling_proportion),
                replace=False, random_state=self.seed)

    def get_X_y(self):
        """
        Get X and y from the data. Uses the sample_indices attribute if it is not None.

        :return: None (Updates its own X and y attributes.)
        """
        if self.sample_indices is not None:
            y = self.data.loc[self.sample_indices, 'info_click_valid']
            X = self.data.loc[self.sample_indices].drop(['info_click_valid'], axis=1)
        else:
            y = self.data['info_click_valid']
            X = self.data.drop(['info_click_valid'], axis=1)

        self.X = X
        self.y = y


    def get_ground_truth(self):
        """
        Get ground truth for the eval_method.

        :return: None (Updates its own ground_truth_singles and ground_truth_first_gen attributes.)
        """
        if self.ground_truth_singles is None or self.ground_truth_first_gen is None:
            self.ground_truth_singles, self.ground_truth_first_gen = eval_algos.get_ground_truths()

        # add smallest value to all scores to avoid negative values for fuzzy jaccard
        if self.eval_method == eval_algos.fuzzy_jaccard:
            self.ground_truth_singles['rig'] = self.ground_truth_singles['rig'].astype(float)
            self.ground_truth_first_gen['rig'] = self.ground_truth_first_gen['rig'].astype(float)

            self.ground_truth_singles['rig'] = self.ground_truth_singles['rig'] + abs(
                self.ground_truth_singles['rig'].min())
            self.ground_truth_first_gen['rig'] = self.ground_truth_first_gen['rig'] + abs(
                self.ground_truth_first_gen['rig'].min())

    def sort_ground_truth(self):
        """
        Sort ground truth by feature names as they are in self.ranking.
        Also converts ground truth to numpy array of floats.

        :return: None (Only updates its own ground_truth_singles and ground_truth_first_gen attributes.)
        """
        if self.ranking is None:
            self.rank()

        # sort ground truth so that index matches ranking
        self.ground_truth_singles = self.ground_truth_singles.reindex(self.ranking)
        self.ground_truth_first_gen = self.ground_truth_first_gen.reindex(self.ranking)

        # convert to numpy array
        self.ground_truth_singles = np.array(self.ground_truth_singles)
        self.ground_truth_first_gen = np.array(self.ground_truth_first_gen)

    @hf.measure_runtime
    def rank(self):
        """
        Feature ranking using the rank_method provided.

        rank_method: function that takes X and y and
        returns two numpy arrays with feature names and corresponding scores, respectively.
        Higher scores should mean more relevant features.

        :return: None (Only updates its own ranking and scores attributes.)
        """

        if self.X is None or self.y is None:
            self.get_X_y()

        start_time = time.time()
        feature_list, score_list = self.rank_method(self.X, self.y)
        end_time = time.time()

        # sort by score descending
        sort_indexes = np.lexsort([feature_list, -score_list])
        feature_list = feature_list[sort_indexes]
        score_list = score_list[sort_indexes]

        self.ranking = feature_list
        self.scores = score_list
        self.exec_time = end_time - start_time

    def evaluate_ranking(self):
        """
        Evaluate the ranking using the eval_method provided.

        :return: None (Only updates its own df_scores attribute.)
        """

        # get ranking and ground truths
        if self.ranking is None or self.scores is None:
            self.rank()
        if self.ground_truth_singles is None or self.ground_truth_first_gen is None:
            self.get_ground_truth()

        # sort ground truths in the same way as ranking
        self.sort_ground_truth()

        # add smallest value to all scores to avoid negative values for fuzzy jaccard
        if self.eval_method == eval_algos.fuzzy_jaccard:
            self.scores = self.scores + abs(self.scores.min())

        evaluation_singles = self.eval_method(self.scores, self.ground_truth_singles)
        evaluation_first_gen = self.eval_method(self.scores, self.ground_truth_first_gen)

        self.eval_res_singles = evaluation_singles
        self.eval_res_first_gen = evaluation_first_gen

    def evaluate_and_log(self, sampling, file_dir='results/autoresults/'):
        """
        Evaluates the ranking by comparing it to both ground truths and writes results to txt file.

        :return: None (Only logs the results.)
        """
        if self.eval_res_singles is None or self.eval_res_first_gen is None:
            self.evaluate_ranking()

        now = datetime.now()
        now = now.strftime("%d_%m_%Y_%H_%M_%S")

        with open(f'{file_dir}result_{now}.txt', 'a') as f:
            f.write(f"Sampling: {sampling}\n")
            f.write(f"Ranking method: {self.rank_method.__name__}\n")
            f.write(f"Evaluation method: {self.eval_method.__name__}\n")
            f.write(f"Execution time [sec]: {self.exec_time}\n")
            f.write(f"Result singles: {self.eval_res_singles}\n")
            f.write(f"Result first gen: {self.eval_res_first_gen}\n")
            f.write(f"Ranking:\n{self.ranking}\n")

    def get_ranking(self):
        """
        Getter for the ranking.

        :return: The ranking.
        """
        if self.ranking is None:
            self.rank()

        return self.ranking

    def get_scores(self):
        """
        Getter for ranking and scores.

        :return: Two numpy arrays with feature names and corresponding scores, respectively.
        """
        if self.scores is None:
            self.rank()

        return self.ranking, self.scores

    def get_eval_res(self):
        """
        Getter for the evaluation results.

        :return: Results of the evaluation for singles and first gen ground truth, respectively.
        """
        if self.eval_res_singles is None or self.eval_res_first_gen is None:
            self.evaluate_ranking()

        return self.eval_res_singles, self.eval_res_first_gen


if __name__ == "__main__":
    data = pd.read_csv('data/full_data.csv')
    #data = data.head(100)

    rank_method = rank_algos.mutual_info_score
    eval_method = eval_algos.jaccard_full_score
    evaluator = RankEval(data, rank_method, eval_method)
    evaluator = RankEval(data, rank_method, eval_method, subsampling_proportion=0.001)
    
    print(evaluator.get_eval_res())

