import pandas as pd
import numpy as np
import time
from datetime import datetime

import eval_algos
import rank_algos
import helper_functions as hf


class RankEval:
    def __init__(self, data, rank_method,
                 eval_method=eval_algos.fuzzy_jaccard,
                 exec_time=None,
                 df_scores=None, ground_truth_singles=None, ground_truth_first_gen=None):
        self.data = data
        self.rank_method = rank_method
        self.eval_method = eval_method
        self.exec_time = exec_time
        self.df_scores = df_scores
        self.ground_truth_singles = ground_truth_singles
        self.ground_truth_first_gen = ground_truth_first_gen

    def get_X_y(self):
        y = self.data['info_click_valid']
        X = self.data.drop(['info_click_valid'], axis=1)
        return X, y

    def get_ground_truth(self):
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

    @hf.measure_runtime
    def score(self):
        """
        Score features using the rank_method.

        :param rank_method: function that takes X and y and returns a list of scores

        :return: None (Only updates its own df_scores attribute as df with feature names and scores.)
        """

        X, y = self.get_X_y()

        scored_features = self.rank_method(X, y)

        df_scores = pd.DataFrame(X.columns, columns=['feature'])
        df_scores['score'] = scored_features

        # add smallest value to all scores to avoid negative values for fuzzy jaccard
        if self.eval_method == eval_algos.fuzzy_jaccard:
            df_scores['score'] = df_scores['score'] + abs(df_scores['score'].min())

        self.df_scores = df_scores

    def rank_order(self):
        """
        Orders the features by their score. (ranking)

        :return: df with feature names and scores sorted by score.
        """
        if self.df_scores is None:
            self.score()

        df_scores_sorted = self.df_scores.sort_values(by=['score'], ascending=False)

        return df_scores_sorted

    def evaluate_single(self, eval_only=False, full_output=False):
        if eval_only:
            return self.eval_method(self.df_scores, self.ground_truth_singles)[1]

        if full_output:
            return self.eval_method(self.df_scores["score"].values, self.ground_truth_singles["rig"].values)

        return self.eval_method(self.df_scores["score"].values, self.ground_truth_singles["rig"].values)[1]

    def evaluate_first_gen(self, eval_only=False, full_output=False):
        if eval_only:
            return self.eval_method(self.df_scores, self.ground_truth_first_gen)[1]

        if full_output:
            return self.eval_method(self.df_scores["score"].values, self.ground_truth_first_gen["rig"].values)

        return self.eval_method(self.df_scores["score"].values, self.ground_truth_first_gen["rig"].values)[1]

    def evaluate(self, eval_only=False, full_output=False):
        """
        Evaluates the ranking by comparing it to both ground truths.

        :param eval_only: If True, only the evaluation is performed. If False, the ranking is performed first.
                            For evaluation only, ground truth and scores must be provided only as numpy array / list!
        :param full_output: If True, the full output of the evaluation method is returned. If False, only the AUC score
                            is returned.

        :return: Scores for both ground truths.
        """
        if eval_only:
            return (self.eval_method(self.df_scores, self.ground_truth_singles)[1],
                    self.eval_method(self.df_scores, self.ground_truth_first_gen)[1])

        if self.df_scores is None:
            start_time = time.time()
            self.score()
            end_time = time.time()
            self.exec_time = end_time - start_time

        self.get_ground_truth()

        if full_output:
            return (self.eval_method(self.df_scores["score"].to_numpy().astype(float),
                                     self.ground_truth_singles["rig"].to_numpy().astype(float)),
                    self.eval_method(self.df_scores["score"].to_numpy().astype(float),
                                     self.ground_truth_first_gen["rig"].to_numpy().astype(float)))

        return (self.eval_method(self.df_scores["score"].to_numpy().astype(float),
                                 self.ground_truth_singles["rig"].to_numpy().astype(float))[1],
                self.eval_method(self.df_scores["score"].to_numpy().astype(float),
                                 self.ground_truth_first_gen["rig"].to_numpy().astype(float))[1])

    def evaluate_and_log(self, sampling, file_dir='results/autoresults/'):
        """
        Evaluates the ranking by comparing it to both ground truths and writes results to txt file.

        :return: None (Only logs the results.)
        """
        auc_singles, auc_first_gen = self.evaluate()
        # write datetime as filename
        now = datetime.now()
        now = now.strftime("%d_%m_%Y_%H_%M_%S")
        print(now)
        with open(f'{file_dir}result_{now}.txt', 'a') as f:
            f.write(f"Sampling: {sampling}\n")
            f.write(f"Ranking method: {self.rank_method.__name__}\n")
            f.write(f"Evaluation method: {self.eval_method.__name__}\n")
            f.write(f"Execution time [sec]: {self.exec_time}\n")
            f.write(f"AUC singles: {auc_singles}\n")
            f.write(f"AUC first gen: {auc_first_gen}\n")
            f.write(f"Ranking:\n{self.rank_order()}\n")


if __name__ == "__main__":
    data = pd.read_csv('data/full_data.csv')
    data = data.head(100)

    rank_method = rank_algos.mutual_info_score
    eval_method = eval_algos.fuzzy_jaccard
    rank_eval = RankEval(data, rank_method, eval_method)

    rank_eval.evaluate_and_log("100 samples")
