import pandas as pd
import numpy as np

import eval_algos
import rank_algos


class RankEval:
    def __init__(self, data, rank_method, eval_method,
                 df_scores=None, ground_truth_singles=None, ground_truth_first_gen=None):
        self.data = data
        self.rank_method = rank_method
        self.eval_method = eval_method
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

            self.ground_truth_singles['rig'] = self.ground_truth_singles['rig'] + abs(self.ground_truth_singles['rig'].min())
            self.ground_truth_first_gen['rig'] = self.ground_truth_first_gen['rig'] + abs(self.ground_truth_first_gen['rig'].min())

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

        :return:
        """
        if eval_only:
            return (self.eval_method(self.df_scores, self.ground_truth_singles)[1],
                    self.eval_method(self.df_scores, self.ground_truth_first_gen)[1])

        if self.df_scores is None:
            self.score()

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


if __name__ == "__main__":
    data = pd.read_csv('data/full_data.csv')
    data = data.head(100)

    rank_method = rank_algos.mutual_info_score
    eval_method = eval_algos.jaccard
    rank_eval = RankEval(data, rank_method, eval_method)

    print(rank_eval.evaluate())

    print(rank_eval.evaluate_single())
