import pandas as pd
import numpy as np

import eval_algos
import rank_algos


class RankEval:
    def __init__(self, data, rank_method, eval_method,
                 df_scores=None, df_ranked_scores = None, ground_truth=None):
        self.data = data
        self.rank_method = rank_method
        self.eval_method = eval_method
        self.df_scores = df_scores
        self.df_ranked_scores = df_ranked_scores
        self.ground_truth = ground_truth

    def get_X_y(self):
        y = self.data['info_click_valid']
        X = self.data.drop(['info_click_valid'], axis=1)
        return X, y

    def get_ground_truth(self):
        if self.ground_truth is None:
            self.ground_truth = eval_algos.get_ground_truth()

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

        self.df_scores = df_scores

    def rank(self):
        if self.df_scores is None:
            self.score()

        df_scores_sorted = self.df_scores.sort_values(by=['score'], ascending=False)

        self.df_ranked_scores = df_scores_sorted

    def evaluate(self):
        if self.df_scores is None:
            self.score()

        if self.ground_truth is None:
            self.get_ground_truth()

        return self.eval_method(self.df_scores["score"].to_numpy().astype(float),
                                self.ground_truth["rig"].to_numpy().astype(float))

    def evaluate_testing(self):
        return self.eval_method(self.df_scores,
                                self.ground_truth)


if __name__ == "__main__":
    data = pd.read_csv('data/full_data.csv')
    data = data.head(100)

    rank_method = rank_algos.mutual_info_score
    eval_method = eval_algos.jaccard
    rank_eval = RankEval(data, rank_method, eval_method)

    print(rank_eval.evaluate())