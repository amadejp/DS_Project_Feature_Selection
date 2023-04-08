import numpy as np
import ReliefF
import reliefe
from skrebate import SURF, SURFstar, MultiSURF, MultiSURFstar
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif



def mutual_info_score(X, y):
    scores = mutual_info_classif(X, y)

    return X.columns, scores

def ReliefF_score(X, y):
    ranker = ReliefF.ReliefF()
    ranker.fit(np.array(X), np.array(y))

    return X.columns, ranker.feature_scores

def ReliefE_score(X, y):
    ranker = reliefe.ReliefE()
    ranker.fit(np.array(X), np.array(y))

    return X.columns, ranker.feature_importances_

def SURF_score(X, y):
    """Warning: This will use all available cores by default.
    
    This algorithm will automatically determine the ideal number of neighbors to consider."""
    ranker = SURF(discrete_threshold=max(X.nunique()),# this forces all features to be treated as discrete
                   n_jobs=-1)
    ranker.fit(np.array(X), np.array(y))

    return X.columns, ranker.feature_importances_

def SURFstar_score(X, y):
    """Warning: This will use all available cores by default."""
    ranker = SURFstar(discrete_threshold=max(X.nunique()),# this forces all features to be treated as discrete
                       n_jobs=-1)
    ranker.fit(np.array(X), np.array(y))

    return X.columns, ranker.feature_importances_

def MultiSURF_score(X, y):
    """Warning: This will use all available cores by default."""
    ranker = MultiSURF(discrete_threshold=max(X.nunique()),# this forces all features to be treated as discrete
                       n_jobs=-1)
    ranker.fit(np.array(X), np.array(y))

    return X.columns, ranker.feature_importances_

def MultiSURFstar_score(X, y):
    """Warning: This will use all available cores by default."""
    ranker = MultiSURFstar(discrete_threshold=max(X.nunique()),# this forces all features to be treated as discrete
                           n_jobs=-1)
    ranker.fit(np.array(X), np.array(y))

    return X.columns, ranker.feature_importances_

def xgboost_score(X, y):
    ranker = xgb.XGBClassifier()
    ranker.fit(np.array(X), np.array(y))

    return X.columns, ranker.feature_importances_

def random_forest_score(X, y):
    ranker = RandomForestClassifier()
    ranker.fit(np.array(X), np.array(y))

    return X.columns, ranker.feature_importances_

def chi2_score(X, y):
    ranker = SelectKBest(score_func=chi2, k='all')
    ranker.fit(np.array(X), np.array(y))

    return X.columns, ranker.scores_