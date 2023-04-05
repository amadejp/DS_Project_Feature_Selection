from sklearn.feature_selection import mutual_info_classif


def mutual_info_score(X, y):
    scores = mutual_info_classif(X, y)

    return X.columns, scores

