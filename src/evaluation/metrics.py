from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score

REGRESSION_SCORING = {
    "mae": "neg_mean_absolute_error",
    "mse": "neg_mean_squared_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}

CLASSIFICATION_SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}

NEGATED = ["mae", "mse", "rmse"]


def summarize_scores(scores, keys):
    row = {}
    for key in keys:
        values = scores["test_" + key]
        if key in NEGATED:
            values = -values
        row[key + "_mean"] = round(float(values.mean()), 4)
        row[key + "_std"] = round(float(values.std()), 4)
    row["fit_time_s"] = round(float(scores["fit_time"].mean()), 3)
    return row