import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from src.config import load_config, path
from src.evaluation.metrics import CLASSIFICATION_SCORING, summarize_scores
from src.evaluation.plots import plot_confusion, plot_roc
from src.features.selection import get_features
from src.preprocessing.pipelines import build_supervised_preprocessor


def build_models(seed):
    return {
        "Dummy (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
    }


def make_pipeline(model, numeric, categorical):
    return Pipeline([
        ("prep", build_supervised_preprocessor(numeric, categorical, scale=True)),
        ("model", model),
    ])


def run_classification_baseline(train):
    cfg = load_config()
    numeric, categorical = get_features("classification")
    X = train[numeric + categorical]
    y = train["IsCanceled"]
    cv = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["seed"])

    rows = []
    for name, model in build_models(cfg["seed"]).items():
        pipe = make_pipeline(model, numeric, categorical)
        scores = cross_validate(pipe, X, y, cv=cv, scoring=CLASSIFICATION_SCORING)
        row = {"model": name}
        row.update(summarize_scores(scores, list(CLASSIFICATION_SCORING)))
        rows.append(row)
    results = pd.DataFrame(rows)

    lr_pipe = make_pipeline(LogisticRegression(max_iter=1000, random_state=cfg["seed"]), numeric, categorical)
    oof_class = cross_val_predict(lr_pipe, X, y, cv=cv)
    oof_proba = cross_val_predict(lr_pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    cm = confusion_matrix(y, oof_class)
    plot_confusion(cm, "classification_baseline_confusion.png", "Logistic Regression: confusion matrix (CV)")
    plot_roc(y, oof_proba, "classification_baseline_roc.png", "Logistic Regression: ROC curve (CV)")
    pd.DataFrame(cm, index=["actual_not_canceled", "actual_canceled"], columns=["pred_not_canceled", "pred_canceled"]).to_csv(
        path("outputs", "tables", "classification_baseline_confusion.csv")
    )

    lr_pipe.fit(X, y)
    names = [n.split("__", 1)[1] for n in lr_pipe.named_steps["prep"].get_feature_names_out()]
    coefs = pd.DataFrame({"feature": names, "coefficient": lr_pipe.named_steps["model"].coef_[0]})
    coefs["abs_coefficient"] = coefs["coefficient"].abs()
    coefs = coefs.sort_values("abs_coefficient", ascending=False).head(15).drop(columns="abs_coefficient")
    coefs["coefficient"] = coefs["coefficient"].round(3)

    joblib.dump(lr_pipe, path("artifacts", "baseline_classification_logreg.joblib"))
    results.to_csv(path("outputs", "tables", "classification_baseline.csv"), index=False)
    coefs.to_csv(path("outputs", "tables", "classification_baseline_top_coefficients.csv"), index=False)
    return results, coefs, cm