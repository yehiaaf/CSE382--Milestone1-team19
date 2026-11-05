import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from src.config import load_config, path
from src.evaluation.metrics import REGRESSION_SCORING, summarize_scores
from src.evaluation.plots import plot_pred_vs_actual, plot_residuals
from src.features.selection import get_features
from src.preprocessing.pipelines import build_supervised_preprocessor


def build_models():
    return {
        "Dummy (mean ADR)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
    }


def make_pipeline(model, numeric, categorical):
    return Pipeline([
        ("prep", build_supervised_preprocessor(numeric, categorical, scale=True)),
        ("model", model),
    ])


def run_regression_baseline(train):
    cfg = load_config()
    numeric, categorical = get_features("regression")
    X = train[numeric + categorical]
    y = train["ADR"]
    cv = KFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["seed"])

    rows = []
    for name, model in build_models().items():
        pipe = make_pipeline(model, numeric, categorical)
        scores = cross_validate(pipe, X, y, cv=cv, scoring=REGRESSION_SCORING)
        row = {"model": name}
        row.update(summarize_scores(scores, list(REGRESSION_SCORING)))
        rows.append(row)
    results = pd.DataFrame(rows)

    lr_pipe = make_pipeline(LinearRegression(), numeric, categorical)
    oof = cross_val_predict(lr_pipe, X, y, cv=cv)
    plot_pred_vs_actual(y, oof, "regression_baseline_pred_vs_actual.png", "Linear Regression: predicted vs actual ADR")
    plot_residuals(y, oof, "regression_baseline_residuals.png", "Linear Regression: residuals (out-of-fold)")

    lr_pipe.fit(X, y)
    names = [n.split("__", 1)[1] for n in lr_pipe.named_steps["prep"].get_feature_names_out()]
    coefs = pd.DataFrame({"feature": names, "coefficient": lr_pipe.named_steps["model"].coef_})
    coefs["abs_coefficient"] = coefs["coefficient"].abs()
    coefs = coefs.sort_values("abs_coefficient", ascending=False).head(15).drop(columns="abs_coefficient")
    coefs["coefficient"] = coefs["coefficient"].round(3)

    joblib.dump(lr_pipe, path("artifacts", "baseline_regression_linear.joblib"))
    results.to_csv(path("outputs", "tables", "regression_baseline.csv"), index=False)
    coefs.to_csv(path("outputs", "tables", "regression_baseline_top_coefficients.csv"), index=False)
    return results, coefs