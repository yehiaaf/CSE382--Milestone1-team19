import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features.selection import (
    CLUSTER_CATEGORICAL, CLUSTER_LOG_COLUMNS, CLUSTER_PLAIN_COLUMNS,
)


def make_encoder():
    return OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=50,
        sparse_output=False,
    )


def build_supervised_preprocessor(numeric, categorical, scale=True):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    cat_steps = [
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", make_encoder()),
    ]
    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("cat", Pipeline(cat_steps), categorical),
    ])


def build_clustering_preprocessor():
    log_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
    ]
    plain_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    cat_steps = [
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", make_encoder()),
    ]
    return ColumnTransformer([
        ("log", Pipeline(log_steps), CLUSTER_LOG_COLUMNS),
        ("plain", Pipeline(plain_steps), CLUSTER_PLAIN_COLUMNS),
        ("cat", Pipeline(cat_steps), CLUSTER_CATEGORICAL),
    ])