import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.config import load_config, path
from src.data.loader import clean_data
from src.data.splitter import get_train_data
from src.features.engineering import add_features, add_status_offset
from src.features.selection import get_features
from src.preprocessing.pipelines import build_supervised_preprocessor

CLASSIFICATION_SUSPECTS = [
    ("ReservationStatus", "cat"),
    ("StatusDateOffset", "num"),
    ("AssignedRoomType", "cat"),
    ("BookingChanges", "num"),
    ("RequiredCarParkingSpaces", "num"),
    ("TotalOfSpecialRequests", "num"),
]

REGRESSION_SUSPECTS = [
    ("IsCanceled", "num"),
    ("ReservationStatus", "cat"),
    ("AssignedRoomType", "cat"),
    ("BookingChanges", "num"),
    ("RequiredCarParkingSpaces", "num"),
    ("TotalOfSpecialRequests", "num"),
]


def leakage_evidence(df):
    d = df.copy()
    d["RoomChanged"] = (d["ReservedRoomType"] != d["AssignedRoomType"]).astype(int)
    d["HasParking"] = (d["RequiredCarParkingSpaces"] > 0).astype(int)
    d["HasSpecialRequest"] = (d["TotalOfSpecialRequests"] > 0).astype(int)
    d["HasBookingChange"] = (d["BookingChanges"] > 0).astype(int)
    d["NonRefund"] = (d["DepositType"] == "Non Refund").astype(int)
    g = d.groupby("IsCanceled")
    rows = [
        ("ReservationStatus", "share labelled Canceled or No-Show (%)", 100 * d.loc[d["IsCanceled"] == 0, "ReservationStatus"].isin(["Canceled", "No-Show"]).mean(), 100 * d.loc[d["IsCanceled"] == 1, "ReservationStatus"].isin(["Canceled", "No-Show"]).mean()),
        ("ReservationStatusDate", "mean days from arrival to status date", g["StatusDateOffset"].mean()[0], g["StatusDateOffset"].mean()[1]),
        ("AssignedRoomType", "share where assigned room differs from reserved (%)", 100 * g["RoomChanged"].mean()[0], 100 * g["RoomChanged"].mean()[1]),
        ("BookingChanges", "share with at least one change (%)", 100 * g["HasBookingChange"].mean()[0], 100 * g["HasBookingChange"].mean()[1]),
        ("RequiredCarParkingSpaces", "share with a parking space (%)", 100 * g["HasParking"].mean()[0], 100 * g["HasParking"].mean()[1]),
        ("TotalOfSpecialRequests", "share with at least one request (%)", 100 * g["HasSpecialRequest"].mean()[0], 100 * g["HasSpecialRequest"].mean()[1]),
        ("DepositType", "share of Non Refund bookings (%)", 100 * g["NonRefund"].mean()[0], 100 * g["NonRefund"].mean()[1]),
    ]
    out = pd.DataFrame(rows, columns=["feature", "measure", "not_canceled", "canceled"])
    out["not_canceled"] = out["not_canceled"].round(2)
    out["canceled"] = out["canceled"].round(2)
    return out


def classification_ablation(df):
    cfg = load_config()
    train = get_train_data(df, "classification")
    numeric, categorical = get_features("classification")
    y = train["IsCanceled"]
    cv = StratifiedKFold(n_splits=cfg["leakage"]["ablation_folds"], shuffle=True, random_state=cfg["seed"])
    scoring = {"accuracy": "accuracy", "roc_auc": "roc_auc"}
    rows = []
    setups = [("clean feature set (used in the project)", [], [])]
    for col, kind in CLASSIFICATION_SUSPECTS:
        setups.append(("clean + " + col, [col] if kind == "num" else [], [col] if kind == "cat" else []))
    for label, extra_num, extra_cat in setups:
        num = numeric + extra_num
        cat = categorical + extra_cat
        pipe = Pipeline([
            ("prep", build_supervised_preprocessor(num, cat, scale=True)),
            ("model", LogisticRegression(max_iter=500, random_state=cfg["seed"])),
        ])
        scores = cross_validate(pipe, train[num + cat], y, cv=cv, scoring=scoring)
        rows.append({"feature set": label, "accuracy": round(scores["test_accuracy"].mean(), 4), "roc_auc": round(scores["test_roc_auc"].mean(), 4)})
    return pd.DataFrame(rows)


def regression_ablation(df):
    cfg = load_config()
    train = get_train_data(df, "regression")
    numeric, categorical = get_features("regression")
    y = train["ADR"]
    cv = KFold(n_splits=cfg["leakage"]["ablation_folds"], shuffle=True, random_state=cfg["seed"])
    scoring = {"mae": "neg_mean_absolute_error", "r2": "r2"}
    rows = []
    setups = [("clean feature set (used in the project)", [], [])]
    for col, kind in REGRESSION_SUSPECTS:
        setups.append(("clean + " + col, [col] if kind == "num" else [], [col] if kind == "cat" else []))
    for label, extra_num, extra_cat in setups:
        num = numeric + extra_num
        cat = categorical + extra_cat
        pipe = Pipeline([
            ("prep", build_supervised_preprocessor(num, cat, scale=True)),
            ("model", LinearRegression()),
        ])
        scores = cross_validate(pipe, train[num + cat], y, cv=cv, scoring=scoring)
        rows.append({"feature set": label, "mae": round(-scores["test_mae"].mean(), 3), "r2": round(scores["test_r2"].mean(), 4)})
    return pd.DataFrame(rows)


def duplicate_inflation(raw):
    cfg = load_config()
    numeric, categorical = get_features("classification")
    rows = []
    for label, flag in [("duplicates removed (used in the project)", True), ("duplicates kept", False)]:
        cleaned, _ = clean_data(raw, drop_duplicates=flag)
        cleaned = add_features(cleaned)
        train = get_train_data(cleaned, "classification")
        cv = StratifiedKFold(n_splits=cfg["leakage"]["ablation_folds"], shuffle=True, random_state=cfg["seed"])
        pipe = Pipeline([
            ("prep", build_supervised_preprocessor(numeric, categorical, scale=True)),
            ("model", LogisticRegression(max_iter=500, random_state=cfg["seed"])),
        ])
        scores = cross_validate(pipe, train[numeric + categorical], train["IsCanceled"], cv=cv, scoring={"accuracy": "accuracy", "roc_auc": "roc_auc"})
        rows.append({
            "dataset": label,
            "rows": len(cleaned),
            "train_rows": len(train),
            "canceled_share_pct": round(100 * cleaned["IsCanceled"].mean(), 1),
            "cv_accuracy": round(scores["test_accuracy"].mean(), 4),
            "cv_roc_auc": round(scores["test_roc_auc"].mean(), 4),
        })
    return pd.DataFrame(rows)


def run_leakage_checks(raw, df):
    df = add_status_offset(df)
    evidence = leakage_evidence(get_train_data(df, "classification"))
    cls = classification_ablation(df)
    reg = regression_ablation(df)
    dup = duplicate_inflation(raw)
    evidence.to_csv(path("outputs", "tables", "leakage_evidence.csv"), index=False)
    cls.to_csv(path("outputs", "tables", "leakage_ablation_classification.csv"), index=False)
    reg.to_csv(path("outputs", "tables", "leakage_ablation_regression.csv"), index=False)
    dup.to_csv(path("outputs", "tables", "leakage_duplicates.csv"), index=False)
    return evidence, cls, reg, dup