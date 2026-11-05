import argparse

import joblib
import pandas as pd

from src.config import ensure_dirs, path
from src.data.eda import run_eda
from src.data.loader import clean_data, download_data, load_raw, verify_h2
from src.data.splitter import get_test_data, get_train_data, make_split
from src.features.engineering import add_features
from src.features.leakage import run_leakage_checks
from src.features.selection import DECISIONS
from src.classification.baseline import run_classification_baseline
from src.clustering.baseline import run_clustering_baseline
from src.regression.baseline import run_regression_baseline


def load_clean():
    raw = load_raw()
    cleaned, log = clean_data(raw)
    return raw, add_features(cleaned), log


def cmd_download(args):
    print("saved to", download_data())


def cmd_analyze(args):
    raw, df, log = load_clean()
    checks = verify_h2(raw)
    checks.to_csv(path("outputs", "tables", "verification.csv"), index=False)
    print(checks.to_string(index=False))
    run_eda(raw, df, log)
    decisions = pd.DataFrame(DECISIONS, columns=["column", "classification", "regression", "clustering", "reason"])
    decisions.to_csv(path("outputs", "tables", "feature_decisions.csv"), index=False)
    print(log.to_string(index=False))
    print("EDA tables and figures saved in outputs/")


def cmd_split(args):
    raw, df, log = load_clean()
    train_idx, test_idx = make_split(df)
    joblib.dump({"train_idx": train_idx, "test_idx": test_idx}, path("artifacts", "split_indices.joblib"))
    rows = []
    for task in ["classification", "regression"]:
        train = get_train_data(df, task)
        test = get_test_data(df, task)
        for name, part in [("train", train), ("test", test)]:
            rows.append({
                "task": task,
                "set": name,
                "rows": len(part),
                "canceled_share_pct": round(100 * part["IsCanceled"].mean(), 2) if name == "train" else "not inspected",
            })
    table = pd.DataFrame(rows)
    table.to_csv(path("outputs", "tables", "split_summary.csv"), index=False)
    print(table.to_string(index=False))


def cmd_leakage(args):
    raw, df, log = load_clean()
    evidence, cls, reg, dup = run_leakage_checks(raw, df)
    for t in [evidence, cls, reg, dup]:
        print(t.to_string(index=False))
        print()


def cmd_train(args):
    raw, df, log = load_clean()
    if args.task in ["regression", "all"]:
        results, coefs = run_regression_baseline(get_train_data(df, "regression"))
        print(results.to_string(index=False))
        print(coefs.to_string(index=False))
    if args.task in ["classification", "all"]:
        results, coefs, cm = run_classification_baseline(get_train_data(df, "classification"))
        print(results.to_string(index=False))
        print(coefs.to_string(index=False))
        print(cm)


def cmd_cluster(args):
    raw, df, log = load_clean()
    baseline, sweep, profile, best_k, n_features = run_clustering_baseline(df)
    print(baseline.to_string(index=False))
    print(sweep.to_string(index=False))
    print("best k by silhouette:", best_k, "| features after encoding:", n_features)
    print(profile.to_string(index=False))


def cmd_all(args):
    cmd_analyze(args)
    cmd_split(args)
    cmd_leakage(args)
    args.task = "all"
    cmd_train(args)
    cmd_cluster(args)


def build_parser():
    parser = argparse.ArgumentParser(description="CSE382 City Hotel (H2) machine learning project - Milestone 1")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download").set_defaults(func=cmd_download)
    sub.add_parser("analyze").set_defaults(func=cmd_analyze)
    sub.add_parser("split").set_defaults(func=cmd_split)
    sub.add_parser("leakage").set_defaults(func=cmd_leakage)
    train = sub.add_parser("train")
    train.add_argument("--task", choices=["regression", "classification", "all"], default="all")
    train.set_defaults(func=cmd_train)
    sub.add_parser("cluster").set_defaults(func=cmd_cluster)
    sub.add_parser("all").set_defaults(func=cmd_all)
    return parser


if __name__ == "__main__":
    ensure_dirs()
    arguments = build_parser().parse_args()
    arguments.func(arguments)