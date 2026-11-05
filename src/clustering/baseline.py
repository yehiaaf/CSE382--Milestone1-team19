import time

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from src.config import load_config, path
from src.evaluation.plots import plot_clusters, plot_k_sweep
from src.features.selection import CLUSTER_CATEGORICAL, CLUSTER_LOG_COLUMNS, CLUSTER_PLAIN_COLUMNS
from src.preprocessing.pipelines import build_clustering_preprocessor


def cluster_scores(X, labels, seed, sample_size):
    return {
        "silhouette": round(float(silhouette_score(X, labels, sample_size=sample_size, random_state=seed)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(X, labels)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(X, labels)), 1),
    }


def profile_clusters(df, labels):
    data = df.copy()
    data["cluster"] = labels
    data["TotalNights"] = data["StaysInWeekendNights"] + data["StaysInWeekNights"]
    rows = []
    for c, g in data.groupby("cluster"):
        rows.append({
            "cluster": c,
            "size": len(g),
            "share_pct": round(100 * len(g) / len(data), 1),
            "lead_time_mean": round(g["LeadTime"].mean(), 1),
            "total_nights_mean": round(g["TotalNights"].mean(), 2),
            "adults_mean": round(g["Adults"].mean(), 2),
            "children_mean": round(g["Children"].mean(), 2),
            "adr_mean": round(g["ADR"].mean(), 1),
            "repeated_guest_pct": round(100 * g["IsRepeatedGuest"].mean(), 1),
            "top_market_segment": g["MarketSegment"].mode().iloc[0],
            "top_customer_type": g["CustomerType"].mode().iloc[0],
            "cancel_rate_pct_posthoc": round(100 * g["IsCanceled"].mean(), 1),
        })
    return pd.DataFrame(rows)


def run_clustering_baseline(df):
    cfg = load_config()
    seed = cfg["seed"]
    sample = cfg["clustering"]["silhouette_sample"]
    cols = CLUSTER_LOG_COLUMNS + CLUSTER_PLAIN_COLUMNS + CLUSTER_CATEGORICAL

    prep = build_clustering_preprocessor()
    X = prep.fit_transform(df[cols])

    start = time.time()
    baseline = KMeans(n_clusters=cfg["clustering"]["baseline_k"], n_init=10, random_state=seed).fit(X)
    baseline_time = round(time.time() - start, 3)
    base_row = {"algorithm": "K-Means (baseline)", "k": cfg["clustering"]["baseline_k"], "fit_time_s": baseline_time}
    base_row.update(cluster_scores(X, baseline.labels_, seed, sample))
    baseline_table = pd.DataFrame([base_row])

    sweep_rows = []
    models = {}
    for k in cfg["clustering"]["k_range"]:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        row = {"k": k, "inertia": round(float(km.inertia_), 1)}
        row.update(cluster_scores(X, km.labels_, seed, sample))
        sweep_rows.append(row)
        models[k] = km
    sweep = pd.DataFrame(sweep_rows)
    sweep["rank_silhouette"] = sweep["silhouette"].rank(ascending=False)
    sweep["rank_davies_bouldin"] = sweep["davies_bouldin"].rank(ascending=True)
    sweep["rank_calinski_harabasz"] = sweep["calinski_harabasz"].rank(ascending=False)
    sweep["rank_sum"] = sweep["rank_silhouette"] + sweep["rank_davies_bouldin"] + sweep["rank_calinski_harabasz"]
    best_k = int(sweep.loc[sweep["rank_sum"].idxmin(), "k"])
    plot_k_sweep(sweep, "clustering_k_sweep.png")

    profile = profile_clusters(df, models[best_k].labels_)
    base_profile = profile_clusters(df, baseline.labels_)

    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(df), size=cfg["clustering"]["plot_sample"], replace=False)
    plot_clusters(coords[idx], models[best_k].labels_[idx], "clustering_pca_best_k.png", "K-Means k=%d in PCA space" % best_k, pca.explained_variance_ratio_)
    plot_clusters(coords[idx], baseline.labels_[idx], "clustering_pca_baseline.png", "K-Means k=%d (baseline) in PCA space" % cfg["clustering"]["baseline_k"], pca.explained_variance_ratio_)

    joblib.dump({"preprocessor": prep, "kmeans_baseline": baseline, "kmeans_best_k": models[best_k], "best_k": best_k}, path("artifacts", "baseline_clustering_kmeans.joblib"))
    baseline_table.to_csv(path("outputs", "tables", "clustering_baseline.csv"), index=False)
    sweep.to_csv(path("outputs", "tables", "clustering_k_sweep.csv"), index=False)
    profile.to_csv(path("outputs", "tables", "clustering_profile_best_k.csv"), index=False)
    base_profile.to_csv(path("outputs", "tables", "clustering_profile_baseline.csv"), index=False)
    pd.DataFrame({"component": ["PC1", "PC2"], "explained_variance_ratio": np.round(pca.explained_variance_ratio_, 4)}).to_csv(
        path("outputs", "tables", "clustering_pca_variance.csv"), index=False
    )
    return baseline_table, sweep, profile, best_k, X.shape[1]