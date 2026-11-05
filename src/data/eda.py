import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import path
from src.data.loader import MONTHS, clean_text
from src.evaluation.plots import COLORS, save_fig

MONTH_ORDER = list(MONTHS.keys())


def save_table(df, name, index=False):
    df.to_csv(path("outputs", "tables", name), index=index)


def make_tables(raw, clean, log):
    text = clean_text(raw)

    overview = pd.DataFrame([
        ("rows in H2.csv", len(raw)),
        ("columns", raw.shape[1]),
        ("exact duplicate rows", int(text.duplicated().sum())),
        ("duplicate share (%)", round(100 * text.duplicated().mean(), 2)),
        ("rows after cleaning", len(clean)),
        ("canceled share, raw (%)", round(100 * text["IsCanceled"].mean(), 2)),
        ("canceled share, cleaned (%)", round(100 * clean["IsCanceled"].mean(), 2)),
    ], columns=["item", "value"])
    save_table(overview, "eda_overview.csv")

    columns = pd.DataFrame({
        "column": text.columns,
        "dtype": [str(t) for t in text.dtypes],
        "n_unique": [text[c].nunique() for c in text.columns],
        "n_missing": [int(text[c].isna().sum()) for c in text.columns],
        "pct_missing": [round(100 * text[c].isna().mean(), 2) for c in text.columns],
    })
    save_table(columns, "eda_columns.csv")

    numeric_cols = [c for c in clean.columns if pd.api.types.is_numeric_dtype(clean[c]) and c not in ["ArrivalDayOfWeek", "HasCompany"]]
    numeric = clean[numeric_cols].describe().T
    numeric["skew"] = clean[numeric_cols].skew()
    numeric["kurtosis"] = clean[numeric_cols].kurt()
    save_table(numeric.round(3).reset_index().rename(columns={"index": "column"}), "eda_numeric_summary.csv")

    cat_cols = [c for c in clean.columns if not pd.api.types.is_numeric_dtype(clean[c])]
    cat_rows = []
    for c in cat_cols:
        counts = clean[c].value_counts()
        cat_rows.append({"column": c, "n_unique": clean[c].nunique(), "top_value": counts.index[0], "top_share_pct": round(100 * counts.iloc[0] / len(clean), 1)})
    save_table(pd.DataFrame(cat_rows), "eda_categorical_summary.csv")

    guests = text["Adults"] + text["Children"].fillna(0) + text["Babies"]
    nights = text["StaysInWeekendNights"] + text["StaysInWeekNights"]
    invalid = pd.DataFrame([
        ("bookings with zero guests", int((guests == 0).sum())),
        ("bookings with zero nights", int((nights == 0).sum())),
        ("ADR equal to 0", int((text["ADR"] == 0).sum())),
        ("ADR above 1000", int((text["ADR"] > 1000).sum())),
        ("Children missing", int(text["Children"].isna().sum())),
        ("Country missing", int(text["Country"].isna().sum())),
        ("Agent NULL", int(text["Agent"].isna().sum())),
        ("Company NULL", int(text["Company"].isna().sum())),
        ("DistributionChannel Undefined", int((text["DistributionChannel"] == "Undefined").sum())),
        ("Babies greater than Adults", int((text["Babies"] > text["Adults"]).sum())),
    ], columns=["issue", "rows"])
    save_table(invalid, "eda_invalid_values.csv")
    save_table(log, "cleaning_log.csv")
    return text


def fig_missing(text):
    miss = text.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(miss.index, miss.values, color=COLORS[0])
    for i, v in enumerate(miss.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_ylabel("Missing values")
    ax.set_title("Missing values per column (NULL counted as missing)")
    save_fig(fig, "eda_missing_values.png")


def fig_target(text, clean):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, data, title in zip(axes, [text, clean], ["Raw (79,330 rows)", "After cleaning"]):
        shares = data["IsCanceled"].value_counts(normalize=True).sort_index() * 100
        ax.bar(["Not canceled", "Canceled"], shares.values, color=[COLORS[0], COLORS[3]])
        for i, v in enumerate(shares.values):
            ax.text(i, v, "%.1f%%" % v, ha="center", va="bottom")
        ax.set_title(title)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Share of bookings (%)")
    save_fig(fig, "eda_target_balance.png")


def fig_adr(text, clean):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(text["ADR"].clip(upper=500), bins=100, color=COLORS[0])
    axes[0].set_title("ADR distribution (raw, clipped at 500)")
    axes[0].set_xlabel("ADR")
    axes[0].set_ylabel("Bookings")
    positive = clean[clean["ADR"] > 0]
    axes[1].hist(np.log1p(positive["ADR"]), bins=80, color=COLORS[1])
    axes[1].set_title("log(1 + ADR), ADR > 0")
    axes[1].set_xlabel("log(1 + ADR)")
    rooms = sorted(positive["ReservedRoomType"].unique())
    axes[2].boxplot([positive.loc[positive["ReservedRoomType"] == r, "ADR"] for r in rooms], tick_labels=rooms, showfliers=False)
    axes[2].set_title("ADR by reserved room type (ADR > 0)")
    axes[2].set_xlabel("Reserved room type")
    save_fig(fig, "eda_adr.png")


def fig_lead_time(clean):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(clean["LeadTime"], bins=80, color=COLORS[0])
    axes[0].set_title("LeadTime (days)")
    axes[1].hist(np.log1p(clean["LeadTime"]), bins=80, color=COLORS[1])
    axes[1].set_title("log(1 + LeadTime)")
    axes[0].set_ylabel("Bookings")
    save_fig(fig, "eda_lead_time.png")


def fig_cancel_rates(clean):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["MarketSegment", "DepositType", "CustomerType"]):
        rate = clean.groupby(col)["IsCanceled"].agg(["mean", "count"]).sort_values("mean", ascending=False)
        ax.bar(rate.index, rate["mean"] * 100, color=COLORS[3])
        for i, (m, n) in enumerate(zip(rate["mean"], rate["count"])):
            ax.text(i, m * 100, "n=%d" % n, ha="center", va="bottom", fontsize=7)
        ax.set_title("Cancellation rate by %s" % col)
        ax.set_ylabel("Cancellation rate (%)")
        ax.tick_params(axis="x", rotation=40)
    save_fig(fig, "eda_cancel_rates.png")


def fig_seasonality(clean):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    monthly = clean.groupby("ArrivalDateMonth")["IsCanceled"].mean().reindex(MONTH_ORDER) * 100
    axes[0].bar(range(12), monthly.values, color=COLORS[3])
    axes[0].set_xticks(range(12))
    axes[0].set_xticklabels([m[:3] for m in MONTH_ORDER])
    axes[0].set_title("Cancellation rate by arrival month")
    axes[0].set_ylabel("Cancellation rate (%)")
    positive = clean[clean["ADR"] > 0]
    for i, year in enumerate(sorted(positive["ArrivalDateYear"].unique())):
        part = positive[positive["ArrivalDateYear"] == year].groupby("ArrivalDateMonth")["ADR"].mean().reindex(MONTH_ORDER)
        axes[1].plot(range(12), part.values, marker="o", color=COLORS[i], label=str(year))
    axes[1].set_xticks(range(12))
    axes[1].set_xticklabels([m[:3] for m in MONTH_ORDER])
    axes[1].set_title("Mean ADR by arrival month and year")
    axes[1].set_ylabel("Mean ADR")
    axes[1].legend()
    save_fig(fig, "eda_seasonality.png")


def fig_correlation(clean):
    cols = [c for c in clean.columns if pd.api.types.is_numeric_dtype(clean[c]) and c not in ["ArrivalDayOfWeek", "HasCompany"]]
    corr = clean[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, "%.1f" % corr.values[i, j], ha="center", va="center", fontsize=5)
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("Correlation matrix of numeric variables")
    save_fig(fig, "eda_correlation.png")


def fig_skewness(clean):
    cols = [c for c in clean.columns if pd.api.types.is_numeric_dtype(clean[c]) and c not in ["ArrivalDayOfWeek", "HasCompany", "IsCanceled"]]
    skew = clean[cols].skew().sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(skew.index, skew.values, color=COLORS[0])
    ax.set_xlabel("Skewness")
    ax.set_title("Skewness of numeric variables")
    save_fig(fig, "eda_skewness.png")


def run_eda(raw, clean, log):
    text = make_tables(raw, clean, log)
    fig_missing(text)
    fig_target(text, clean)
    fig_adr(text, clean)
    fig_lead_time(clean)
    fig_cancel_rates(clean)
    fig_seasonality(clean)
    fig_correlation(clean)
    fig_skewness(clean)