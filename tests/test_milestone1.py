import numpy as np
import pytest

from src.data.loader import EXPECTED_COLUMNS, clean_data, load_raw, verify_h2
from src.data.splitter import get_test_data, get_train_data, make_split
from src.features.engineering import add_features
from src.features.selection import DECISIONS, get_features, source_columns
from src.preprocessing.pipelines import build_supervised_preprocessor


@pytest.fixture(scope="module")
def raw():
    try:
        return load_raw()
    except Exception as error:
        pytest.skip("H2.csv is missing and could not be downloaded: %s" % error)


@pytest.fixture(scope="module")
def clean(raw):
    cleaned, _ = clean_data(raw)
    return add_features(cleaned)


def test_schema_checks_pass(raw):
    assert verify_h2(raw)["passed"].all()


def test_cleaning_removes_invalid_rows(clean):
    assert not clean.drop(columns=["ArrivalDayOfWeek", "HasCompany"]).duplicated().any()
    assert ((clean["Adults"] + clean["Children"].fillna(0) + clean["Babies"]) > 0).all()
    assert ((clean["StaysInWeekendNights"] + clean["StaysInWeekNights"]) > 0).all()
    assert (clean["ADR"] <= 1000).all()


def test_text_columns_are_stripped(clean):
    assert clean["Meal"].isin(["BB", "HB", "FB", "SC"]).all()
    assert not clean["Agent"].astype(str).str.contains(" ").any()


def test_split_has_no_overlap(clean):
    train_idx, test_idx = make_split(clean)
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(train_idx) + len(test_idx) == len(clean)
    assert abs(len(test_idx) / len(clean) - 0.2) < 0.001


def test_split_is_stratified_and_reproducible(clean):
    a_train, a_test = make_split(clean)
    b_train, b_test = make_split(clean)
    assert np.array_equal(a_train, b_train)
    assert np.array_equal(a_test, b_test)
    gap = abs(clean.iloc[a_train]["IsCanceled"].mean() - clean.iloc[a_test]["IsCanceled"].mean())
    assert gap < 0.005


def test_regression_rows_have_positive_adr(clean):
    assert (get_train_data(clean, "regression")["ADR"] > 0).all()
    assert (get_test_data(clean, "regression")["ADR"] > 0).all()


def test_feature_sets_match_decision_table(clean):
    assert sorted(c for c, *_ in DECISIONS) == sorted(EXPECTED_COLUMNS)
    for task, position in [("classification", 1), ("regression", 2)]:
        numeric, categorical = get_features(task)
        used = source_columns(numeric + categorical)
        expected = {row[0] for row in DECISIONS if row[position] == "use"}
        assert used == expected


def test_leaky_columns_are_not_features():
    banned = {"IsCanceled", "ReservationStatus", "ReservationStatusDate", "AssignedRoomType", "BookingChanges", "RequiredCarParkingSpaces", "TotalOfSpecialRequests"}
    for task in ["classification", "regression"]:
        numeric, categorical = get_features(task)
        assert banned.isdisjoint(source_columns(numeric + categorical))
    assert "ADR" not in sum(get_features("regression"), [])


def test_preprocessor_learns_statistics_from_train_only(clean):
    numeric, categorical = get_features("classification")
    train = get_train_data(clean, "classification")
    prep = build_supervised_preprocessor(numeric, categorical).fit(train[numeric + categorical])
    scaler = prep.named_transformers_["num"].named_steps["scaler"]
    assert np.isclose(scaler.mean_[numeric.index("LeadTime")], train["LeadTime"].mean())
    assert not np.isclose(scaler.mean_[numeric.index("LeadTime")], clean["LeadTime"].mean(), atol=1e-6)


def test_pipeline_handles_unseen_category_and_missing_values(clean):
    numeric, categorical = get_features("classification")
    train = get_train_data(clean, "classification")
    prep = build_supervised_preprocessor(numeric, categorical).fit(train[numeric + categorical])
    row = train[numeric + categorical].head(1).copy()
    row["Country"] = "ZZZ"
    row["Children"] = np.nan
    out = prep.transform(row)
    assert out.shape[0] == 1
    assert not np.isnan(out).any()


def test_only_h2_is_used(raw):
    from src.config import load_config
    cfg = load_config()
    assert cfg["data"]["raw_path"].endswith("H2.csv")
    assert len(raw) == 79330
    assert raw["ArrivalDateYear"].min() == 2015