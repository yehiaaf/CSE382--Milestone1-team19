import os
import urllib.request

import numpy as np
import pandas as pd

from src.config import load_config, path

URL = "https://raw.githubusercontent.com/afaq-ahmad/Hotel-booking-demand-Case-Study/master/H2.csv"

MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}

EXPECTED_COLUMNS = [
    "IsCanceled", "LeadTime", "ArrivalDateYear", "ArrivalDateMonth", "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth", "StaysInWeekendNights", "StaysInWeekNights", "Adults", "Children",
    "Babies", "Meal", "Country", "MarketSegment", "DistributionChannel", "IsRepeatedGuest",
    "PreviousCancellations", "PreviousBookingsNotCanceled", "ReservedRoomType", "AssignedRoomType",
    "BookingChanges", "DepositType", "Agent", "Company", "DaysInWaitingList", "CustomerType",
    "ADR", "RequiredCarParkingSpaces", "TotalOfSpecialRequests", "ReservationStatus",
    "ReservationStatusDate",
]


def download_data():
    cfg = load_config()
    target = path(cfg["data"]["raw_path"])
    os.makedirs(os.path.dirname(target), exist_ok=True)
    urllib.request.urlretrieve(URL, target)
    return target


def load_raw():
    cfg = load_config()
    target = path(cfg["data"]["raw_path"])
    if not os.path.exists(target):
        download_data()
    return pd.read_csv(target)


def arrival_dates(df):
    months = df["ArrivalDateMonth"].map(MONTHS)
    return pd.to_datetime({
        "year": df["ArrivalDateYear"],
        "month": months,
        "day": df["ArrivalDateDayOfMonth"],
    })


def verify_h2(df):
    cfg = load_config()
    dates = arrival_dates(df)
    rows = [
        ("number of rows", cfg["data"]["expected_rows"], len(df)),
        ("number of columns", cfg["data"]["expected_columns"], df.shape[1]),
        ("column names match Section 4 labels", True, list(df.columns) == EXPECTED_COLUMNS),
        ("earliest arrival date", "2015-07-01", str(dates.min().date())),
        ("latest arrival date", "2017-08-31", str(dates.max().date())),
        ("IsCanceled only 0/1", True, set(df["IsCanceled"].unique()) == {0, 1}),
    ]
    out = pd.DataFrame(rows, columns=["check", "expected", "found"])
    out["passed"] = out["expected"].astype(str) == out["found"].astype(str)
    return out


def clean_text(df):
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].str.strip()
            out[col] = out[col].replace("NULL", np.nan)
    return out


def clean_data(df, drop_duplicates=None):
    cfg = load_config()
    if drop_duplicates is None:
        drop_duplicates = cfg["data"]["drop_duplicates"]
    out = clean_text(df)
    log = [("raw rows", len(out))]
    if drop_duplicates:
        out = out.drop_duplicates()
        log.append(("after removing exact duplicate rows", len(out)))
    guests = out["Adults"] + out["Children"].fillna(0) + out["Babies"]
    out = out[guests > 0]
    log.append(("after removing bookings with zero guests", len(out)))
    nights = out["StaysInWeekendNights"] + out["StaysInWeekNights"]
    out = out[nights > 0]
    log.append(("after removing bookings with zero nights", len(out)))
    out = out[out["ADR"] <= cfg["data"]["adr_max"]]
    log.append(("after removing ADR above %d" % cfg["data"]["adr_max"], len(out)))
    out = out.reset_index(drop=True)
    return out, pd.DataFrame(log, columns=["step", "rows"])
