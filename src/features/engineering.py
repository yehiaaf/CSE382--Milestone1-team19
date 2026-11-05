import pandas as pd

from src.data.loader import arrival_dates


def add_features(df):
    out = df.copy()
    out["ArrivalDayOfWeek"] = arrival_dates(out).dt.day_of_week
    out["HasCompany"] = out["Company"].notna().astype(int)
    return out


def add_status_offset(df):
    out = df.copy()
    status_dates = pd.to_datetime(out["ReservationStatusDate"])
    out["StatusDateOffset"] = (status_dates - arrival_dates(out)).dt.days
    return out