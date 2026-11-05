from src.config import load_config

NUMERIC_BASE = [
    "LeadTime", "ArrivalDateYear", "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth", "ArrivalDayOfWeek", "StaysInWeekendNights", "StaysInWeekNights",
    "Adults", "Children", "Babies", "IsRepeatedGuest", "PreviousCancellations",
    "PreviousBookingsNotCanceled", "DaysInWaitingList", "HasCompany",
]

CATEGORICAL_BASE = [
    "ArrivalDateMonth", "Meal", "Country", "MarketSegment", "DistributionChannel", "ReservedRoomType",
    "DepositType", "Agent", "CustomerType",
]

CLUSTER_LOG_COLUMNS = ["LeadTime", "StaysInWeekendNights", "StaysInWeekNights", "ADR"]
CLUSTER_PLAIN_COLUMNS = ["Adults", "Children", "IsRepeatedGuest"]
CLUSTER_CATEGORICAL = ["MarketSegment", "CustomerType"]

DERIVED_FROM = {
    "ArrivalDayOfWeek": ["ArrivalDateYear", "ArrivalDateMonth", "ArrivalDateDayOfMonth"],
    "HasCompany": ["Company"],
}

DECISIONS = [
    ("IsCanceled", "target", "exclude", "exclude", "Final outcome. Target of Task B; unknown at the prediction point for Task A and Task C."),
    ("LeadTime", "use", "use", "use", "Known when the booking enters the PMS."),
    ("ArrivalDateYear", "use", "use", "exclude", "Stay date is fixed at booking. Not a segment-defining attribute for clustering."),
    ("ArrivalDateMonth", "use", "use", "exclude", "Stay date is fixed at booking. One-hot encoded so seasonality is not forced to be linear. Not used in clustering."),
    ("ArrivalDateWeekNumber", "use", "use", "exclude", "Stay date is fixed at booking."),
    ("ArrivalDateDayOfMonth", "use", "use", "exclude", "Stay date is fixed at booking."),
    ("StaysInWeekendNights", "use", "use", "use", "Requested stay length is part of the booking."),
    ("StaysInWeekNights", "use", "use", "use", "Requested stay length is part of the booking."),
    ("Adults", "use", "use", "use", "Party size declared at booking."),
    ("Children", "use", "use", "use", "Party size declared at booking."),
    ("Babies", "use", "use", "exclude", "Declared at booking. Almost always zero, so useless for distance-based clustering."),
    ("Meal", "use", "use", "exclude", "Chosen at booking. Low information for segmentation."),
    ("Country", "use", "use", "exclude", "Known at booking. 166 levels would dominate the distance in clustering."),
    ("MarketSegment", "use", "use", "use", "Known at booking."),
    ("DistributionChannel", "use", "use", "exclude", "Known at booking. Overlaps with MarketSegment for clustering."),
    ("IsRepeatedGuest", "use", "use", "use", "Guest history is known before the booking."),
    ("PreviousCancellations", "use", "use", "exclude", "Refers to earlier bookings, not this one. Heavily zero inflated for clustering."),
    ("PreviousBookingsNotCanceled", "use", "use", "exclude", "Refers to earlier bookings, not this one. Heavily zero inflated for clustering."),
    ("ReservedRoomType", "use", "use", "exclude", "Room requested at booking."),
    ("AssignedRoomType", "exclude", "exclude", "exclude", "Assigned at or near check-in. Rarely differs from ReservedRoomType for canceled bookings, so it leaks the outcome."),
    ("BookingChanges", "exclude", "exclude", "exclude", "Counts amendments up to check-in or cancellation, so it keeps growing after the prediction point."),
    ("DepositType", "use", "use", "exclude", "Known at booking. 97% of Non Refund bookings (99.8% before duplicate removal) were canceled, so it would make clusters mirror the outcome."),
    ("Agent", "use", "use", "exclude", "Known at booking. NULL means no agent. 224 levels."),
    ("Company", "use", "use", "exclude", "Known at booking. 95% NULL, so only the HasCompany flag is used."),
    ("DaysInWaitingList", "use", "use", "exclude", "Fixed by the time the booking is confirmed, which is the prediction point."),
    ("CustomerType", "use", "use", "use", "Known at booking."),
    ("ADR", "use", "target", "use", "Quoted rate for the booking. Target of Task A."),
    ("RequiredCarParkingSpaces", "exclude", "exclude", "exclude", "Zero for every canceled booking in the data, so it is recorded only for guests who arrive."),
    ("TotalOfSpecialRequests", "exclude", "exclude", "exclude", "Requests accumulate after booking and are far fewer for canceled bookings."),
    ("ReservationStatus", "exclude", "exclude", "exclude", "Final status. Directly encodes IsCanceled."),
    ("ReservationStatusDate", "exclude", "exclude", "exclude", "Date of the final status. Equals the cancellation date for canceled bookings."),
]


def source_columns(features):
    cols = set()
    for f in features:
        cols.update(DERIVED_FROM.get(f, [f]))
    return cols


def get_features(task):
    if task == "classification":
        return NUMERIC_BASE + ["ADR"], list(CATEGORICAL_BASE)
    if task == "regression":
        return list(NUMERIC_BASE), list(CATEGORICAL_BASE)
    raise ValueError(task)


def get_target(task):
    return {"classification": "IsCanceled", "regression": "ADR"}[task]


def task_frame(df, task):
    if task == "regression":
        cfg = load_config()
        return df[df["ADR"] > cfg["regression"]["adr_min_exclusive"]]
    return df