import numpy as np
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.features.selection import task_frame


def make_split(df):
    cfg = load_config()
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["seed"],
        stratify=df["IsCanceled"],
    )
    return np.sort(train_idx), np.sort(test_idx)


def get_train_data(df, task):
    train_idx, _ = make_split(df)
    train = task_frame(df.iloc[train_idx], task)
    return train.reset_index(drop=True)


def get_test_data(df, task):
    _, test_idx = make_split(df)
    test = task_frame(df.iloc[test_idx], task)
    return test.reset_index(drop=True)