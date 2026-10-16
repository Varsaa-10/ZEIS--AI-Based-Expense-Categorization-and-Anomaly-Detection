"""Chronological split and split-integrity checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C


def assign_split(df: pd.DataFrame) -> pd.Series:
    d = df["expense_date"]
    s = np.where(d < pd.Timestamp(C.VAL_START), "train", np.where(d < pd.Timestamp(C.TEST_START), "validation", "test"))
    return pd.Series(s, index=df.index, name="split")


def check_split(df: pd.DataFrame) -> dict:
    """Return named boolean checks; all must be True."""
    sp = assign_split(df)
    rng = {k: (df.loc[sp == k, "expense_date"].min(), df.loc[sp == k, "expense_date"].max()) for k in ["train", "validation", "test"]}
    res = {
        "train_before_validation": bool(rng["train"][1] < rng["validation"][0]),
        "validation_before_test": bool(rng["validation"][1] < rng["test"][0]),
    }
    for col in ["duplicate_group_id", "split_group_id"]:
        g = df[df[col] != ""].assign(split=sp)
        res[f"no_{col}_straddles_splits"] = bool((g.groupby(col)["split"].nunique() <= 1).all())
    res["all_rows_assigned"] = bool(sp.isin(["train", "validation", "test"]).all())
    return res
