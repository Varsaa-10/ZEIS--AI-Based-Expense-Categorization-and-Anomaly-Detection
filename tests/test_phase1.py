"""Phase 1 tests: reproducibility, validation, split integrity, leakage-free history features."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.data_gen import generate, df_hash, load
from src.features import history_features, FEATURES, assemble_features, fit_stats
from src.split import assign_split, check_split
from src.validate import validate


@pytest.fixture(scope="module")
def df():
    return load()


def test_generator_is_reproducible():
    a = generate(6000, 7)
    b = generate(6000, 7)
    assert df_hash(a) == df_hash(b)
    assert df_hash(a) != df_hash(generate(6000, 8))


def test_validation_passes(df):
    res = validate(df)
    failed = [k for k, v in res["checks"].items() if not v]
    assert not failed, failed


def test_split_has_no_overlap_or_straddling_groups(df):
    res = check_split(df)
    assert all(res.values()), res
    sp = assign_split(df)
    assert df[sp == "train"].expense_date.max() < df[sp == "validation"].expense_date.min()
    assert df[sp == "validation"].expense_date.max() < df[sp == "test"].expense_date.min()


def test_reserved_merchants_only_in_later_period(df):
    sp = assign_split(df)
    assert (sp[df.merchant_reserved == 1] != "train").all()
    assert (sp[df.phrasing_reserved == 1] != "train").all()


def test_history_features_use_only_earlier_rows(df):
    sub = df[df.expense_date.between("2024-01-01", "2024-03-31")].copy().reset_index(drop=True)
    base = history_features(sub)
    cut = sub.expense_date.quantile(0.5)
    tampered = sub.copy()
    later = tampered.expense_date > cut
    tampered.loc[later, "amount"] = tampered.loc[later, "amount"] * 7 + 13   # change the FUTURE
    tampered.loc[later, "merchant"] = "ZZZ"
    new = history_features(tampered)
    earlier = (sub.expense_date < cut).to_numpy()
    pd.testing.assert_frame_equal(base[earlier], new[earlier])


def test_history_features_window_equivalence(df):
    """Features from a 90-day seed + new rows equal the features computed on the full dataset."""
    full = history_features(df)
    target = df[df.expense_date >= "2024-12-20"].sample(40, random_state=1)
    for emp, grp in target.groupby("employee_id"):
        d0 = grp.expense_date.min()
        hist = df[(df.employee_id == emp) & (df.expense_date >= d0 - pd.Timedelta(days=C.HISTORY_DAYS)) & (~df.index.isin(grp.index))
                  & (df.expense_date <= grp.expense_date.max())]
        comb = pd.concat([hist.assign(_ord=hist.index), grp.assign(_ord=grp.index)])
        comb = comb.sort_values(["expense_date", "_ord"]).reset_index(drop=True)
        # keep only history rows that are earlier than each target row in global order (rows between targets stay)
        got = history_features(comb.assign(_ord=comb["_ord"]))
        for gi in grp.index:
            pos = comb.index[comb["_ord"] == gi][0]
            np.testing.assert_allclose(got.loc[pos].to_numpy(), full.loc[gi].to_numpy())


def test_feature_matrix_shape_and_no_label_columns(df):
    small = df.iloc[:500].copy()
    stats = fit_stats(small)
    classes = C.CATEGORIES
    proba = np.full((len(small), len(classes)), 1 / len(classes))
    f = assemble_features(small, history_features(small), stats, proba, classes)
    assert list(f.columns) == FEATURES and not f.isna().any().any()
    for banned in ["is_anomaly", "anomaly_type", "duplicate_group_id", "split_group_id", "merchant_reserved", "phrasing_reserved"]:
        assert banned not in FEATURES
