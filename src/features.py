"""Feature engineering shared by training and inference (single code path).

History features look only at STRICTLY EARLIER rows (ordered by expense_date, then row order) within a 90-day window,
so the same function gives identical values whether it sees the whole dataset (training) or only a 90-day history seed
plus the new claims (inference).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C

VAGUE_TEXTS = {"misc", "client meeting", "expense", "reimbursement", "as discussed", "office", "business expense",
               "monthly", "refund pending", "other", "n/a", "na", "-", "."}
HIST_COLS = ["emp_n_90d", "emp_n_7d", "emp_mean_ratio_90d", "emp_has_history", "dup_same_merchant_amt_7d",
             "dup_same_amt_any_merchant_7d", "same_day_n", "same_day_sum", "same_day_merchant_n", "same_day_merchant_sum"]
CAT_COLS = [f"cat_{i}" for i in range(len(C.CATEGORIES))]
FEATURES = (["log_amount", "amount_over_cat_median", "amount_over_cat_p95"] + HIST_COLS +
            ["just_below_threshold", "split_sum_over_threshold",
             "dow", "is_weekend", "month", "day_of_month", "is_month_end", "weekend_business_cat",
             "desc_len", "desc_empty", "desc_vague",
             "claimed_present", "p_claimed_cat", "pred_conf", "cat_mismatch"] + CAT_COLS)


def normalise_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.lower().str.strip()


def make_text(description: pd.Series, merchant: pd.Series) -> pd.Series:
    """Text fed to Model A: description tokens plus one merchant token."""
    m = merchant.fillna("").astype(str).str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True)
    return normalise_text(description) + " mrc_" + m


def fit_stats(train: pd.DataFrame) -> dict:
    """Category amount statistics learned from TRAIN rows only (labels are not used)."""
    g = train.groupby("category")["amount"]
    return {"cat_median": g.median().to_dict(), "cat_p95": g.quantile(0.95).to_dict(),
            "global_median": float(train["amount"].median()), "global_p95": float(train["amount"].quantile(0.95))}


def history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Employee-history features from strictly earlier rows. Needs employee_id, expense_date, merchant, amount.
    An optional `_ord` column fixes the tie-break order for rows on the same date (default: row position)."""
    d = df[["employee_id", "expense_date", "merchant", "amount"]].copy().reset_index(drop=True)
    d["_pos"] = np.arange(len(d))
    d["_o"] = df["_ord"].to_numpy() if "_ord" in df.columns else d["_pos"].to_numpy()
    d = d.sort_values(["expense_date", "_o"], kind="mergesort").reset_index(drop=True)
    day = ((d["expense_date"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(days=1)).to_numpy().astype(np.int64)
    amt = d["amount"].to_numpy(dtype=float)
    mer = pd.factorize(d["merchant"])[0]
    out = np.zeros((len(d), len(HIST_COLS)))
    for _, idx in d.groupby("employee_id").indices.items():
        dd, a, m = day[idx], amt[idx], mer[idx]
        cs = np.concatenate([[0.0], np.cumsum(a)])
        for i in range(len(idx)):
            lo90 = np.searchsorted(dd, dd[i] - C.HISTORY_DAYS, "left")
            lo7 = np.searchsorted(dd, dd[i] - C.DUP_WINDOW_DAYS, "left")
            lo0 = np.searchsorted(dd, dd[i], "left")
            n90 = i - lo90
            s90 = cs[i] - cs[lo90]
            w_m, w_a = m[lo7:i], a[lo7:i]
            same_amt = np.abs(w_a - a[i]) < 0.005
            sd_m = m[lo0:i] == m[i]
            out[idx[i]] = [
                n90, i - lo7, (a[i] / (s90 / n90)) if n90 > 0 else 1.0, float(n90 > 0),
                np.sum((w_m == m[i]) & same_amt), np.sum(same_amt),
                i - lo0, a[lo0:i].sum() + a[i], sd_m.sum(), a[lo0:i][sd_m].sum() + a[i],
            ]
    res = pd.DataFrame(out, columns=HIST_COLS)
    res.index = d["_pos"].to_numpy()
    res = res.sort_index()
    res.index = df.index
    return res


def category_model_features(df: pd.DataFrame, proba: np.ndarray, classes: list[str]) -> pd.DataFrame:
    """Features derived from Model A's class probabilities for each row's description (+merchant)."""
    cls_idx = {c: i for i, c in enumerate(classes)}
    claimed = df["category"].map(cls_idx) if "category" in df else pd.Series(np.nan, index=df.index)
    claimed = claimed.fillna(-1).to_numpy().astype(int)
    pred = proba.argmax(1)
    conf = proba.max(1)
    present = claimed >= 0
    p_claimed = np.where(present, proba[np.arange(len(proba)), np.clip(claimed, 0, None)], conf)
    mismatch = present & (pred != claimed) & (conf >= 0.5)
    return pd.DataFrame({"claimed_present": present.astype(float), "p_claimed_cat": p_claimed, "pred_conf": conf,
                         "cat_mismatch": mismatch.astype(float)}, index=df.index)


def assemble_features(df: pd.DataFrame, hist: pd.DataFrame, stats: dict, proba: np.ndarray, classes: list[str]) -> pd.DataFrame:
    """Build the final model-B feature matrix (columns = FEATURES)."""
    f = pd.DataFrame(index=df.index)
    amount = df["amount"].astype(float)
    cat = df["category"] if "category" in df else pd.Series("", index=df.index)
    known = cat.isin(C.CATEGORIES)
    med = cat.map(stats["cat_median"]).where(known).fillna(stats["global_median"])
    p95 = cat.map(stats["cat_p95"]).where(known).fillna(stats["global_p95"])
    f["log_amount"] = np.log1p(amount)
    f["amount_over_cat_median"] = amount / med
    f["amount_over_cat_p95"] = amount / p95
    for c in HIST_COLS:
        f[c] = hist[c].to_numpy()
    f["just_below_threshold"] = ((amount >= 0.8 * C.APPROVAL_THRESHOLD) & (amount < C.APPROVAL_THRESHOLD)).astype(float)
    f["split_sum_over_threshold"] = ((hist["same_day_merchant_sum"] > C.APPROVAL_THRESHOLD) & (hist["same_day_merchant_n"] >= 1)
                                     & (amount < C.APPROVAL_THRESHOLD)).astype(float).to_numpy()
    dt = df["expense_date"]
    f["dow"] = dt.dt.dayofweek.astype(float)
    f["is_weekend"] = (dt.dt.dayofweek >= 5).astype(float)
    f["month"] = dt.dt.month.astype(float)
    f["day_of_month"] = dt.dt.day.astype(float)
    f["is_month_end"] = (dt.dt.day >= 28).astype(float)
    f["weekend_business_cat"] = f["is_weekend"] * cat.isin(C.BUSINESS_CATEGORIES).astype(float)
    desc = normalise_text(df["description"]) if "description" in df else pd.Series("", index=df.index)
    f["desc_len"] = desc.str.len().astype(float)
    f["desc_empty"] = (desc == "").astype(float)
    f["desc_vague"] = desc.isin(VAGUE_TEXTS).astype(float)
    cm = category_model_features(df, proba, classes)
    cm.loc[~known, ["claimed_present", "cat_mismatch"]] = 0.0     # unrecognised category: no mismatch claim
    for c in cm.columns:
        f[c] = cm[c]
    for i, name in enumerate(C.CATEGORIES):
        f[f"cat_{i}"] = (cat == name).astype(float)
    return f[FEATURES].astype(float)
