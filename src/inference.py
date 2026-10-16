"""Inference: loads versioned artifacts once and reuses the training feature code."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config as C
from src.features import assemble_features, history_features, make_text, normalise_text
from src.scoring import AnomalyScorer  # noqa: F401  (needed to unpickle)

SIGNAL, INFO = "signal", "info"


def _reason(code: str, severity: str, message: str) -> dict:
    return {"code": code, "severity": severity, "message": message}


class Predictor:
    """Wraps Model A (categorization), Model B (anomaly scoring), the risk-tier policy and reason codes."""

    def __init__(self, model_dir: Path | str | None = None):
        if model_dir is None:
            version = (C.MODELS_DIR / "LATEST").read_text().strip()
            model_dir = C.MODELS_DIR / version
        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / "metadata.json").read_text())
        self.version = self.meta["model_version"]
        self.cat_model = joblib.load(model_dir / "category_model.joblib")
        self.scorer = joblib.load(model_dir / "anomaly_model.joblib")
        self.classes = self.meta["classes"]
        self.review_t = self.meta["review_threshold"]
        self.flag_t = self.meta["flag_threshold"]
        self.stats = self.meta["stats"]
        self.known_merchants = {m.lower() for m in self.meta["merchants_seen_in_train"]}
        seed = pd.read_csv(model_dir / "history_seed.csv", keep_default_na=False)
        seed["expense_date"] = pd.to_datetime(seed["expense_date"])
        self._hist = {e: g[["employee_id", "expense_date", "merchant", "amount"]].reset_index(drop=True)
                      for e, g in seed.groupby("employee_id")}
        self._lock = threading.Lock()

    # ----------------------------------------------------------------------------------------- categorize
    def categorize(self, items: list[dict]) -> list[dict]:
        df = pd.DataFrame(items)
        if "merchant" not in df:
            df["merchant"] = ""
        proba = self.cat_model.predict_proba(make_text(df["description"], df["merchant"]))
        out = []
        for p in proba:
            order = np.argsort(-p)[:3]
            out.append({"predicted_category": self.classes[order[0]], "category_score": round(float(p[order[0]]), 4),
                        "top_3": [{"category": self.classes[i], "score": round(float(p[i]), 4)} for i in order]})
        return out

    # ----------------------------------------------------------------------------------------- analyze
    def _history_for(self, employees) -> pd.DataFrame:
        with self._lock:
            parts = [self._hist[e] for e in employees if e in self._hist]
        if not parts:
            return pd.DataFrame(columns=["employee_id", "expense_date", "merchant", "amount"]).astype(
                {"amount": float, "expense_date": "datetime64[ns]"})
        return pd.concat(parts, ignore_index=True)

    def analyze(self, items: list[dict], record_history: bool = False) -> list[dict]:
        new = pd.DataFrame(items).reset_index(drop=True)
        new["expense_date"] = pd.to_datetime(new["expense_date"])
        new["description"] = new["description"].fillna("").astype(str)
        new["merchant"] = new["merchant"].fillna("").astype(str)
        new["category"] = new["category"].fillna("").astype(str) if "category" in new else ""
        new["amount"] = new["amount"].astype(float)
        proba = self.cat_model.predict_proba(make_text(new["description"], new["merchant"]))
        hist = self._history_for(new["employee_id"].unique())
        cols = ["employee_id", "expense_date", "merchant", "amount"]
        comb = pd.concat([hist[cols].astype({"expense_date": "datetime64[ns]"}), new[cols].astype({"expense_date": "datetime64[ns]"})],
                         ignore_index=True)
        comb["_ord"] = np.arange(len(comb))
        hf = history_features(comb).iloc[len(hist):].reset_index(drop=True)
        X = assemble_features(new, hf, self.stats, proba, self.classes)
        scores = self.scorer.score(X)
        results = []
        for i in range(len(new)):
            s = float(scores[i])
            tier = "flag" if s >= self.flag_t else "review" if s >= self.review_t else "approve"
            pred = self.classes[int(proba[i].argmax())]
            reasons = self._reasons(new.iloc[i], X.iloc[i], pred, tier)
            results.append({
                "predicted_category": pred, "category_score": round(float(proba[i].max()), 4),
                "claimed_category": new.at[i, "category"] or None, "anomaly_score": round(s, 4), "risk_tier": tier,
                "reason_codes": reasons, "model_version": self.version})
        if record_history:
            with self._lock:
                for e, g in new.groupby("employee_id"):
                    base = self._hist.get(e)
                    add = g[cols]
                    self._hist[e] = pd.concat([base, add], ignore_index=True) if base is not None else add.reset_index(drop=True)
        return results

    # ----------------------------------------------------------------------------------------- reason codes
    def _reasons(self, row: pd.Series, f: pd.Series, predicted: str, tier: str) -> list[dict]:
        r: list[dict] = []
        if f["dup_same_merchant_amt_7d"] >= 1:
            r.append(_reason("POSSIBLE_DUPLICATE", SIGNAL, f"Same employee, merchant and amount already claimed {int(f['dup_same_merchant_amt_7d'])} time(s) in the previous 7 days."))
        if f["amount_over_cat_p95"] >= 1.5:
            r.append(_reason("AMOUNT_HIGH_FOR_CATEGORY", SIGNAL, f"Amount is {f['amount_over_cat_p95']:.1f}x the typical upper range (95th percentile) for this category."))
        if f["split_sum_over_threshold"] == 1:
            r.append(_reason("POSSIBLE_SPLIT_CLAIM", SIGNAL, f"Claims today at this merchant total {f['same_day_merchant_sum']:.0f}, above the approval threshold of {C.APPROVAL_THRESHOLD:.0f}, while each is below it."))
        elif f["just_below_threshold"] == 1:
            r.append(_reason("NEAR_APPROVAL_THRESHOLD", INFO, "Amount is just below the approval threshold."))
        if f["weekend_business_cat"] == 1 and f["amount_over_cat_median"] >= 1.2:
            r.append(_reason("UNUSUAL_WEEKEND_CLAIM", SIGNAL, "Weekend claim in a category normally claimed on working days, with an above-typical amount."))
        if f["cat_mismatch"] == 1:
            r.append(_reason("CATEGORY_DESCRIPTION_MISMATCH", SIGNAL, f"Description and merchant look like '{predicted}' but the claim is filed under '{row['category']}'."))
        if f["emp_has_history"] == 1 and f["emp_mean_ratio_90d"] >= 3:
            r.append(_reason("HIGH_VS_EMPLOYEE_HISTORY", SIGNAL, f"Amount is {f['emp_mean_ratio_90d']:.1f}x this employee's recent average."))
        if f["desc_empty"] == 1:
            r.append(_reason("EMPTY_DESCRIPTION", INFO, "No description supplied; category checks are weaker."))
        elif f["desc_vague"] == 1:
            r.append(_reason("VAGUE_DESCRIPTION", INFO, "Description is vague; category checks are weaker."))
        if f["emp_has_history"] == 0:
            r.append(_reason("NO_EMPLOYEE_HISTORY", INFO, "No recent claims found for this employee, so history-based signals are unavailable."))
        if row["category"] and row["category"] not in C.CATEGORIES:
            r.append(_reason("UNRECOGNISED_CATEGORY", INFO, "Category is not one the model was trained on; category-based signals are neutral."))
        if str(row["merchant"]).strip().lower() not in self.known_merchants:
            r.append(_reason("UNSEEN_MERCHANT", INFO, "Merchant was not seen in training data; text-based category scores are less reliable."))
        if tier != "approve" and not any(x["severity"] == SIGNAL for x in r):
            r.append(_reason("MODEL_SCORE_ONLY", INFO, "Elevated model score without a single dominant rule-based signal."))
        return r
