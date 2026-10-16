"""Schema / dtype / missing / uniqueness / range / label-distribution validation for the generated dataset."""
from __future__ import annotations

import json

import pandas as pd

from src import config as C
from src.data_gen import load
from src.split import assign_split, check_split

REQUIRED = ["expense_id", "employee_id", "department", "expense_date", "merchant", "description", "category", "amount", "payment_method"]


def validate(df: pd.DataFrame) -> dict:
    checks: dict[str, bool] = {}
    info: dict = {}
    checks["required_columns_present"] = all(c in df.columns for c in REQUIRED)
    checks["row_count_in_20k_50k"] = 20000 <= len(df) <= 50000
    checks["expense_id_unique"] = bool(df["expense_id"].is_unique)
    checks["expense_date_is_datetime"] = bool(pd.api.types.is_datetime64_any_dtype(df["expense_date"]))
    checks["amount_is_numeric"] = bool(pd.api.types.is_numeric_dtype(df["amount"]))
    for c in ["employee_id", "department", "expense_date", "merchant", "category", "amount", "payment_method"]:
        checks[f"no_missing_{c}"] = bool(df[c].notna().all() and (df[c].astype(str) != "").all())
    info["empty_description_rate"] = float((df["description"] == "").mean())
    checks["empty_description_rate_below_3pct"] = info["empty_description_rate"] < 0.03
    checks["amount_positive"] = bool((df["amount"] > 0).all())
    checks["amount_below_max"] = bool((df["amount"] <= C.MAX_AMOUNT).all())
    checks["dates_in_range"] = bool(df["expense_date"].between(C.START_DATE, C.END_DATE).all())
    checks["categories_valid"] = bool(df["category"].isin(C.CATEGORIES).all())
    checks["departments_valid"] = bool(df["department"].isin(C.DEPARTMENTS).all())
    checks["payment_methods_valid"] = bool(df["payment_method"].isin(C.PAYMENT_METHODS).all())
    checks["anomaly_label_binary"] = bool(df["is_anomaly"].isin([0, 1]).all())
    checks["anomaly_type_consistent"] = bool(((df["is_anomaly"] == 0) == (df["anomaly_type"] == "none")).all())
    # uniqueness of business content: every identical (employee,date,merchant,amount,description) pair must be a labelled duplicate
    key = ["employee_id", "expense_date", "merchant", "amount", "description"]
    dups = df[df.duplicated(key, keep="first")]
    info["exact_repeat_rows"] = int(len(dups))
    info["exact_repeat_rows_labelled_duplicate"] = int((dups["anomaly_type"] == "duplicate_claim").sum())
    info["label_distribution"] = df["anomaly_type"].value_counts().to_dict()
    info["anomaly_rate"] = float(df["is_anomaly"].mean())
    checks["anomaly_rate_between_2_and_10pct"] = 0.02 <= info["anomaly_rate"] <= 0.10
    sp = assign_split(df)
    info["split_sizes"] = sp.value_counts().to_dict()
    info["anomaly_rate_by_split"] = df.groupby(sp)["is_anomaly"].mean().round(4).to_dict()
    info["anomaly_types_by_split"] = df.groupby([sp, "anomaly_type"]).size().unstack(fill_value=0).to_dict("index")
    checks.update({f"split_{k}": v for k, v in check_split(df).items()})
    info["dtypes"] = {k: str(v) for k, v in df.dtypes.items()}
    info["amount_summary"] = df["amount"].describe().round(2).to_dict()
    return {"all_passed": all(checks.values()), "checks": checks, "info": info}


if __name__ == "__main__":
    res = validate(load())
    out = C.EVIDENCE_DIR / "phase1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_report.json").write_text(json.dumps(res, indent=2, default=str))
    for k, v in res["checks"].items():
        print(("PASS " if v else "FAIL ") + k)
    print("ALL PASSED" if res["all_passed"] else "SOME CHECKS FAILED")
    print(json.dumps(res["info"]["label_distribution"]), json.dumps(res["info"]["split_sizes"]))
    print("anomaly rate by split:", res["info"]["anomaly_rate_by_split"])
    raise SystemExit(0 if res["all_passed"] else 1)
