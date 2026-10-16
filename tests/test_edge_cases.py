"""Phase 2 edge-case tests on the Predictor (requires trained artifacts: run `python -m src.train` first)."""
import pytest

from src.inference import Predictor


@pytest.fixture(scope="module")
def P():
    return Predictor()


def base(**kw):
    d = dict(employee_id="E0001", department="Finance", expense_date="2025-01-10", merchant="Swiggy",
             description="team lunch", category="Meals", amount=620.0, payment_method="UPI")
    d.update(kw)
    return d


def test_normal_claim_is_approved(P):
    r = P.analyze([base()])[0]
    assert r["risk_tier"] == "approve" and r["predicted_category"] == "Meals"


def test_new_employee_has_no_history_and_does_not_crash(P):
    r = P.analyze([base(employee_id="E9999")])[0]
    assert any(x["code"] == "NO_EMPLOYEE_HISTORY" for x in r["reason_codes"])


def test_empty_description(P):
    r = P.analyze([base(description="")])[0]
    assert any(x["code"] == "EMPTY_DESCRIPTION" for x in r["reason_codes"]) and 0 <= r["anomaly_score"] <= 1


def test_huge_amount_is_flagged(P):
    r = P.analyze([base(amount=5_000_000.0)])[0]
    assert r["risk_tier"] in ("review", "flag")
    assert any(x["code"] == "AMOUNT_HIGH_FOR_CATEGORY" for x in r["reason_codes"])


def test_unseen_merchant_and_category(P):
    r = P.analyze([base(merchant="Totally New Vendor", category="Pet Supplies")])[0]
    codes = {x["code"] for x in r["reason_codes"]}
    assert {"UNSEEN_MERCHANT", "UNRECOGNISED_CATEGORY"} <= codes


def test_missing_category_is_allowed(P):
    d = base(); d.pop("category")
    r = P.analyze([d])[0]
    assert r["claimed_category"] is None


def test_duplicate_within_batch_is_detected(P):
    a = base(employee_id="E0007", merchant="Starbucks", amount=1234.5, description="coffee with team")
    r = P.analyze([a, dict(a)])
    assert any(x["code"] == "POSSIBLE_DUPLICATE" for x in r[1]["reason_codes"]) and r[1]["risk_tier"] != "approve"
    assert not any(x["code"] == "POSSIBLE_DUPLICATE" for x in r[0]["reason_codes"])


def test_category_mismatch_is_detected(P):
    r = P.analyze([base(merchant="Zomato", description="dinner with Acme", category="Software & Subscriptions", amount=1800.0)])[0]
    assert any(x["code"] == "CATEGORY_DESCRIPTION_MISMATCH" for x in r["reason_codes"])


def test_categorize_returns_top3(P):
    r = P.categorize([{"description": "flight to Mumbai", "merchant": "IndiGo"}])[0]
    assert r["predicted_category"] == "Travel" and len(r["top_3"]) == 3
