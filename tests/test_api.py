"""Phase 3 API tests (requires trained artifacts)."""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def valid(**kw):
    d = dict(employee_id="E0042", department="Sales", expense_date="2025-01-10", merchant="Swiggy", description="team lunch",
             category="Meals", amount=620.0, payment_method="UPI")
    d.update(kw)
    return d


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["models_loaded"] is True


def test_docs_available(client):
    assert client.get("/docs").status_code == 200 and client.get("/openapi.json").status_code == 200


def test_categorize_valid(client):
    r = client.post("/categorize", json={"description": "flight to Mumbai", "merchant": "IndiGo"})
    j = r.json()
    assert r.status_code == 200 and j["predicted_category"] == "Travel" and 0 <= j["category_score"] <= 1 and len(j["top_3"]) == 3


def test_categorize_missing_description(client):
    assert client.post("/categorize", json={"merchant": "IndiGo"}).status_code == 422


def test_analyze_valid_schema(client):
    r = client.post("/analyze-expense", json=valid())
    j = r.json()
    assert r.status_code == 200
    assert set(j) >= {"predicted_category", "category_score", "anomaly_score", "risk_tier", "reason_codes", "model_version"}
    assert j["risk_tier"] in ("approve", "review", "flag") and 0 <= j["anomaly_score"] <= 1


@pytest.mark.parametrize("field", ["employee_id", "expense_date", "merchant", "amount"])
def test_analyze_missing_required_field(client, field):
    d = valid(); d.pop(field)
    assert client.post("/analyze-expense", json=d).status_code == 422


@pytest.mark.parametrize("amount,status", [(0, 422), (-5, 422), (0.01, 200), (10_000_000, 200), (10_000_000.01, 422), ("abc", 422)])
def test_amount_boundaries(client, amount, status):
    assert client.post("/analyze-expense", json=valid(amount=amount)).status_code == status


def test_invalid_date_and_empty_employee(client):
    assert client.post("/analyze-expense", json=valid(expense_date="not-a-date")).status_code == 422
    assert client.post("/analyze-expense", json=valid(employee_id="")).status_code == 422


def test_unseen_category_merchant_and_new_employee(client):
    r = client.post("/analyze-expense", json=valid(employee_id="NEW-001", merchant="Brand New Vendor", category="Gadgets"))
    codes = {x["code"] for x in r.json()["reason_codes"]}
    assert r.status_code == 200 and {"UNSEEN_MERCHANT", "UNRECOGNISED_CATEGORY", "NO_EMPLOYEE_HISTORY"} <= codes


def test_empty_description_and_no_category(client):
    d = valid(description=""); d.pop("category")
    assert client.post("/analyze-expense", json=d).status_code == 200


def test_huge_amount_goes_to_review_or_flag(client):
    r = client.post("/analyze-expense", json=valid(amount=5_000_000))
    assert r.json()["risk_tier"] in ("review", "flag")


def test_batch_valid_and_duplicate_detection(client):
    a = valid(employee_id="E0007", merchant="Starbucks", description="coffee with team", amount=1234.5)
    r = client.post("/analyze-batch", json={"expenses": [a, dict(a), valid()]})
    j = r.json()
    assert r.status_code == 200 and j["count"] == 3 and sum(j["summary"].values()) == 3
    assert any(x["code"] == "POSSIBLE_DUPLICATE" for x in j["results"][1]["reason_codes"])


def test_batch_limits(client):
    assert client.post("/analyze-batch", json={"expenses": [valid()] * 100}).status_code == 200
    assert client.post("/analyze-batch", json={"expenses": [valid()] * 101}).status_code == 422
    assert client.post("/analyze-batch", json={"expenses": []}).status_code == 422
    assert client.post("/analyze-batch", json={}).status_code == 422


def test_record_history_flag(client):
    a = valid(employee_id="E0100", merchant="Uber", description="cab to office", category="Local Transport", amount=777.77,
              expense_date="2025-02-01")
    first = client.post("/analyze-expense?record_history=true", json=a).json()
    second = client.post("/analyze-expense", json=a).json()
    assert not any(x["code"] == "POSSIBLE_DUPLICATE" for x in first["reason_codes"])
    assert any(x["code"] == "POSSIBLE_DUPLICATE" for x in second["reason_codes"])


def test_models_not_loaded_returns_503():
    with TestClient(app) as c:
        saved = app.state.predictor
        app.state.predictor = None
        try:
            assert c.get("/health").status_code == 503
            assert c.post("/categorize", json={"description": "x"}).status_code == 503
        finally:
            app.state.predictor = saved
