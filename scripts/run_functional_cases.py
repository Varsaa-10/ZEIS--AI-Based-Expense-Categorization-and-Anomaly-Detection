"""Functional test cases with expected vs observed results (executed against the app via TestClient)."""
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app
from src import config as C

def claim(**kw):
    d = dict(employee_id="E0042", department="Sales", expense_date="2025-01-10", merchant="Swiggy", description="team lunch",
             category="Meals", amount=620.0, payment_method="UPI"); d.update(kw); return d

def has(code):
    return lambda j: any(x["code"] == code for x in j["reason_codes"])

cases = [
 ("TC01", "GET /health returns ok", "GET", "/health", None, 200, lambda j: j["status"] == "ok", "status = ok"),
 ("TC02", "Categorize clear travel text", "POST", "/categorize", {"description": "flight to Mumbai", "merchant": "IndiGo"}, 200, lambda j: j["predicted_category"] == "Travel", "category = Travel"),
 ("TC03", "Normal meal claim", "POST", "/analyze-expense", claim(), 200, lambda j: j["risk_tier"] == "approve", "tier = approve"),
 ("TC04", "Huge amount", "POST", "/analyze-expense", claim(amount=5_000_000), 200, lambda j: j["risk_tier"] in ("review", "flag") and has("AMOUNT_HIGH_FOR_CATEGORY")(j), "tier review/flag + AMOUNT_HIGH_FOR_CATEGORY"),
 ("TC05", "Category/description mismatch", "POST", "/analyze-expense", claim(merchant="Zomato", description="dinner with Acme", category="Software & Subscriptions", amount=1800), 200, has("CATEGORY_DESCRIPTION_MISMATCH"), "CATEGORY_DESCRIPTION_MISMATCH present"),
 ("TC06", "Unseen merchant + unknown category + new employee", "POST", "/analyze-expense", claim(employee_id="NEW-1", merchant="Brand New Vendor", category="Gadgets"), 200, lambda j: all(has(c)(j) for c in ("UNSEEN_MERCHANT", "UNRECOGNISED_CATEGORY", "NO_EMPLOYEE_HISTORY")), "three info codes present"),
 ("TC07", "Empty description", "POST", "/analyze-expense", claim(description=""), 200, has("EMPTY_DESCRIPTION"), "EMPTY_DESCRIPTION present"),
 ("TC08", "Amount = 0", "POST", "/analyze-expense", claim(amount=0), 422, None, "validation error"),
 ("TC09", "Negative amount", "POST", "/analyze-expense", claim(amount=-10), 422, None, "validation error"),
 ("TC10", "Amount above maximum (10,000,000.01)", "POST", "/analyze-expense", claim(amount=10_000_000.01), 422, None, "validation error"),
 ("TC11", "Missing merchant", "POST", "/analyze-expense", {k: v for k, v in claim().items() if k != "merchant"}, 422, None, "validation error"),
 ("TC12", "Invalid date", "POST", "/analyze-expense", claim(expense_date="31-31-2025"), 422, None, "validation error"),
 ("TC13", "Batch of 100", "POST", "/analyze-batch", {"expenses": [claim()] * 100}, 200, lambda j: j["count"] == 100, "count = 100"),
 ("TC14", "Batch of 101 (oversized)", "POST", "/analyze-batch", {"expenses": [claim()] * 101}, 422, None, "validation error"),
 ("TC15", "Duplicate pair inside a batch", "POST", "/analyze-batch", {"expenses": [claim(employee_id="E0007", merchant="Starbucks", amount=1234.5), claim(employee_id="E0007", merchant="Starbucks", amount=1234.5)]}, 200, lambda j: has("POSSIBLE_DUPLICATE")(j["results"][1]), "2nd claim has POSSIBLE_DUPLICATE"),
]
rows = []
with TestClient(app) as c:
    for tid, name, m, path, body, exp_status, check, exp_text in cases:
        r = c.request(m, path, json=body)
        ok = r.status_code == exp_status and (check is None or check(r.json()))
        obs = f"HTTP {r.status_code}" + ("" if check is None else "; " + ("expected condition met" if ok else "expected condition NOT met"))
        rows.append(dict(id=tid, case=name, expected=f"HTTP {exp_status}" + ("" if check is None else f"; {exp_text}"), observed=obs, result="PASS" if ok else "FAIL"))
out = pd.DataFrame(rows); out.to_csv(C.EVIDENCE_DIR / "phase3" / "functional_cases.csv", index=False)
print(out.to_string(index=False)); raise SystemExit(0 if (out.result == "PASS").all() else 1)
