"""Latency over real HTTP: 100 sequential requests per endpoint against a running server (default http://127.0.0.1:8000).
Method: 5 warm-up calls, then 100 timed calls with time.perf_counter around httpx client.post/get; mean and p95 in ms."""
import sys, time, numpy as np, pandas as pd, httpx
from src import config as C

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
claim = dict(employee_id="E0042", department="Sales", expense_date="2025-01-10", merchant="Swiggy", description="team lunch",
             category="Meals", amount=620.0, payment_method="UPI")
cases = [("GET /health", "GET", "/health", None), ("POST /categorize", "POST", "/categorize", {"description": "flight to Mumbai", "merchant": "IndiGo"}),
         ("POST /analyze-expense", "POST", "/analyze-expense", claim),
         ("POST /analyze-batch (10 claims)", "POST", "/analyze-batch", {"expenses": [dict(claim, amount=100.0 + i) for i in range(10)]})]
rows = []
with httpx.Client(base_url=base, timeout=30) as c:
    for name, m, path, body in cases:
        for _ in range(5):
            c.request(m, path, json=body)
        t = []
        for _ in range(100):
            s = time.perf_counter(); r = c.request(m, path, json=body); t.append((time.perf_counter() - s) * 1000); assert r.status_code == 200
        rows.append(dict(endpoint=name, requests=100, mean_ms=np.mean(t), p95_ms=np.percentile(t, 95)))
out = pd.DataFrame(rows).round(2)
print(out.to_string(index=False)); out.to_csv(C.EVIDENCE_DIR / "phase3" / "latency.csv", index=False)
