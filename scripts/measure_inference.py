"""Measure in-process inference time of Predictor (no HTTP). Output: evidence/phase2/inference_timing.csv"""
import time, numpy as np, pandas as pd
from src import config as C
from src.data_gen import load
from src.inference import Predictor

P = Predictor()
df = load().tail(2000)
items = [dict(employee_id=r.employee_id, department=r.department, expense_date=str(r.expense_date.date()), merchant=r.merchant,
              description=r.description, category=r.category, amount=float(r.amount), payment_method=r.payment_method)
         for r in df.itertuples()]
rows = []
for name, fn, n in [("analyze single claim", lambda i: P.analyze([items[i]]), 100),
                    ("categorize single", lambda i: P.categorize([items[i]]), 100)]:
    fn(0)
    t = []
    for i in range(n):
        s = time.perf_counter(); fn(i + 1); t.append((time.perf_counter() - s) * 1000)
    rows.append(dict(operation=name, n=n, mean_ms=np.mean(t), p95_ms=np.percentile(t, 95)))
t = []
for k in range(10):
    s = time.perf_counter(); P.analyze(items[k * 100:(k + 1) * 100]); t.append((time.perf_counter() - s) * 1000)
rows.append(dict(operation="analyze batch of 100 (per batch)", n=10, mean_ms=np.mean(t), p95_ms=np.percentile(t, 95)))
out = pd.DataFrame(rows).round(2); print(out.to_string(index=False)); out.to_csv(C.EVIDENCE_DIR / "phase2" / "inference_timing.csv", index=False)
