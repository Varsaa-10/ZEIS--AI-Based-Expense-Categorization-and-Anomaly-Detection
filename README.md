# ZEIS – Zubera Expense Intelligence Service

AI-based **expense categorization and anomaly detection** (FastAPI). Trained on a **synthetic** dataset – no company data, no live
integration. Scores are model scores, **not calibrated probabilities**; output is **never proof of fraud** and never authorises reimbursement.

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000                           # pre-trained models are in models/1.0.0
# open http://localhost:8000/docs  (Swagger UI)   and   http://localhost:8000/health
```
Python 3.10+ (developed and tested on 3.12.3).

## Re-train from scratch (regenerates the dataset, ~1–3 minutes)
```bash
python -m src.train        # generates data/generated/expenses.csv if missing, trains, selects on validation, tests once
```
Outputs: `models/1.0.0/` (artifacts + `metadata.json`); result tables are written to `evidence/`.

## Other commands
```bash
python -m src.data_gen          # write the dataset only (prints SHA-256; identical on every run)
python -m src.validate          # schema / range / label / split checks
python -m src.eda               # EDA charts -> evidence/phase1/charts
python -m pytest tests -q       # all tests (needs trained models)
python -m scripts.measure_latency http://127.0.0.1:8000   # while the server runs
```

## API
| endpoint | purpose |
|---|---|
| `GET /health` | status, model version |
| `POST /categorize` | `{description, merchant?}` → predicted category, confidence score, top 3 |
| `POST /analyze-expense` | one claim → category, anomaly score, risk tier (approve/review/flag), reason codes. Query `?record_history=true` adds the claim to in-memory history |
| `POST /analyze-batch` | `{expenses:[…]}` (1–100 claims; claims see earlier claims in the batch) |

```bash
curl -s -X POST localhost:8000/analyze-expense -H 'Content-Type: application/json' -d '{
  "employee_id":"E0042","department":"Sales","expense_date":"2025-01-10","merchant":"Zomato",
  "description":"dinner with Acme","category":"Software & Subscriptions","amount":250000}'
```
Required: `employee_id, expense_date (YYYY-MM-DD), merchant, amount (0 < amount ≤ 10,000,000)`. Optional: `department, description, category, payment_method`.
Invalid input → HTTP 422.

## Docker
```bash
docker build -t zeis:1.0.0 .
docker run --rm -p 8000:8000 zeis:1.0.0
curl localhost:8000/health
```

## Layout
`src/` data generation, validation, features, training, inference · `app/` FastAPI · `models/` pre-trained artifacts · `tests/` · `scripts/` latency and functional checks.

## Reproducibility
Seed 42 (`src/config.py`), pinned `requirements.txt`, dataset SHA-256 stored in `models/1.0.0/metadata.json`. The full generated dataset is not shipped (regenerable).

## Limitations
Synthetic anomalies follow simple rules that the features mirror, so reported scores are optimistic. A supervised model detects only anomaly types it was trained on
(leave-one-type-out recall collapses).
