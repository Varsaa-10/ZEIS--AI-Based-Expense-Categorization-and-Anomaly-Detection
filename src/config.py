"""Central configuration for ZEIS (Zubera Expense Intelligence Service)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GEN_DIR = DATA_DIR / "generated"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = ROOT / "models"
EVIDENCE_DIR = ROOT / "evidence"
DOCS_DIR = ROOT / "docs"

MODEL_VERSION = "1.0.0"
SEED = 42
N_TARGET_ROWS = 30000
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"
VAL_START = "2024-07-01"    # train: START_DATE .. VAL_START-1
TEST_START = "2024-10-01"   # validation: VAL_START .. TEST_START-1 ; test: TEST_START .. END_DATE
APPROVAL_THRESHOLD = 5000.0  # INR; claims above this need manager approval (synthetic policy)
HISTORY_DAYS = 90            # window for all employee-history features
DUP_WINDOW_DAYS = 7
MAX_AMOUNT = 10_000_000.0
MAX_BATCH = 100

CATEGORIES = [
    "Travel", "Accommodation", "Meals", "Client Entertainment",
    "Software & Subscriptions", "Office Supplies", "Training & Courses",
    "Telecom & Utilities", "Local Transport", "Marketing",
]
# Categories that are normally claimed on working days (used for the weekend rule)
BUSINESS_CATEGORIES = [
    "Software & Subscriptions", "Office Supplies", "Training & Courses",
    "Telecom & Utilities", "Marketing",
]
PAYMENT_METHODS = ["Corporate Card", "Personal Card", "UPI", "Cash", "Bank Transfer"]
DEPARTMENTS = ["Engineering", "Sales", "Marketing", "Finance", "Operations", "HR"]
ANOMALY_TYPES = ["duplicate_claim", "large_amount", "split_claim", "weekend_unusual", "category_mismatch"]
