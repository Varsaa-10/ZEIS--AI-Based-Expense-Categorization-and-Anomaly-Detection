"""Deterministic synthetic expense generator.

Every anomaly has a documented rule. Labels mean
"unusual claim pattern that deserves human review", NOT "proven fraud".
"""
from __future__ import annotations

import hashlib
import sys

import numpy as np
import pandas as pd

from src import config as C

# category: median amount, log-sigma, weekend probability, merchants, reserved merchant, templates, reserved template
CAT = {
    "Travel": dict(med=3000, sg=0.55, wk=0.15,
        m=["IndiGo", "Air India", "IRCTC", "MakeMyTrip", "RedBus", "SpiceJet"], rm="Cleartrip",
        t=["flight to {city}", "train ticket {city}", "bus fare {city} trip", "airfare for {client} visit",
           "travel to {city} for site visit", "client meeting {city} travel"], rt="cab to airport and flight {city}"),
    "Accommodation": dict(med=4500, sg=0.45, wk=0.15,
        m=["Taj Hotels", "OYO", "Marriott", "MakeMyTrip", "Treebo"], rm="Airbnb",
        t=["hotel stay {city}", "{n} nights lodging {city}", "accommodation for {client} workshop",
           "hotel booking {city} conference", "room charges {city}"], rt="guest house stay {city} onsite"),
    "Meals": dict(med=450, sg=0.5, wk=0.2,
        m=["Swiggy", "Zomato", "Starbucks", "Domino's", "Cafe Coffee Day"], rm="Zepto Cafe",
        t=["team lunch", "working dinner", "coffee with team", "lunch during {city} trip", "snacks for team meeting",
           "client meeting lunch"], rt="breakfast order for standup"),
    "Client Entertainment": dict(med=2200, sg=0.5, wk=0.12,
        m=["Taj Hotels", "Zomato", "Marriott", "Barbeque Nation", "The Park"], rm="Hard Rock Cafe",
        t=["dinner with {client}", "client lunch {client}", "client meeting hosting {client}",
           "team lunch with {client} delegates", "hospitality for {client} visit"], rt="networking evening with {client}"),
    "Software & Subscriptions": dict(med=1500, sg=0.6, wk=0.03,
        m=["Google", "Microsoft", "Zoom", "Atlassian", "GitHub", "Amazon Web Services", "Notion"], rm="JetBrains",
        t=["monthly subscription {item}", "software license {item}", "cloud hosting charges", "annual plan {item}",
           "subscription renewal {item}"], rt="IDE licence renewal {item}"),
    "Office Supplies": dict(med=1100, sg=0.6, wk=0.03,
        m=["Amazon", "Flipkart", "Staples", "Reliance Digital", "Croma"], rm="Moglix",
        t=["stationery purchase", "printer cartridges", "office supplies {item}", "keyboard and mouse",
           "whiteboard markers and paper"], rt="desk organiser and cables"),
    "Training & Courses": dict(med=3000, sg=0.6, wk=0.03,
        m=["Udemy", "Coursera", "Amazon", "Flipkart", "Simplilearn"], rm="upGrad",
        t=["online course {item}", "training fee {item}", "certification exam fee", "books for {item} course",
           "workshop registration {item}"], rt="bootcamp seat {item}"),
    "Telecom & Utilities": dict(med=900, sg=0.35, wk=0.03,
        m=["Airtel", "Jio", "BSNL", "ACT Fibernet", "Vodafone Idea"], rm="Tata Play Fiber",
        t=["mobile bill", "internet charges", "broadband bill monthly", "data pack recharge", "phone recharge"],
        rt="wifi bill home office"),
    "Local Transport": dict(med=350, sg=0.6, wk=0.2,
        m=["Uber", "Ola", "Rapido", "Namma Yatri", "Chennai Metro"], rm="BluSmart",
        t=["cab to office", "taxi to client site", "auto fare", "metro card recharge", "cab fare {city}",
           "client meeting cab"], rt="ride to customer premises"),
    "Marketing": dict(med=2500, sg=0.6, wk=0.03,
        m=["Google", "Meta", "LinkedIn", "Canva", "Vistaprint"], rm="Mailchimp",
        t=["ad spend {item}", "campaign credits", "design subscription {item}", "banner printing",
           "sponsored post {item}"], rt="newsletter tool plan {item}"),
}
CITIES = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Pune", "Kolkata", "Chennai", "Ahmedabad", "Kochi"]
CLIENTS = ["Acme", "Globex", "Initech", "Umbrella", "Wayne Corp", "Stark Ltd", "Hooli", "Vandelay"]
ITEMS = ["pro", "team", "basic", "q1", "q2", "q3", "q4", "starter", "plus", "advanced"]
VAGUE = ["misc", "client meeting", "expense", "reimbursement", "as discussed", "office", "business expense",
         "monthly", "refund pending", "other"]
ABBR = {"travel": "trvl", "hotel": "htl", "subscription": "subs", "stationery": "stat.", "internet": "net",
        "lunch": "lnch", "dinner": "dnr", "client": "clnt", "ticket": "tkt", "charges": "chrg",
        "monthly": "mthly", "training": "trng", "course": "crs"}
DEPT_CAT_W = {  # relative category weights per department (order of C.CATEGORIES)
    "Engineering": [1, 1, 2, 0.3, 4, 2, 3, 2, 2, 0.2],
    "Sales": [4, 3, 3, 4, 1, 1, 0.5, 2, 3, 1],
    "Marketing": [1.5, 1, 2, 1.5, 2, 1, 1, 1.5, 1.5, 5],
    "Finance": [1, 0.7, 1.5, 0.5, 2, 2, 1.5, 2, 1, 0.3],
    "Operations": [2, 1.5, 2, 0.5, 1, 3, 0.5, 2, 3, 0.3],
    "HR": [1, 0.7, 2, 0.5, 1, 2, 3, 1.5, 1, 0.5],
}
PAY_W = {"Travel": [.5, .3, .1, 0, .1], "Meals": [.2, .3, .35, .15, 0], "Local Transport": [.1, .2, .6, .1, 0]}
DEFAULT_PAY = [.4, .3, .15, .02, .13]


def _typo(s: str, rng) -> str:
    words = s.split(" ")
    j = int(rng.integers(0, len(words)))
    w = words[j]
    if len(w) >= 4:
        i = int(rng.integers(1, len(w) - 1))
        op = int(rng.integers(0, 3))
        if op == 0:
            w = w[:i] + w[i + 1:]
        elif op == 1:
            w = w[:i] + w[i] + w[i:]
        else:
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
    words[j] = w
    return " ".join(words)


def _describe(cat: str, date: pd.Timestamp, rng, force_template: str | None = None) -> tuple[str, bool]:
    """Return (description, used_reserved_phrasing)."""
    spec = CAT[cat]
    u = rng.random()
    if u < 0.01:
        return "", False
    if u < 0.07:
        return str(rng.choice(VAGUE)), False
    templates = list(spec["t"])
    reserved = False
    if date >= pd.Timestamp(C.VAL_START) and rng.random() < 0.25:
        templates = [spec["rt"]]
        reserved = True
    tpl = force_template or str(rng.choice(templates))
    d = tpl.format(city=rng.choice(CITIES), client=rng.choice(CLIENTS), item=rng.choice(ITEMS), n=int(rng.integers(1, 5)))
    words = d.split(" ")
    words = [ABBR[w] if (w in ABBR and rng.random() < 0.15) else w for w in words]
    d = " ".join(words)
    if rng.random() < 0.18:
        d = _typo(d, rng)
    if rng.random() < 0.1:
        d = d.upper() if rng.random() < 0.5 else d.capitalize()
    return d, reserved


def _pick_merchant(cat: str, date: pd.Timestamp, rng) -> tuple[str, bool]:
    spec = CAT[cat]
    if date >= pd.Timestamp(C.VAL_START) and rng.random() < 0.12:
        return spec["rm"], True
    return str(rng.choice(spec["m"])), False


def _amount(cat: str, mult: float, rng) -> float:
    spec = CAT[cat]
    return float(round(spec["med"] * mult * np.exp(rng.normal(0, spec["sg"])), 2))


def _row(emp: dict, cat: str, date: pd.Timestamp, rng, amount: float | None = None) -> dict:
    merchant, mres = _pick_merchant(cat, date, rng)
    desc, pres = _describe(cat, date, rng)
    pay = rng.choice(C.PAYMENT_METHODS, p=PAY_W.get(cat, DEFAULT_PAY))
    return dict(employee_id=emp["id"], department=emp["dept"], expense_date=date, merchant=merchant,
                description=desc, category=cat, amount=amount if amount is not None else _amount(cat, emp["mult"], rng),
                payment_method=str(pay), is_anomaly=0, anomaly_type="none", duplicate_group_id="", split_group_id="",
                merchant_reserved=int(mres), phrasing_reserved=int(pres))


def generate(n_target: int = C.N_TARGET_ROWS, seed: int = C.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range(C.START_DATE, C.END_DATE, freq="D")
    wk_days = days[days.dayofweek >= 5]
    wd_days = days[days.dayofweek < 5]
    n_emp = 250
    emps = []
    for i in range(n_emp):
        join = days[0] if rng.random() < 0.8 else days[int(rng.integers(0, len(days) - 60))]
        emps.append(dict(id=f"E{i + 1:04d}", dept=C.DEPARTMENTS[int(rng.integers(0, len(C.DEPARTMENTS)))],
                         mult=float(np.exp(rng.normal(0, 0.25))), rate=float(np.exp(rng.normal(0, 0.6))), join=join))
    joins = np.array([e["join"] for e in emps], dtype="datetime64[ns]")
    rates = np.array([e["rate"] for e in emps])

    def active_emp(date: pd.Timestamp, dept_w=None):
        mask = joins <= np.datetime64(date)
        p = rates * mask
        p = p / p.sum()
        return emps[int(rng.choice(n_emp, p=p))]

    n_base = int(n_target * 0.945)
    rows: list[dict] = []
    # --- normal rows -------------------------------------------------------------------------------
    for _ in range(int(n_base * 0.96)):
        date_pool_idx = int(rng.integers(0, len(days)))
        base_date = days[date_pool_idx]
        emp = active_emp(base_date)
        w = np.array(DEPT_CAT_W[emp["dept"]], dtype=float)
        cat = C.CATEGORIES[int(rng.choice(len(C.CATEGORIES), p=w / w.sum()))]
        weekend = rng.random() < CAT[cat]["wk"]
        pool = wk_days if weekend else wd_days
        date = pool[int(rng.integers(0, len(pool)))]
        if date < emp["join"]:
            date = pool[pool >= emp["join"]][0] if (pool >= emp["join"]).any() else date
        rows.append(_row(emp, cat, date, rng))
    # legitimate recurring monthly subscriptions / bills (same amount ~30 days apart)
    n_rec = int(n_base * 0.04) // 24 + 1
    for _ in range(n_rec):
        emp = emps[int(rng.integers(0, n_emp))]
        cat = str(rng.choice(["Software & Subscriptions", "Telecom & Utilities"]))
        amt = _amount(cat, emp["mult"], rng)
        d = max(pd.Timestamp(emp["join"]), days[0]) + pd.Timedelta(days=int(rng.integers(0, 25)))
        merchant = str(rng.choice(CAT[cat]["m"]))
        desc = "monthly subscription" if cat.startswith("Software") else "monthly bill"
        while d <= days[-1]:
            if d.dayofweek < 5:
                r = _row(emp, cat, d, rng, amount=amt)
                r["merchant"], r["description"] = merchant, desc
                r["merchant_reserved"] = r["phrasing_reserved"] = 0
                rows.append(r)
            d += pd.Timedelta(days=int(rng.integers(29, 32)))
    # hard negatives: legitimate second claim, same employee/day/merchant, ordinary amount
    n_norm = len(rows)
    for i in rng.choice(n_norm, size=int(n_norm * 0.012), replace=False):
        r = dict(rows[int(i)])
        if r["category"] in ("Meals", "Local Transport", "Office Supplies", "Client Entertainment"):
            r["amount"] = _amount(r["category"], 1.0, rng)
            r["description"], pres = _describe(r["category"], r["expense_date"], rng)
            r["phrasing_reserved"] = int(pres)
            rows.append(r)
    normal_df = pd.DataFrame(rows).reset_index(drop=True)
    normal_df["_ord"] = np.arange(len(normal_df))
    n_norm = len(normal_df)
    boundaries = [pd.Timestamp(C.VAL_START), pd.Timestamp(C.TEST_START)]
    anomalies: list[dict] = []
    gid = 0

    # --- 1. duplicate claims: later copy of an earlier claim ----------------------------------------
    for i in rng.choice(n_norm, size=int(n_target * 0.012), replace=False):
        o = normal_df.iloc[int(i)].to_dict()
        d0 = pd.Timestamp(o["expense_date"])
        d1 = d0 + pd.Timedelta(days=int(rng.integers(0, 6)))
        if any(d0 < b <= d1 for b in boundaries) or d1 > days[-1]:
            d1 = d0
        gid += 1
        g = f"D{gid:05d}"
        normal_df.loc[int(i), "duplicate_group_id"] = g
        o.update(expense_date=d1, is_anomaly=1, anomaly_type="duplicate_claim", duplicate_group_id=g)
        anomalies.append(o)
    # --- 2. unusually large amount for the category ---------------------------------------------------
    for _ in range(int(n_target * 0.012)):
        date = days[int(rng.integers(0, len(days)))]
        emp = active_emp(date)
        w = np.array(DEPT_CAT_W[emp["dept"]], dtype=float)
        cat = C.CATEGORIES[int(rng.choice(len(C.CATEGORIES), p=w / w.sum()))]
        p95 = CAT[cat]["med"] * emp["mult"] * np.exp(1.645 * CAT[cat]["sg"])
        r = _row(emp, cat, date, rng, amount=float(round(p95 * rng.uniform(1.8, 5.0), 2)))
        r.update(is_anomaly=1, anomaly_type="large_amount")
        anomalies.append(r)
    # --- 3. split claims just under the approval threshold ---------------------------------------------
    for _ in range(int(n_target * 0.005)):
        date = wd_days[int(rng.integers(0, len(wd_days)))]
        emp = active_emp(date)
        cat = str(rng.choice(["Office Supplies", "Marketing", "Client Entertainment", "Training & Courses",
                              "Software & Subscriptions"]))
        merchant, mres = _pick_merchant(cat, date, rng)
        k = int(rng.integers(2, 4))
        gid += 1
        g = f"S{gid:05d}"
        for j in range(k):
            r = _row(emp, cat, date, rng, amount=float(round(C.APPROVAL_THRESHOLD * rng.uniform(0.82, 0.98), 2)))
            r.update(merchant=merchant, merchant_reserved=int(mres), split_group_id=g)
            if j >= 1:  # first claim of the group looks like an ordinary claim; later ones are the anomaly
                r.update(is_anomaly=1, anomaly_type="split_claim")
            anomalies.append(r)
    # --- 4. unusual weekend claims in working-day categories with elevated amount ---------------------------
    for _ in range(int(n_target * 0.008)):
        date = wk_days[int(rng.integers(0, len(wk_days)))]
        emp = active_emp(date)
        cat = str(rng.choice(C.BUSINESS_CATEGORIES))
        r = _row(emp, cat, date, rng)
        r["amount"] = float(round(CAT[cat]["med"] * emp["mult"] * rng.uniform(1.3, 2.6), 2))
        r.update(is_anomaly=1, anomaly_type="weekend_unusual")
        anomalies.append(r)
    # --- 5. description/category mismatch: text+merchant from category X, claimed category Y ------------------
    for _ in range(int(n_target * 0.010)):
        date = days[int(rng.integers(0, len(days)))]
        emp = active_emp(date)
        x = C.CATEGORIES[int(rng.integers(0, len(C.CATEGORIES)))]
        y = str(rng.choice([c for c in C.CATEGORIES if c != x]))
        r = _row(emp, x, date, rng)
        r.update(category=y, is_anomaly=1, anomaly_type="category_mismatch")
        anomalies.append(r)

    an = pd.DataFrame(anomalies)
    an["_ord"] = np.arange(n_norm, n_norm + len(an))
    df = pd.concat([normal_df, an], ignore_index=True)
    df["expense_date"] = pd.to_datetime(df["expense_date"])
    df = df.sort_values(["expense_date", "_ord"], kind="mergesort").reset_index(drop=True)
    df.insert(0, "expense_id", [f"X{i + 1:06d}" for i in range(len(df))])
    df = df.drop(columns="_ord")
    df["amount"] = df["amount"].round(2)
    return df


def df_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False, date_format="%Y-%m-%d").encode()).hexdigest()


def write(out_dir=C.GEN_DIR, seed: int = C.SEED, n: int = C.N_TARGET_ROWS) -> tuple[pd.DataFrame, str]:
    df = generate(n, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "expenses.csv"
    df.to_csv(path, index=False, date_format="%Y-%m-%d")
    C.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    df.sample(200, random_state=seed).sort_values("expense_id").to_csv(C.SAMPLE_DIR / "sample_200.csv", index=False,
                                                                       date_format="%Y-%m-%d")
    return df, hashlib.sha256(path.read_bytes()).hexdigest()


def load(path=None) -> pd.DataFrame:
    path = path or (C.GEN_DIR / "expenses.csv")
    if not path.exists():
        write()
    df = pd.read_csv(path, keep_default_na=False, dtype={"description": str, "merchant": str})
    df["expense_date"] = pd.to_datetime(df["expense_date"])
    return df


if __name__ == "__main__":
    out = write()
    print("rows:", len(out[0]), "sha256:", out[1])
    sys.exit(0)
