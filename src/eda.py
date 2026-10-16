"""EDA charts saved to evidence/phase1/charts/."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config as C
from src.data_gen import load
from src.split import assign_split


def main() -> None:
    df = load()
    df["split"] = assign_split(df)
    out = C.EVIDENCE_DIR / "phase1" / "charts"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    vc = df["anomaly_type"].value_counts()
    ax.barh(vc.index, vc.values, color="#3b6ea5")
    ax.set_xlabel("rows"); ax.set_title("Label distribution (synthetic data)")
    for i, v in enumerate(vc.values):
        ax.text(v, i, f" {v}", va="center")
    fig.tight_layout(); fig.savefig(out / "01_label_distribution.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    cats = C.CATEGORIES
    ax.boxplot([np.log10(df.loc[df.category == c, "amount"]) for c in cats], tick_labels=cats, showfliers=False)
    ax.set_ylabel("log10(amount, INR)"); ax.set_title("Amount by claimed category")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout(); fig.savefig(out / "02_amount_by_category.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    m = df.set_index("expense_date").resample("MS")
    ax.plot(m.size().index, m.size().values, label="all claims")
    ax.plot(m["is_anomaly"].sum().index, m["is_anomaly"].sum().values, label="labelled anomalies")
    for x, name in [(C.VAL_START, "val start"), (C.TEST_START, "test start")]:
        ax.axvline(np.datetime64(x), color="grey", ls="--"); ax.text(np.datetime64(x), ax.get_ylim()[1] * 0.9, name, rotation=90)
    ax.set_title("Claims per month and chronological split"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "03_monthly_volume_and_splits.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    t = df.groupby(["split", "anomaly_type"]).size().unstack(fill_value=0).drop(columns="none").loc[["train", "validation", "test"]]
    t.plot(kind="bar", stacked=True, ax=ax); ax.set_title("Anomalies per split by type"); ax.set_ylabel("rows")
    plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout(); fig.savefig(out / "04_anomalies_by_split.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    wk = df.assign(weekend=df.expense_date.dt.dayofweek >= 5).groupby("category")["weekend"].mean().sort_values()
    ax.barh(wk.index, wk.values, color="#a56b3b"); ax.set_title("Share of claims on weekends by category")
    fig.tight_layout(); fig.savefig(out / "05_weekend_share_by_category.png", dpi=130); plt.close(fig)
    print("charts:", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
