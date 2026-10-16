"""End-to-end training: `python -m src.train`.

Selection and thresholds use VALIDATION only. The TEST split is evaluated exactly once at the very end.
"""
from __future__ import annotations

import json
import platform

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_recall_fscore_support, classification_report)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from src import config as C
from src.data_gen import load
from src.features import FEATURES, assemble_features, fit_stats, history_features, make_text, normalise_text
from src.scoring import AnomalyScorer
from src.split import assign_split, check_split

EV = C.EVIDENCE_DIR / "phase2"
SEED = C.SEED


def jdump(obj, name):
    (EV / name).write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


# ----------------------------------------------------------------------------------------------- Model A
def text_pipeline(clf):
    return Pipeline([("tfidf", FeatureUnion([
        ("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2))])),
        ("clf", clf)])


def model_a(df, text, tr_idx, va_idx, te_idx):
    print("\n=== MODEL A: categorization ===")
    ok = df["anomaly_type"] != "category_mismatch"
    fit_idx = tr_idx[ok.loc[tr_idx].to_numpy()]
    va_ok = va_idx[ok.loc[va_idx].to_numpy()]
    rows, fitted = [], {}
    cands = [("LogReg C=1", LogisticRegression(C=1.0, max_iter=2000)), ("LogReg C=10", LogisticRegression(C=10.0, max_iter=2000)),
             ("LogReg C=50", LogisticRegression(C=50.0, max_iter=2000)),
             ("LinearSVM C=0.1", LinearSVC(C=0.1)), ("LinearSVM C=0.5", LinearSVC(C=0.5)), ("LinearSVM C=1", LinearSVC(C=1.0))]
    for name, clf in cands:
        p = text_pipeline(clf).fit(text.loc[fit_idx], df.loc[fit_idx, "category"])
        pred = p.predict(text.loc[va_ok])
        rows.append(dict(model=name, val_accuracy=accuracy_score(df.loc[va_ok, "category"], pred),
                         val_macro_f1=f1_score(df.loc[va_ok, "category"], pred, average="macro")))
        fitted[name] = p
        print(rows[-1])
    comp = pd.DataFrame(rows).round(4)
    comp.to_csv(EV / "model_a_comparison.csv", index=False)
    lr_rows = comp[comp.model.str.startswith("LogReg")]
    best_name = lr_rows.sort_values("val_macro_f1", ascending=False).iloc[0]["model"]
    best_c = float(best_name.split("=")[1])
    best_overall = comp.sort_values("val_macro_f1", ascending=False).iloc[0]["model"]
    print("deployed Model A:", best_name, "| best overall on validation:", best_overall)
    model = fitted[best_name]
    classes = list(model.classes_)

    # detailed validation metrics
    pred = model.predict(text.loc[va_ok])
    y = df.loc[va_ok, "category"]
    rep = classification_report(y, pred, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.round(4).to_csv(EV / "model_a_per_class_val.csv")
    cm = confusion_matrix(y, pred, labels=classes)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(EV / "model_a_confusion_val.csv")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right"); ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    ax.set_title("Model A confusion matrix (validation)"); fig.tight_layout(); fig.savefig(EV / "model_a_confusion_val.png", dpi=130); plt.close(fig)

    # error analysis
    proba = model.predict_proba(text.loc[va_ok])
    conf = proba.max(1)
    err = df.loc[va_ok].assign(pred=pred, conf=conf)
    err = err[err.pred != err.category]
    conf_pairs = err.groupby(["category", "pred"]).size().sort_values(ascending=False).head(10)
    desc = normalise_text(err["description"])
    from src.features import VAGUE_TEXTS
    lines = [f"validation errors: {len(err)} of {len(va_ok)} ({len(err) / len(va_ok):.2%})",
             f"errors with empty description: {(desc == '').sum()}", f"errors with vague description: {desc.isin(VAGUE_TEXTS).sum()}",
             "", "top confusions (true -> predicted : count):"] + [f"  {a} -> {b} : {n}" for (a, b), n in conf_pairs.items()]
    lines += ["", "sample misclassified rows (description | merchant | true | predicted | confidence):"]
    for _, r in err.sample(min(15, len(err)), random_state=SEED).iterrows():
        lines.append(f"  {r.description!r} | {r.merchant} | {r.category} | {r.pred} | {r.conf:.2f}")
    (EV / "model_a_error_analysis.txt").write_text("\n".join(lines))

    # unseen merchants / phrasings (relative to TRAIN)
    tr_m = set(df.loc[fit_idx, "merchant"]); tr_d = set(normalise_text(df.loc[fit_idx, "description"]))
    def unseen_table(idx):
        sub = df.loc[idx]
        um = ~sub.merchant.isin(tr_m); ud = ~normalise_text(sub.description).isin(tr_d)
        out = []
        for lab, m in [("all", np.ones(len(sub), bool)), ("seen merchant & seen text", ~um.to_numpy() & ~ud.to_numpy()),
                       ("unseen merchant", um.to_numpy()), ("unseen description text", ud.to_numpy()), ("reserved phrasing template", sub.phrasing_reserved.to_numpy() == 1)]:
            s = sub[m]
            if len(s) == 0:
                continue
            p = model.predict(text.loc[s.index])
            out.append(dict(subset=lab, n=len(s), accuracy=accuracy_score(s.category, p), macro_f1=f1_score(s.category, p, average="macro")))
        return pd.DataFrame(out).round(4)
    unseen_val = unseen_table(va_ok)
    unseen_val.to_csv(EV / "model_a_unseen_val.csv", index=False)
    print(unseen_val.to_string(index=False))

    # confidence reliability table (NOT a calibration claim)
    bins = pd.cut(conf, [0, 0.4, 0.6, 0.8, 0.95, 1.0001])
    rel = pd.DataFrame({"bin": bins, "correct": (pred == y.to_numpy())}).groupby("bin", observed=True)["correct"].agg(["count", "mean"]).rename(columns={"mean": "accuracy"})
    rel.round(4).to_csv(EV / "model_a_confidence_vs_accuracy_val.csv")
    return dict(model=model, classes=classes, best_name=best_name, best_c=best_c, fit_idx=fit_idx, ok=ok,
                comparison=comp, best_overall=best_overall, tr_m=tr_m)


def oof_probabilities(df, text, tr_idx, fit_idx_set, best_c, classes, k=5):
    """Out-of-fold class probabilities for TRAIN rows (contiguous blocks); Model-A fitted without the block."""
    blocks = np.array_split(tr_idx, k)
    P = np.zeros((len(tr_idx), len(classes)))
    off = 0
    for b in blocks:
        fit = np.array([i for i in tr_idx if i not in set(b) and i in fit_idx_set])
        m = text_pipeline(LogisticRegression(C=best_c, max_iter=2000)).fit(text.loc[fit], df.loc[fit, "category"])
        assert list(m.classes_) == classes
        P[off:off + len(b)] = m.predict_proba(text.loc[b])
        off += len(b)
    return P


# ----------------------------------------------------------------------------------------------- Model B
def make_model(name, pos_weight):
    if name == "iforest":
        return IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1)
    if name == "logreg":
        return Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced", C=1.0, max_iter=5000))])
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    if name in ("xgboost", "xgboost_unweighted"):
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=pos_weight if name == "xgboost" else 1.0, eval_metric="aucpr", n_jobs=4,
                             random_state=SEED, tree_method="hist")
    raise ValueError(name)


def fit_scorer(name, X, y):
    pw = float((y == 0).sum() / max((y == 1).sum(), 1))
    m = make_model(name, pw)
    if name == "iforest":
        m.fit(X)
        raw = -m.score_samples(X)
        return AnomalyScorer(m, "iforest", float(raw.min()), float(raw.max()))
    m.fit(X, y)
    return AnomalyScorer(m, name)


def tune_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    p, r = p[:-1], r[:-1]
    f1 = 2 * p * r / (p + r + 1e-12)
    i = int(np.argmax(f1))
    review_t = float(t[i])
    ok = np.where((p >= 0.90) & (t >= review_t))[0]
    flag_t = float(t[ok[0]]) if len(ok) else review_t
    return review_t, flag_t


def prf(y, s, thr):
    pred = (s >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(precision=p, recall=r, f1=f, tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def per_type(df_sub, s, thr):
    out = []
    for t in C.ANOMALY_TYPES:
        m = (df_sub.anomaly_type == t).to_numpy()
        if m.sum():
            out.append(dict(type=t, n=int(m.sum()), recall=float((s[m] >= thr).mean())))
    nm = (df_sub.anomaly_type == "none").to_numpy()
    out.append(dict(type="none (false-positive rate)", n=int(nm.sum()), recall=float((s[nm] >= thr).mean())))
    return pd.DataFrame(out).round(4)


def model_b(df, X, y, tr_idx, va_idx):
    print("\n=== MODEL B: anomaly detection ===")
    Xtr, ytr, Xva, yva = X.loc[tr_idx], y.loc[tr_idx], X.loc[va_idx], y.loc[va_idx]
    names = ["iforest", "logreg", "random_forest", "xgboost", "xgboost_unweighted"]
    res, scorers, sva = [], {}, {}
    for n in names:
        sc = fit_scorer(n, Xtr, ytr)
        s = sc.score(Xva)
        rt, ft = tune_threshold(yva, s)
        m = prf(yva, s, rt)
        res.append(dict(model=n, val_pr_auc=average_precision_score(yva, s), review_threshold=rt, flag_threshold=ft, **m))
        scorers[n], sva[n] = sc, s
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[-1].items()})
    comp = pd.DataFrame(res).round(4)
    comp.to_csv(EV / "model_b_comparison_val.csv", index=False)
    base_rate = float(yva.mean())
    print("validation positive rate (PR-AUC chance level):", round(base_rate, 4))
    # PR curves
    fig, ax = plt.subplots(figsize=(7, 5))
    for n in names:
        p, r, _ = precision_recall_curve(yva, sva[n]); ax.plot(r, p, label=f"{n} (AP={average_precision_score(yva, sva[n]):.3f})")
    ax.axhline(base_rate, color="grey", ls=":"); ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title("Model B precision-recall (validation)"); ax.legend()
    fig.tight_layout(); fig.savefig(EV / "model_b_pr_curves_val.png", dpi=130); plt.close(fig)

    # time-aware CV on TRAIN (expanding window)
    cv_rows = []
    sup = ["logreg", "random_forest", "xgboost"]
    tscv = TimeSeriesSplit(n_splits=4)
    for n in sup:
        aps = []
        for a, b in tscv.split(Xtr):
            sc = fit_scorer(n, Xtr.iloc[a], ytr.iloc[a])
            aps.append(average_precision_score(ytr.iloc[b], sc.score(Xtr.iloc[b])))
        cv_rows.append(dict(model=n, folds=len(aps), pr_auc_mean=np.mean(aps), pr_auc_std=np.std(aps), per_fold=[round(a, 4) for a in aps]))
        print("time-aware CV", cv_rows[-1])
    pd.DataFrame(cv_rows).round(4).to_csv(EV / "model_b_time_cv_train.csv", index=False)

    # selection: best validation PR-AUC (supervised candidates; ties -> simpler)
    sel = comp[comp.model.isin(["logreg", "random_forest", "xgboost"])].sort_values("val_pr_auc", ascending=False).iloc[0]["model"]
    print("selected Model B (highest validation PR-AUC):", sel)
    sc = scorers[sel]
    rt, ft = tune_threshold(yva, sva[sel])
    return dict(comp=comp, selected=sel, scorer=sc, review_t=rt, flag_t=ft, s_val=sva[sel], Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, cv=cv_rows)


def tier_table(df_sub, s, rt, ft):
    tier = np.where(s >= ft, "flag", np.where(s >= rt, "review", "approve"))
    t = pd.DataFrame({"tier": tier, "y": df_sub["is_anomaly"].to_numpy()}).groupby("tier")["y"].agg(["count", "sum"])
    t = t.rename(columns={"sum": "true_anomalies"}).reindex(["approve", "review", "flag"]).fillna(0)
    t["share_of_claims"] = t["count"] / t["count"].sum()
    t["anomaly_rate_in_tier"] = t["true_anomalies"] / t["count"].replace(0, np.nan)
    return t.round(4)


def loto(df, X, y, tr_idx, va_idx, B):
    print("\n=== Leave-one-anomaly-type-out (validation) ===")
    sel = B["selected"]
    full_s = B["s_val"]
    rows = []
    for t in C.ANOMALY_TYPES:
        keep = tr_idx[~((df.loc[tr_idx, "anomaly_type"] == t).to_numpy())]
        sc = fit_scorer(sel, X.loc[keep], y.loc[keep])
        s = sc.score(X.loc[va_idx])
        vt = (df.loc[va_idx, "anomaly_type"] == t).to_numpy()
        rt, _ = tune_threshold(y.loc[va_idx][~vt], s[~vt])            # threshold tuned WITHOUT the held-out type
        rec_held = float((s[vt] >= rt).mean())
        rec_full = float((full_s[vt] >= B["review_t"]).mean())
        rows.append(dict(held_out_type=t, n_val_positives=int(vt.sum()), recall_full_model=rec_full, recall_when_type_unseen=rec_held,
                         recall_drop=rec_full - rec_held, pr_auc_full_val=average_precision_score(y.loc[va_idx], full_s),
                         pr_auc_unseen_type_val=average_precision_score(y.loc[va_idx], s)))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()})
    out = pd.DataFrame(rows).round(4)
    out.to_csv(EV / "leave_one_type_out_val.csv", index=False)
    return out


def importance(B, df, va_idx):
    sc, sel = B["scorer"], B["selected"]
    rows = []
    if sel in ("random_forest", "xgboost"):
        gi = sc.model.feature_importances_
        rows = [dict(feature=f, model_importance=float(v)) for f, v in zip(FEATURES, gi)]
    else:
        coef = sc.model.named_steps["clf"].coef_[0]
        rows = [dict(feature=f, model_importance=float(abs(v))) for f, v in zip(FEATURES, coef)]
    imp = pd.DataFrame(rows)
    sub = B["Xva"]; ys = B["yva"]
    pi = permutation_importance(sc.model, sub, ys, scoring="average_precision", n_repeats=5, random_state=SEED, n_jobs=1)
    imp["permutation_pr_auc_drop_val"] = pi.importances_mean
    imp = imp.sort_values("permutation_pr_auc_drop_val", ascending=False).round(5)
    imp.to_csv(EV / "feature_importance.csv", index=False)
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5)); ax.barh(top.feature, top.permutation_pr_auc_drop_val, color="#3b6ea5")
    ax.set_title(f"Permutation importance on validation ({sel})"); ax.set_xlabel("drop in PR-AUC")
    fig.tight_layout(); fig.savefig(EV / "feature_importance.png", dpi=130); plt.close(fig)
    return imp


def main():
    EV.mkdir(parents=True, exist_ok=True)
    df = load()
    data_sha = __import__("hashlib").sha256((C.GEN_DIR / "expenses.csv").read_bytes()).hexdigest()
    df["split"] = assign_split(df)
    assert all(check_split(df).values()), check_split(df)
    tr_idx, va_idx, te_idx = [df.index[df.split == s].to_numpy() for s in ("train", "validation", "test")]
    print("rows:", len(df), {s: int((df.split == s).sum()) for s in ("train", "validation", "test")}, "data sha256:", data_sha)
    text = make_text(df["description"], df["merchant"])

    A = model_a(df, text, tr_idx, va_idx, te_idx)
    classes = A["classes"]
    fit_set = set(A["fit_idx"].tolist())
    print("\ncomputing out-of-fold Model A probabilities for training rows ...")
    P = np.zeros((len(df), len(classes)))
    P[tr_idx] = oof_probabilities(df, text, tr_idx, fit_set, A["best_c"], classes)
    rest = np.concatenate([va_idx, te_idx])
    P[rest] = A["model"].predict_proba(text.loc[rest])

    stats = fit_stats(df.loc[tr_idx])
    hist = history_features(df)
    X = assemble_features(df, hist, stats, P, classes)
    y = df["is_anomaly"].astype(int)
    B = model_b(df, X, y, tr_idx, va_idx)
    sel, rt, ft = B["selected"], B["review_t"], B["flag_t"]
    print(f"\nselected={sel} review_threshold={rt:.4f} flag_threshold={ft:.4f}")
    pt_val = per_type(df.loc[va_idx], B["s_val"], rt); pt_val.to_csv(EV / "model_b_per_type_recall_val.csv", index=False)
    tier_val = tier_table(df.loc[va_idx], B["s_val"], rt, ft); tier_val.to_csv(EV / "risk_tiers_val.csv")
    print(pt_val.to_string(index=False)); print(tier_val.to_string())
    lo = loto(df, X, y, tr_idx, va_idx, B)
    imp = importance(B, df, va_idx)
    print("\ntop features:\n", imp.head(8)[["feature", "permutation_pr_auc_drop_val"]].to_string(index=False))

    # ---- artifacts ---------------------------------------------------------------------------------------
    vdir = C.MODELS_DIR / C.MODEL_VERSION
    vdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(A["model"], vdir / "category_model.joblib")
    joblib.dump(B["scorer"], vdir / "anomaly_model.joblib")
    last = df.expense_date.max()
    seed_hist = df[df.expense_date > last - pd.Timedelta(days=C.HISTORY_DAYS)][["employee_id", "expense_date", "merchant", "amount", "category"]]
    seed_hist.to_csv(vdir / "history_seed.csv", index=False, date_format="%Y-%m-%d")
    meta = dict(model_version=C.MODEL_VERSION, data_sha256=data_sha, seed=SEED, features=FEATURES, classes=classes,
                model_a=A["best_name"], model_b=sel, review_threshold=rt, flag_threshold=ft, stats=stats,
                merchants_seen_in_train=sorted(A["tr_m"]), approval_threshold=C.APPROVAL_THRESHOLD,
                history_days=C.HISTORY_DAYS, history_seed_rows=int(len(seed_hist)),
                libs=dict(python=platform.python_version(), sklearn=sklearn.__version__, xgboost=xgboost.__version__,
                          numpy=np.__version__, pandas=pd.__version__),
                note="Scores are model scores, NOT calibrated probabilities. Trained on synthetic data only.")
    (vdir / "metadata.json").write_text(json.dumps(meta, indent=2))
    (C.MODELS_DIR / "LATEST").write_text(C.MODEL_VERSION)
    print("artifacts saved to", vdir)

    # ---- FINAL, ONE-TIME TEST EVALUATION (no selection beyond this point) ---------------------------------------
    print("\n=== FINAL TEST EVALUATION (used once) ===")
    ok_te = te_idx[A["ok"].loc[te_idx].to_numpy()]
    pa = A["model"].predict(text.loc[ok_te])
    test = dict(model_a=dict(model=A["best_name"], n=int(len(ok_te)), accuracy=accuracy_score(df.loc[ok_te, "category"], pa),
                             macro_f1=f1_score(df.loc[ok_te, "category"], pa, average="macro")))
    tr_d = set(normalise_text(df.loc[A["fit_idx"], "description"]))
    sub = df.loc[ok_te]
    for lab, m in [("unseen_merchant", ~sub.merchant.isin(A["tr_m"])), ("unseen_description_text", ~normalise_text(sub.description).isin(tr_d))]:
        s = sub[m.to_numpy()]
        p = A["model"].predict(text.loc[s.index])
        test["model_a"][lab] = dict(n=int(len(s)), accuracy=accuracy_score(s.category, p))
    s_te = B["scorer"].score(X.loc[te_idx])
    yte = y.loc[te_idx]
    test["model_b"] = dict(model=sel, n=int(len(te_idx)), positive_rate=float(yte.mean()), pr_auc=average_precision_score(yte, s_te),
                           review_threshold=rt, flag_threshold=ft, at_review_threshold=prf(yte, s_te, rt), at_flag_threshold=prf(yte, s_te, ft),
                           per_type_recall=per_type(df.loc[te_idx], s_te, rt).to_dict("records"),
                           risk_tiers=tier_table(df.loc[te_idx], s_te, rt, ft).reset_index().to_dict("records"))
    jdump(test, "test_results.json")
    print(json.dumps(test, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))[:3500])
    jdump(dict(model_a_deployed=A["best_name"], model_a_best_on_validation=A["best_overall"], model_b_selected=sel,
               model_b_selection_rule="highest validation PR-AUC among supervised candidates",
               val_pr_auc=B["comp"].set_index("model")["val_pr_auc"].to_dict(), review_threshold=rt, flag_threshold=ft,
               threshold_rule="review = F1-maximising threshold on validation; flag = lowest threshold >= review with validation precision >= 0.90"),
          "selection.json")


if __name__ == "__main__":
    main()
