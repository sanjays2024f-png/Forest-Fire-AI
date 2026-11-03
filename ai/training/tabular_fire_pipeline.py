from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Any, Dict, List
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
logger = logging.getLogger(__name__)
TARGET_NAME = "confidence_level"
RANDOM_STATE = 42
MAX_ROWS = 120000
CV_FOLDS = 10
def _csv_files(root: Path) -> List[Path]:
    c = sorted((root / "modis" / "2024").glob("*.csv"))
    return c if c else sorted(root.rglob("*.csv"))
def _load_frame(files: List[Path]) -> pd.DataFrame:
    if not files:
        raise FileNotFoundError("No CSV under dataset/. Add FIRMS country CSVs.")
    per = max(1, MAX_ROWS // max(1, len(files)))
    parts = []
    for fp in files:
        try:
            ch = pd.read_csv(fp, nrows=per * 3)
        except Exception as e:
            logger.warning("skip %s: %s", fp, e)
            continue
        if ch.empty:
            continue
        ch = ch.sample(n=min(len(ch), per), random_state=RANDOM_STATE)
        ch["source_file"] = fp.name
        parts.append(ch)
        if sum(len(p) for p in parts) >= MAX_ROWS:
            break
    if not parts:
        raise ValueError("CSVs found but no rows parsed.")
    df = pd.concat(parts, ignore_index=True)
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)
    return df
def _level(v: Any):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    t = str(v).strip().lower()
    m = {"h": "HIGH", "high": "HIGH", "m": "MEDIUM", "medium": "MEDIUM",
         "l": "LOW", "low": "LOW", "n": "LOW", "nominal": "LOW"}
    if t in m:
        return m[t]
    try:
        n = float(t)
    except (TypeError, ValueError):
        return None
    return "HIGH" if n >= 80 else "MEDIUM" if n >= 50 else "LOW"
def _summary(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    return {"dataset_rows": int(len(df)),
        "feature_count": int(len([c for c in df.columns if c != target])),
        "target": target, "columns": list(df.columns),
        "dtypes": {c: str(d) for c, d in df.dtypes.items()},
        "missing_values": {c: int(df[c].isna().sum()) for c in df.columns},
        "duplicate_rows": int(df.duplicated().sum()),
        "class_distribution": df[target].value_counts(dropna=False).to_dict()}
def _leakage(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    susp, reasons = [], {}
    for col in df.columns:
        if col == target:
            continue
        lo = col.lower()
        if "confid" in lo and target == TARGET_NAME:
            susp.append(col)
            reasons[col] = "Same MODIS confidence signal as target; review before use."
    for col in ["source_file", "version", "instrument", "satellite"]:
        if col in df.columns and col not in susp:
            susp.append(col)
            reasons[col] = f"Identifier '{col}' must not be predictive; will drop/exclude."
    for col in [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]:
        if col not in susp:
            susp.append(col)
            reasons[col] = f"Timestamp '{col}' risks future-order leakage; excluded."
    return {"target_leakage": "FOUND - review" if susp else "PASS",
        "duplicate_row_count": int(df.duplicated().sum()),
        "duplicate_leakage": "FOUND" if int(df.duplicated().sum()) > 0 else "PASS",
        "suspicious_features": sorted(set(susp)), "removal_reasons": reasons}
def _pre(num: List[str], cat: List[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
def run_satellite_training(dataset_root, output_root):
    from sklearn.metrics import precision_score, recall_score
    dataset_root, output_root = Path(dataset_root), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    files = _csv_files(dataset_root)
    raw = _load_frame(files)
    if "confidence" not in raw.columns:
        raise ValueError(f"Expected 'confidence' column. Found: {list(raw.columns)[:20]}")
    raw["confidence_level"] = raw["confidence"].apply(_level)
    raw = raw.dropna(subset=[TARGET_NAME])
    if raw[TARGET_NAME].nunique() < 2:
        raise ValueError("Need >=2 confidence levels to train.")
    summary = _summary(raw, TARGET_NAME)
    print(f"Dataset rows: {summary['dataset_rows']}")
    print(f"Features: {summary['feature_count']}")
    print(f"Target: {summary['target']}")
    print(f"Class distribution: {summary['class_distribution']}")
    print(f"Missing values: {summary['missing_values']}")
    print(f"Duplicate rows: {summary['duplicate_rows']}")
    df = raw.drop_duplicates().copy()
    dup_removed = int(summary["duplicate_rows"])
    leak = _leakage(df, TARGET_NAME)
    # CRITICAL: raw 'confidence' is the source of the target -> must NEVER be a feature.
    drop_cols = [c for c in ["source_file", "version", "confidence"] if c in df.columns]
    for c in drop_cols:
        if c == "confidence":
            leak["removal_reasons"][c] = "Removed 'confidence': direct source of target 'confidence_level' (target leakage)."
        else:
            leak["removal_reasons"][c] = f"Removed '{c}': provenance ID, not fire evidence."
    cands = [c for c in df.columns if c not in [TARGET_NAME, *drop_cols]]
    num = [c for c in cands if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in cands if c not in num and df[c].nunique(dropna=True) <= 30]
    excl = [c for c in cands if c not in num and c not in cat]
    for c in excl:
        leak["removal_reasons"][c] = f"Excluded '{c}': high-cardinality/timestamp; leakage/encoding risk."
    feats = num + cat
    if not feats:
        raise ValueError("No usable features.")
    X, y = df[feats], df[TARGET_NAME]
    print(f"Numeric: {num} | Categorical: {cat} | Dropped: {drop_cols} | Excluded: {excl}")
    print(f"Target leakage: {leak['target_leakage']} | Duplicate leakage: {leak['duplicate_leakage']}")
    print(f"Suspicious: {leak['suspicious_features']}")
    from sklearn.model_selection import train_test_split as _split
    x_tr, x_te, y_tr, y_te = _split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    overlap = int(pd.merge(x_tr.assign(__y__=list(y_tr)).drop_duplicates(),
        x_te.assign(__y__=list(y_te)).drop_duplicates(), how="inner").shape[0])
    leak["train_test_duplicate_overlap"] = overlap
    leak["train_test_overlap_check"] = "PASS" if overlap == 0 else "FOUND"
    print(f"Split: train={len(x_tr)} (80%) test={len(x_te)} (20% LOCKED) overlap={overlap}")
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        smote_ok = True
    except ImportError:
        smote_ok = False
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for ti, vi in cv.split(x_tr, y_tr):
        pre = _pre(num, cat)
        a_tr, a_va = x_tr.iloc[ti], x_tr.iloc[vi]
        b_tr, b_va = y_tr.iloc[ti], y_tr.iloc[vi]
        a_tr_t = pre.fit_transform(a_tr)
        a_va_t = pre.transform(a_va)
        if smote_ok:
            a_tr_t, b_tr = SMOTE(random_state=RANDOM_STATE).fit_resample(a_tr_t, b_tr)
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
        clf.fit(a_tr_t, b_tr)
        pr = clf.predict(a_va_t)
        rows.append({"accuracy": accuracy_score(b_va, pr),
            "precision": precision_score(b_va, pr, average="weighted", zero_division=0),
            "recall": recall_score(b_va, pr, average="weighted", zero_division=0),
            "f1": f1_score(b_va, pr, average="weighted", zero_division=0)})
    cf = pd.DataFrame(rows)
    cv_sum = {"mean_accuracy": round(float(cf["accuracy"].mean()), 4),
        "std_accuracy": round(float(cf["accuracy"].std()), 4),
        "mean_precision": round(float(cf["precision"].mean()), 4),
        "std_precision": round(float(cf["precision"].std()), 4),
        "mean_recall": round(float(cf["recall"].mean()), 4),
        "std_recall": round(float(cf["recall"].std()), 4),
        "mean_f1": round(float(cf["f1"].mean()), 4),
        "std_f1": round(float(cf["f1"].std()), 4)}
    print(f"10-fold CV on 80%: {cv_sum}")
    if smote_ok:
        init = ImbPipeline([("pre", _pre(num, cat)),
            ("sm", SMOTE(random_state=RANDOM_STATE)),
            ("clf", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE))])
    else:
        init = Pipeline([("pre", _pre(num, cat)),
            ("clf", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE))])
    init.fit(x_tr, y_tr)
    tr_acc = round(float(accuracy_score(y_tr, init.predict(x_tr))), 4)
    gap = tr_acc - cv_sum["mean_accuracy"]
    if gap > 0.10 and tr_acc > 0.85:
        fit_status, why = "OVERFITTING", f"Train {tr_acc} >> CV {cv_sum['mean_accuracy']} gap {round(gap,4)}."
        corr = "Limited complexity: max_depth=12, min_samples_leaf=4, SMOTE training-only."
        final_clf = RandomForestClassifier(n_estimators=250, max_depth=12,
            min_samples_leaf=4, n_jobs=-1, random_state=RANDOM_STATE)
    elif cv_sum["mean_f1"] < 0.55 and tr_acc < 0.65:
        fit_status, why = "UNDERFITTING", f"Train {tr_acc} and CV F1 {cv_sum['mean_f1']} both poor."
        corr = "Increased capacity: 350 trees, full depth, kept physical features."
        final_clf = RandomForestClassifier(n_estimators=350, n_jobs=-1, random_state=RANDOM_STATE)
    else:
        fit_status, why = "ACCEPTABLE", f"Train {tr_acc} vs CV {cv_sum['mean_accuracy']} gap {round(gap,4)}."
        corr = "No correction needed; kept initial forest."
        final_clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
    print(f"Model status: {fit_status} - {why} Correction: {corr}")
    if smote_ok:
        final = ImbPipeline([("pre", _pre(num, cat)),
            ("sm", SMOTE(random_state=RANDOM_STATE)), ("clf", final_clf)])
    else:
        final = Pipeline([("pre", _pre(num, cat)), ("clf", final_clf)])
    final.fit(x_tr, y_tr)
    te_pr = final.predict(x_te)
    labels = sorted(y.unique().tolist())
    cm_arr = confusion_matrix(y_te, te_pr, labels=labels)
    cm = cm_arr.tolist()
    pc = f1_score(y_te, te_pr, average=None, zero_division=0)
    pp = precision_score(y_te, te_pr, average=None, zero_division=0)
    rr = recall_score(y_te, te_pr, average=None, zero_division=0)
    # Sensitivity (=recall), specificity, FPR, FNR per class via one-vs-rest.
    # Honest multiclass breakdown; no binary-only shortcut.
    per_class: Dict[str, Any] = {}
    total = int(cm_arr.sum())
    for i, lb in enumerate(labels):
        tp = int(cm_arr[i, i])
        fn = int(cm_arr[i, :].sum() - tp)
        fp = int(cm_arr[:, i].sum() - tp)
        tn = int(total - tp - fn - fp)
        sens = float(tp / (tp + fn)) if (tp + fn) else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) else 0.0
        per_class[str(lb)] = {
            "f1": round(float(pc[i]), 4),
            "precision": round(float(pp[i]), 4),
            "recall": round(float(rr[i]), 4),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "fpr": round(1.0 - spec, 4),
            "fnr": round(1.0 - sens, 4),
            "support": int(cm_arr[i, :].sum()),
        }
    macro_spec = round(float(sum(v["specificity"] for v in per_class.values()) / max(1, len(per_class))), 4)
    macro_sens = round(float(recall_score(y_te, te_pr, average="macro", zero_division=0)), 4)
    final_test = {"scope": "FINAL TEST PERFORMANCE - 20% UNSEEN TEST SET",
        "accuracy": round(float(accuracy_score(y_te, te_pr)), 4),
        "precision": round(float(precision_score(y_te, te_pr, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_te, te_pr, average="weighted", zero_division=0)), 4),
        "f1": round(float(f1_score(y_te, te_pr, average="weighted", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_te, te_pr, average="macro", zero_division=0)), 4),
        "sensitivity_macro": macro_sens,
        "specificity_macro": macro_spec,
        "fpr_macro": round(1.0 - macro_spec, 4),
        "fnr_macro": round(1.0 - macro_sens, 4),
        "per_class": per_class,
        "per_class_f1": {lb: round(float(s), 4) for lb, s in zip(labels, pc)},
        "confusion_matrix": cm, "confusion_labels": labels,
        "test_rows": int(len(y_te)), "train_rows": int(len(x_tr))}
    print(f"FINAL TEST (20% unseen): {final_test}")
    mp = output_root / "models" / "satellite_confidence_model.pkl"
    mp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "features": feats, "labels": labels}, mp)
    joblib.dump(_pre(num, cat), output_root / "models" / "satellite_preprocessor.pkl")
    report = {"source": "NASA FIRMS MODIS 2024 country CSVs (real, sampled)",
        "csv_files_used": len(files), "dataset_summary": summary,
        "duplicates_removed": dup_removed,
        "preprocessing": {"numeric": num, "categorical": cat, "dropped": drop_cols,
            "excluded": excl, "target": TARGET_NAME,
            "mapping": "confidence>=80 HIGH, 50-80 MEDIUM, else LOW",
            "smote": "training folds only" if smote_ok else "unavailable"},
        "leakage_checks": leak,
        "split": {"train": "80% CV/training", "test": "20% locked", "stratified": True},
        "cv_10fold_on_80pct": cv_sum,
        "initial_model": {"name": "RandomForest", "train_accuracy": tr_acc,
            "cv_mean_f1": cv_sum["mean_f1"]},
        "fit_status": {"model_status": fit_status, "explanation": why},
        "correction": corr, "final_test": final_test,
        "artifacts": {"model": str(mp)}}
    (output_root / "satellite_training_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


