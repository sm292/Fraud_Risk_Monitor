"""Compare supervised models and create retrospective test-period scores."""
from pathlib import Path
import json
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.data import ROOT, TARGET, download_dataset, validate_frame

ARTIFACTS, OUT, SEED = ROOT / "artifacts", ROOT / "data" / "processed", 42
N_JOBS = max(1, int(os.environ.get("FRAUD_N_JOBS", "2")))


def choose_threshold(y_true, scores, min_precision=0.10, review_rate_cap=0.02):
    """Maximize recall within validation precision and queue-rate constraints."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    options = []
    for i, threshold in enumerate(thresholds):
        rate = float((scores >= threshold).mean())
        if precision[i] >= min_precision and rate <= review_rate_cap and rate > 0:
            options.append((float(recall[i]), float(precision[i]), -rate, float(threshold)))
    # If constraints cannot both be met, choose the highest threshold that
    # still flags one transaction; test metrics make the fallback visible.
    return max(options)[3] if options else float(np.nextafter(np.max(scores), -np.inf))


def build_models():
    return {
        "logistic_regression": Pipeline([("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()), ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED))]),
        "random_forest": RandomForestClassifier(n_estimators=120, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=N_JOBS, random_state=SEED),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
            max_leaf_nodes=31, class_weight="balanced", random_state=SEED),
    }


def scores_for(model, X):
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)


def run():
    print("Preparing folders and loading dataset...", flush=True)
    ARTIFACTS.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(download_dataset())
    validate_frame(frame)
    print(f"Loaded {len(frame):,} rows. Creating chronological splits...", flush=True)
    ordered = frame.sort_values("Time", kind="stable").reset_index(drop=True)
    a, b = int(len(ordered) * .70), int(len(ordered) * .85)
    train, valid, test = ordered.iloc[:a], ordered.iloc[a:b], ordered.iloc[b:]
    for name, part in (("train", train), ("validation", valid), ("test", test)):
        if part[TARGET].nunique() != 2:
            raise ValueError(f"Chronological {name} split must contain both classes")
    features = [c for c in frame.columns if c != TARGET]
    models, comparison, valid_scores = build_models(), [], {}
    for name, model in models.items():
        print(f"Training {name.replace('_', ' ')}...", flush=True)
        model.fit(train[features], train[TARGET])
        print(f"Scoring {name.replace('_', ' ')} on validation data...", flush=True)
        valid_scores[name] = scores_for(model, valid[features])
        comparison.append({"model": name, "validation_average_precision": float(average_precision_score(valid[TARGET], valid_scores[name])),
                           "validation_roc_auc": float(roc_auc_score(valid[TARGET], valid_scores[name]))})
    selected = max(comparison, key=lambda row: row["validation_average_precision"])["model"]
    print(f"Selected {selected.replace('_', ' ')}. Evaluating held-out test data...", flush=True)
    threshold = choose_threshold(valid[TARGET].to_numpy(), valid_scores[selected])
    model = models[selected]
    test_scores = scores_for(model, test[features])
    flags, labels = test_scores >= threshold, test[TARGET].to_numpy()
    tp = int((flags & (labels == 1)).sum()); fp = int((flags & (labels == 0)).sum())
    fn = int((~flags & (labels == 1)).sum())
    metrics = {"selected_model": selected, "threshold": threshold,
        "selection_rule": "highest validation average precision; maximize validation recall at precision >= 0.10 and review rate <= 0.02 when feasible",
        "test_average_precision": float(average_precision_score(labels, test_scores)),
        "test_roc_auc": float(roc_auc_score(labels, test_scores)),
        "test_precision": tp / (tp + fp) if tp + fp else 0.0,
        "test_recall": tp / (tp + fn) if tp + fn else 0.0,
        "test_review_rate": float(flags.mean()), "test_true_positives": tp, "test_false_positives": fp,
        "test_fraud_count": int(labels.sum()), "train_rows": len(train), "validation_rows": len(valid),
        "test_rows": len(test), "note": "Retrospective chronological benchmark only; not live or production performance."}
    print("Fitting optional Isolation Forest anomaly detector...", flush=True)
    anomaly = IsolationForest(n_estimators=150, contamination="auto", n_jobs=N_JOBS, random_state=SEED)
    anomaly.fit(train.loc[train[TARGET] == 0, features])
    output = test[["Time", "Amount", TARGET]].copy()
    output.insert(0, "transaction_id", np.arange(len(output)))
    output["risk_score"] = test_scores
    output["flagged"] = flags
    output["anomaly_score"] = -anomaly.score_samples(test[features])
    output["review_status"] = np.where(flags, "Open", "Not flagged")
    output.to_csv(OUT / "scored_transactions.csv", index=False)
    output.loc[output["flagged"]].sort_values("risk_score", ascending=False).to_csv(
        OUT / "investigation_queue.csv", index=False
    )
    (ARTIFACTS / "metrics.json").write_text(json.dumps({"model_comparison": comparison, **metrics}, indent=2), encoding="utf-8")
    joblib.dump({"model": model, "features": features, "threshold": threshold,
                 "anomaly_model": anomaly, "model_name": selected}, ARTIFACTS / "fraud_model.joblib")
    print("Training complete. Metrics, model, and scored transactions have been saved.", flush=True)
    print(json.dumps(metrics, indent=2))
    print(f"Scores saved to {OUT / 'scored_transactions.csv'}")


if __name__ == "__main__":
    run()
