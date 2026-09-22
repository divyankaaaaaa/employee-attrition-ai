"""Ablation study: how much does the model depend on specific features?

Trains the same models under different feature sets and compares scores, so you can
report honest numbers (e.g. without a feature that may leak the outcome).

Run from the project root:
    python src/ablation.py
"""
import json
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORICAL, NUMERIC, load_raw, prepare
from train import DATA_PATH, SEED, candidate_models

REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "ablation.json"

# Each scenario = a list of features to REMOVE from the full set
SCENARIOS = {
    "A. All features": [],
    "B. Without Job Role Match": ["Job Role Match"],
    "C. Without Job Role Match + Gender + Marital Status": [
        "Job Role Match", "Gender", "Marital Status",
    ],
}


def make_pipeline(model, drop):
    num = [c for c in NUMERIC if c not in drop]
    cat = [c for c in CATEGORICAL if c not in drop]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    return Pipeline([("prep", prep), ("model", model)]), num + cat


def main():
    X, y = prepare(load_raw(DATA_PATH))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results = {}
    print(f"{'Scenario':55s} {'Model':20s} {'CV AUC':>14s} {'Test AUC':>9s}")
    print("-" * 102)
    for scen, drop in SCENARIOS.items():
        results[scen] = {}
        for name in ["logistic_regression", "random_forest", "gradient_boosting"]:
            pipe, cols = make_pipeline(candidate_models()[name], drop)
            scores = cross_val_score(pipe, X_train[cols], y_train, cv=cv, scoring="roc_auc")
            pipe.fit(X_train[cols], y_train)
            test_auc = roc_auc_score(y_test, pipe.predict_proba(X_test[cols])[:, 1])
            results[scen][name] = {
                "cv_auc_mean": round(scores.mean(), 4),
                "cv_auc_std": round(scores.std(), 4),
                "test_auc": round(test_auc, 4),
            }
            print(f"{scen:55s} {name:20s} {scores.mean():.3f}+/-{scores.std():.3f}    {test_auc:.3f}")
        print()

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Saved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
