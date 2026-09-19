"""Train, compare and save the attrition model.

Run from the project root:
    python src/train.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORICAL, NUMERIC, load_raw, prepare

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Table_1.csv"
MODEL_PATH = ROOT / "models" / "attrition_pipeline.joblib"
REPORT_PATH = ROOT / "reports" / "metrics.json"
SEED = 42


def build_preprocessor():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore": a new city/group at prediction time won't crash the app
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])


def candidate_models():
    return {
        "logistic_regression": LogisticRegression(
            C=0.5, class_weight="balanced", max_iter=1000, random_state=SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced",
            random_state=SEED, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
    }


def main():
    df = load_raw(DATA_PATH)
    X, y = prepare(df)
    print(f"Rows: {len(X)} | Left rate: {y.mean():.1%}")

    # Hold out a test set that is never used for model selection
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # 1) Compare models with stratified 5-fold CV on the training data only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = ["roc_auc", "f1", "precision", "recall", "accuracy"]
    results = {}
    print("\n5-fold cross-validation (mean +/- std):")
    for name, clf in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", clf)])
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        results[name] = {m: (scores[f"test_{m}"].mean(), scores[f"test_{m}"].std()) for m in scoring}
        r = results[name]
        print(f"  {name:20s} AUC {r['roc_auc'][0]:.3f}+/-{r['roc_auc'][1]:.3f} | "
              f"F1 {r['f1'][0]:.3f} | precision {r['precision'][0]:.3f} | "
              f"recall {r['recall'][0]:.3f} | acc {r['accuracy'][0]:.3f}")

    # 2) Pick best by ROC-AUC (threshold-independent, robust to class imbalance)
    best_name = max(results, key=lambda n: results[n]["roc_auc"][0])
    print(f"\nBest model by CV ROC-AUC: {best_name}")

    # 3) Refit on the full training set, evaluate ONCE on the held-out test set
    best = Pipeline([("prep", build_preprocessor()), ("model", candidate_models()[best_name])])
    best.fit(X_train, y_train)
    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    test_auc = roc_auc_score(y_test, proba)
    print(f"\nHeld-out test ROC-AUC: {test_auc:.3f}")
    print(classification_report(y_test, pred, target_names=["Stay", "Left"]))
    print("Confusion matrix [rows=true, cols=pred] (Stay, Left):")
    print(confusion_matrix(y_test, pred))

    # 4) Save the WHOLE pipeline (encoding + model) as one artifact
    MODEL_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps({
        "best_model": best_name,
        "cv": {n: {m: round(v[0], 4) for m, v in r.items()} for n, r in results.items()},
        "test_roc_auc": round(test_auc, 4),
        "baseline_left_rate": round(float(y.mean()), 4),
    }, indent=2))
    print(f"\nSaved model -> {MODEL_PATH}\nSaved metrics -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
