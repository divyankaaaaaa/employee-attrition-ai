"""SHAP explanations for the attrition model.

  python src/explain.py          -> saves global charts to reports/ and explains a demo employee
  from explain import explain_one -> per-employee reasons, for use in the web app / API

SHAP values here are in PROBABILITY points: +0.20 means "this feature raised the chance of
leaving by 20 percentage points compared with the average employee".
"""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save charts to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from features import CATEGORICAL, NUMERIC, load_raw, prepare
from train import DATA_PATH

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "attrition_pipeline.joblib"
REPORTS = ROOT / "reports"

_cache = {}


def _pretty(name: str) -> str:
    """'cat__Job Role Match_No' -> 'Job Role Match = No';  'num__Tenure_years' -> 'Tenure years'"""
    if name.startswith("num__"):
        return name[5:].replace("_", " ")
    name = name[5:]
    for col in CATEGORICAL:  # split "<column>_<value>" using the known column names
        if name.startswith(col + "_"):
            return f"{col} = {name[len(col) + 1:]}"
    return name


def _raw_field(name: str) -> str:
    """Map a transformed column back to the ORIGINAL input field it came from."""
    if name.startswith("num__"):
        return name[5:]
    name = name[5:]
    for col in CATEGORICAL:
        if name.startswith(col + "_"):
            return col
    return name


def _display(field: str, value) -> str:
    """Human label using the employee's ACTUAL value, e.g. 'Job Role Match = No'."""
    if field in NUMERIC:
        return f"{field.replace('_years', '')} = {float(value):.1f} yrs"
    return f"{field} = {value}"


def _load():
    if not _cache:
        pipe = joblib.load(MODEL_PATH)
        prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
        raw = list(prep.get_feature_names_out())
        names = [_pretty(n) for n in raw]
        fields = [_raw_field(n) for n in raw]
        _cache.update(pipe=pipe, prep=prep, model=model, names=names, fields=fields,
                      explainer=shap.TreeExplainer(model))
    return _cache


def _dense(a):
    return a.toarray() if hasattr(a, "toarray") else np.asarray(a)


def _shap_for_left(explainer, X_t):
    """Return SHAP values for the 'Left' class as a 2D array, across shap versions."""
    sv = explainer.shap_values(X_t)
    if isinstance(sv, list):          # older shap: [class0, class1]
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv   # newer shap: (rows, features, classes)


def _base_value(explainer):
    ev = np.atleast_1d(explainer.expected_value)
    return float(ev[1] if len(ev) > 1 else ev[0])


def explain_one(employee: dict, top_n: int = 5) -> dict:
    """Explain one employee: probability of leaving + the features pushing it up / down."""
    c = _load()
    X, _ = prepare(pd.DataFrame([employee]))
    X_t = _dense(c["prep"].transform(X))
    sv = _shap_for_left(c["explainer"], X_t)[0]
    prob = float(c["pipe"].predict_proba(X)[0, 1])

    # Add up all one-hot columns that came from the same original field, then label
    # each field with the employee's actual value.
    totals = {}
    for field, val in zip(c["fields"], sv):
        totals[field] = totals.get(field, 0.0) + float(val)
    rows = sorted(
        ([total, _display(field, X.iloc[0][field])] for field, total in totals.items()),
        key=lambda r: abs(r[0]), reverse=True,
    )

    def fmt(r):
        return {"factor": r[1], "effect": round(r[0], 3)}

    return {
        "probability_of_leaving": round(prob, 3),
        "average_employee_risk": round(_base_value(c["explainer"]), 3),
        "raises_risk": [fmt(r) for r in rows if r[0] > 0.005][:top_n],
        "lowers_risk": [fmt(r) for r in rows if r[0] < -0.005][:top_n],
    }


def global_charts(sample_size: int = 300):
    """Bar chart of overall feature importance + beeswarm plot, saved to reports/."""
    c = _load()
    X, _ = prepare(load_raw(DATA_PATH))
    X = X.sample(min(sample_size, len(X)), random_state=42)
    X_t = _dense(c["prep"].transform(X))
    sv = _shap_for_left(c["explainer"], X_t)

    # Sum one-hot columns per original field -> importance per real feature
    fields = c["fields"]
    per_field = pd.DataFrame(sv, columns=fields).T.groupby(level=0).sum().T
    importance = per_field.abs().mean().sort_values()

    REPORTS.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 5))
    importance.plot.barh(color="#3b7dd8")
    plt.xlabel("Mean |SHAP value| (average change in leaving probability)")
    plt.title("What drives attrition predictions?")
    plt.tight_layout()
    plt.savefig(REPORTS / "shap_importance.png", dpi=150)
    plt.close()

    shap.summary_plot(sv, X_t, feature_names=c["names"], show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(REPORTS / "shap_beeswarm.png", dpi=150)
    plt.close()
    return importance.sort_values(ascending=False)


if __name__ == "__main__":
    imp = global_charts()
    print("Global feature importance (mean |SHAP|, probability points):")
    print(imp.round(3).to_string())
    print(f"\nSaved charts -> {REPORTS / 'shap_importance.png'} and shap_beeswarm.png")

    demo = {
        "Location": "Pune", "Emp. Group": "B2", "Function": "Operation", "Gender": "Male",
        "Tenure": 0.0, "Experience (YY.MM)": 6.08, "Marital Status": "Single",
        "Age in YY.": 27.12, "Hiring Source": "Direct",
        "Promoted/Non Promoted": "Non Promoted", "Job Role Match": "No",
    }
    r = explain_one(demo)
    print(f"\nDemo employee: {r['probability_of_leaving']:.0%} chance of leaving "
          f"(average employee: {r['average_employee_risk']:.0%})")
    print("Raises risk:")
    for f in r["raises_risk"]:
        print(f"  {f['factor']:35s} {f['effect']:+.3f}")
    print("Lowers risk:")
    for f in r["lowers_risk"]:
        print(f"  {f['factor']:35s} {f['effect']:+.3f}")
