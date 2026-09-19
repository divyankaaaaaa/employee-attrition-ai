"""Load the saved pipeline and predict for a single employee (or a DataFrame)."""
from pathlib import Path

import joblib
import pandas as pd

from features import prepare

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "attrition_pipeline.joblib"
_model = None


def get_model():
    global _model
    if _model is None:  # load once, reuse for every request
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_one(employee: dict) -> dict:
    """employee uses the same column names as the CSV, e.g.
    {"Location": "Pune", "Emp. Group": "B2", "Function": "Operation", "Gender": "Male",
     "Tenure": 0.06, "Experience (YY.MM)": 6.08, "Marital Status": "Single",
     "Age in YY.": 27.12, "Hiring Source": "Direct",
     "Promoted/Non Promoted": "Non Promoted", "Job Role Match": "Yes"}
    """
    X, _ = prepare(pd.DataFrame([employee]))
    prob_left = float(get_model().predict_proba(X)[0, 1])
    return {
        "prediction": "Left" if prob_left >= 0.5 else "Stay",
        "probability_of_leaving": round(prob_left, 3),
        "risk_level": "High" if prob_left >= 0.6 else "Medium" if prob_left >= 0.4 else "Low",
    }


if __name__ == "__main__":
    demo = {
        "Location": "Pune", "Emp. Group": "B2", "Function": "Operation", "Gender": "Male",
        "Tenure": 0.0, "Experience (YY.MM)": 6.08, "Marital Status": "Single",
        "Age in YY.": 27.12, "Hiring Source": "Direct",
        "Promoted/Non Promoted": "Non Promoted", "Job Role Match": "Yes",
    }
    print(predict_one(demo))
