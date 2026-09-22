"""FastAPI app for the attrition model.

Run from the project root:
    uvicorn api.main:app --reload

Then open:
    http://127.0.0.1:8000/          -> simple HTML form
    http://127.0.0.1:8000/docs      -> interactive Swagger API docs (free, auto-generated)
"""
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# so "from features import ..." and "from predict import ..." (written for src/) still work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from explain import explain_one  # noqa: E402
from predict import predict_one  # noqa: E402

app = FastAPI(
    title="Employee Attrition Prediction API",
    description="Predicts whether an employee is likely to stay or leave, with SHAP-based reasons.",
    version="1.0.0",
)


# Literal[...] means FastAPI validates the input and rejects anything not in this list,
# and /docs shows it as a dropdown automatically. Values come straight from the training data.
class Employee(BaseModel):
    location: Literal[
        "Bangalore", "Chennai", "Gurgaon", "Hyderabad", "Kolkata", "Lucknow",
        "Madurai", "Mumbai", "Nagpur", "Noida", "Pune", "Vijayawada",
    ] = Field(..., description="Employee's work location")
    emp_group: Literal["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "C3", "D2"] = Field(
        ..., description="Employee grade / band"
    )
    function: Literal["Operation", "Sales", "Support"]
    gender: Literal["Male", "Female", "other"]
    marital_status: Literal["Single", "Marr.", "Div.", "Sep.", "NTBD"]
    hiring_source: Literal["Agency", "Direct", "Employee Referral"]
    promoted: Literal["Promoted", "Non Promoted"]
    job_role_match: Literal["Yes", "No"]
    tenure_years: float = Field(..., ge=0, le=45, description="Years at the company")
    experience_years: float = Field(..., ge=0, le=45, description="Total years of experience")
    age_years: float = Field(..., ge=18, le=70, description="Employee's age")

    model_config = {
        "json_schema_extra": {
            "example": {
                "location": "Pune", "emp_group": "B2", "function": "Operation",
                "gender": "Male", "marital_status": "Single", "hiring_source": "Direct",
                "promoted": "Non Promoted", "job_role_match": "No",
                "tenure_years": 0.5, "experience_years": 6.7, "age_years": 28,
            }
        }
    }

    def to_raw_columns(self) -> dict:
        """Map the API's clean field names back to the original CSV column names/units
        that features.py and the saved pipeline expect."""
        def to_yymm(years: float) -> float:
            y = int(years)
            m = round((years - y) * 12)
            return y + m / 100

        return {
            "Location": self.location,
            "Emp. Group": self.emp_group,
            "Function": self.function,
            "Gender": self.gender,
            "Marital Status": self.marital_status,
            "Hiring Source": self.hiring_source,
            "Promoted/Non Promoted": self.promoted,
            "Job Role Match": self.job_role_match,
            "Tenure": to_yymm(self.tenure_years),
            "Experience (YY.MM)": to_yymm(self.experience_years),
            "Age in YY.": to_yymm(self.age_years),
        }


class Factor(BaseModel):
    factor: str
    effect: float


class PredictionResponse(BaseModel):
    prediction: Literal["Stay", "Left"]
    probability_of_leaving: float
    risk_level: Literal["Low", "Medium", "High"]
    raises_risk: list[Factor]
    lowers_risk: list[Factor]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(employee: Employee):
    try:
        raw = employee.to_raw_columns()
        base = predict_one(raw)
        reasons = explain_one(raw)
    except Exception as e:  # model/file issues -> a clean 500 instead of a raw traceback
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return {
        **base,
        "raises_risk": reasons["raises_risk"],
        "lowers_risk": reasons["lowers_risk"],
    }


@app.get("/", response_class=HTMLResponse)
def form():
    """A plain HTML form so this can be tried in a browser without a separate frontend."""
    return """
    <html><head><title>Attrition Predictor</title>
    <style>
      body{font-family:system-ui,sans-serif;max-width:640px;margin:40px auto;padding:0 16px}
      label{display:block;margin-top:12px;font-size:14px;color:#333}
      select,input{width:100%;padding:8px;margin-top:4px;box-sizing:border-box}
      button{margin-top:20px;padding:10px 20px;background:#2563eb;color:#fff;border:0;
             border-radius:6px;cursor:pointer;font-size:15px}
      #result{margin-top:24px;padding:16px;border-radius:8px;display:none}
      .stay{background:#dcfce7} .left{background:#fee2e2}
      ul{margin:6px 0}
    </style></head><body>
    <h2>Employee Attrition Predictor</h2>
    <p>Fill in an employee's details. Full API docs: <a href="/docs">/docs</a></p>
    <form id="f">
      <label>Location<select name="location">
        <option>Bangalore</option><option>Chennai</option><option>Gurgaon</option>
        <option>Hyderabad</option><option>Kolkata</option><option>Lucknow</option>
        <option>Madurai</option><option>Mumbai</option><option>Nagpur</option>
        <option>Noida</option><option selected>Pune</option><option>Vijayawada</option>
      </select></label>
      <label>Employee Group<select name="emp_group">
        <option>B0</option><option>B1</option><option selected>B2</option><option>B3</option>
        <option>B4</option><option>B5</option><option>B6</option><option>B7</option>
        <option>C3</option><option>D2</option>
      </select></label>
      <label>Function<select name="function">
        <option selected>Operation</option><option>Sales</option><option>Support</option>
      </select></label>
      <label>Gender<select name="gender">
        <option selected>Male</option><option>Female</option><option>other</option>
      </select></label>
      <label>Marital Status<select name="marital_status">
        <option selected>Single</option><option>Marr.</option><option>Div.</option>
        <option>Sep.</option><option>NTBD</option>
      </select></label>
      <label>Hiring Source<select name="hiring_source">
        <option selected>Direct</option><option>Agency</option><option>Employee Referral</option>
      </select></label>
      <label>Promotion<select name="promoted">
        <option selected>Non Promoted</option><option>Promoted</option>
      </select></label>
      <label>Job Role Match<select name="job_role_match">
        <option>Yes</option><option selected>No</option>
      </select></label>
      <label>Tenure (years)<input type="number" step="0.1" name="tenure_years" value="0.5"></label>
      <label>Experience (years)<input type="number" step="0.1" name="experience_years" value="6.7"></label>
      <label>Age (years)<input type="number" step="0.1" name="age_years" value="28"></label>
      <button type="submit">Predict</button>
    </form>
    <div id="result"></div>
    <script>
      document.getElementById('f').addEventListener('submit', async (e) => {
        e.preventDefault();
        const data = Object.fromEntries(new FormData(e.target));
        for (const k of ['tenure_years','experience_years','age_years']) data[k] = parseFloat(data[k]);
        const res = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'},
                                              body: JSON.stringify(data)});
        const j = await res.json();
        const div = document.getElementById('result');
        div.style.display = 'block';
        if (!res.ok) { div.className=''; div.innerHTML = '<b>Error:</b> ' + (j.detail || res.statusText); return; }
        div.className = j.prediction === 'Left' ? 'left' : 'stay';
        const li = a => a.map(f => `<li>${f.factor}: ${f.effect > 0 ? '+' : ''}${(f.effect*100).toFixed(1)} pts</li>`).join('');
        div.innerHTML = `<h3>${j.prediction} (${j.risk_level} risk) — ${(j.probability_of_leaving*100).toFixed(0)}% chance of leaving</h3>
          <b>Raises risk:</b><ul>${li(j.raises_risk) || '<li>none</li>'}</ul>
          <b>Lowers risk:</b><ul>${li(j.lowers_risk) || '<li>none</li>'}</ul>`;
      });
    </script>
    </body></html>
    """