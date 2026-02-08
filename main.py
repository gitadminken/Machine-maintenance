import pickle
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pydantic import BaseModel


app = FastAPI(title="Machine Failure Prediction System")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

test_df = pd.read_csv("notebook/data/test.csv")


def prepare_input(type_val: str, air_temp: float, process_temp: float,
                  rpm: int, torque: float, tool_wear: int) -> pd.DataFrame:
    data = {
        'Air temperature [K]': [air_temp],
        'Process temperature [K]': [process_temp],
        'Rotational speed [rpm]': [rpm],
        'Torque [Nm]': [torque],
        'Tool wear [min]': [tool_wear],
        'Type_L': [1 if type_val == 'L' else 0],
        'Type_M': [1 if type_val == 'M' else 0]
    }
    return pd.DataFrame(data)


# Compute metrics and test data once at startup (these never change)
def _compute_model_metrics():
    X_test = test_df.drop(columns=['Machine failure'])
    y_test = test_df['Machine failure']

    X_scaled = preprocessor.transform(X_test)
    y_pred = model.predict(X_scaled)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'total_failures': int((y_test == 1).sum()),
        'model_type': 'XGBoost (Tuned)'
    }


def _compute_test_data_raw():
    results = []
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        if row.get('Type_L') == 1:
            type_val = 'L'
        elif row.get('Type_M') == 1:
            type_val = 'M'
        else:
            type_val = 'H'
        results.append({
            'Type': type_val,
            'Air temperature [K]': float(row['Air temperature [K]']),
            'Process temperature [K]': float(row['Process temperature [K]']),
            'Rotational speed [rpm]': int(row['Rotational speed [rpm]']),
            'Torque [Nm]': float(row['Torque [Nm]']),
            'Tool wear [min]': int(row['Tool wear [min]']),
            'Machine failure': int(row['Machine failure'])
        })
    return results


MODEL_METRICS = _compute_model_metrics()
TEST_DATA = _compute_test_data_raw()


@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})


@app.get("/app", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": None,
        "confidence": None,
        "metrics": MODEL_METRICS,
        "test_data": TEST_DATA,
        "selected_row_index": None
    })


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": None,
        "confidence": None,
        "metrics": MODEL_METRICS,
        "test_data": TEST_DATA,
        "selected_row_index": None
    })


class PredictionInput(BaseModel):
    type: str
    air_temp: float
    process_temp: float
    rpm: int
    torque: float
    tool_wear: int


def get_action(failure_probability: float):
    if failure_probability >= 0.7:
        return "CRITICAL", "Immediate maintenance required - High failure risk", "#dc2626"
    elif failure_probability >= 0.4:
        return "WARNING", "Schedule inspection soon - Moderate failure risk", "#f59e0b"
    elif failure_probability >= 0.2:
        return "MONITOR", "Monitor closely - Low failure risk detected", "#3b82f6"
    else:
        return "NORMAL", "No action needed - Machine operating normally", "#22c55e"


@app.post("/api/predict", response_class=JSONResponse)
async def predict_api(input: PredictionInput):
    try:
        input_df = prepare_input(
            input.type, input.air_temp, input.process_temp,
            input.rpm, input.torque, input.tool_wear
        )
        input_scaled = preprocessor.transform(input_df)

        prediction = int(model.predict(input_scaled)[0])
        proba = model.predict_proba(input_scaled)[0]

        failure_probability = float(proba[1])
        confidence = float(proba[1] if prediction == 1 else proba[0])
        action, action_text, action_color = get_action(failure_probability)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "failure_probability": failure_probability,
            "action": action,
            "action_text": action_text,
            "action_color": action_color
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
