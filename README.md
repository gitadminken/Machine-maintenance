# Machine Failure Prediction

FastAPI web app that predicts machine failures from sensor data. Uses an XGBoost model trained on the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (10,000 records).

## What it does

You enter sensor readings (temperature, RPM, torque, tool wear) and the model tells you if the machine is likely to fail. The dashboard also shows the failure probability and suggests an action level (normal / monitor / warning / critical).

There's a JSON API at `POST /api/predict` too.

## Model details

The dataset is heavily imbalanced (~3.4% failure rate). I compared 7 classifiers — Logistic Regression, KNN, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, and XGBoost. Picked XGBoost because it had the best balance between recall and precision.

Why that balance matters: in an industrial setting, a model with high recall but terrible precision generates too many false alarms. Technicians start ignoring the alerts because they've been burned too many times — and then a real failure gets missed. The model needs to catch failures (recall) but also be credible when it does flag something (precision), so the maintenance team actually trusts it and responds.

Key decisions:
- Used `scale_pos_weight` to handle class imbalance instead of oversampling
- Tuned with `RandomizedSearchCV` (30 iterations, 5-fold stratified CV, scored on F1)
- Scored on F1 specifically because it penalizes models that sacrifice precision for recall or vice versa
- Only scaled numerical features (temperatures, rpm, torque, tool wear), left one-hot encoded `Type` columns as-is

Test set results (20% holdout, stratified):
- ROC-AUC: 0.96
- Recall: ~76%
- 52 true positives, 22 false positives out of 2,000 samples

Training notebook is in `notebook/training.ipynb`.

## Project structure

```
main.py                  # FastAPI app (UI + JSON API)
models/
  model.pkl              # trained XGBoost classifier
  preprocessor.pkl       # StandardScaler for numerical features
notebook/
  training.ipynb         # model training and evaluation
  data/                  # dataset and train/test splits
templates/index.html     # dashboard UI
static/style.css
```

## Run locally

```bash
pip install -r requirements.txt
python main.py
```

Open http://localhost:8000

## Run with Docker

```bash
docker-compose up --build
```

Open http://localhost:8001
