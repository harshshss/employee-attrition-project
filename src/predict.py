"""
============================================================
 PREDICTION ENGINE -- src/predict.py
============================================================
 Loads saved model + scaler + metadata and provides inference
 API for single and batch predictions.
============================================================
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from src.utils import get_project_root


def load_model():
    """Load the trained model, scaler, and metadata from disk."""
    root = get_project_root()
    model_dir = os.path.join(root, "models")
    model = joblib.load(os.path.join(model_dir, "attrition_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    with open(os.path.join(model_dir, "model_metadata.json"), "r") as f:
        metadata = json.load(f)
    return model, scaler, metadata


def prepare_input(user_input, feature_names, scaler):
    """Convert raw user-input dict into a scaled feature vector."""
    input_df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

    for key, value in user_input.items():
        if key in feature_names:
            input_df[key] = value
        else:
            dummy_col = f"{key}_{value}"
            if dummy_col in feature_names:
                input_df[dummy_col] = 1

    return scaler.transform(input_df)


def get_risk_level(probability):
    """Map probability to risk level and color."""
    if probability < 0.30:
        return "Low Risk", "#27ae60"
    elif probability < 0.60:
        return "Medium Risk", "#f39c12"
    else:
        return "High Risk", "#e74c3c"


def generate_recommendation(user_input, probability):
    """Generate context-aware retention recommendations."""
    recommendations = []
    risk_factors = []

    if user_input.get("OverTime", "No") == "Yes":
        risk_factors.append("excessive overtime")
        recommendations.append("- **Reduce Overtime:** Redistribute workload or hire additional staff.")

    if user_input.get("JobSatisfaction", 3) <= 2:
        risk_factors.append("low job satisfaction")
        recommendations.append("- **Improve Job Satisfaction:** Conduct 1-on-1 meetings, consider role enrichment.")

    if user_input.get("EnvironmentSatisfaction", 3) <= 2:
        risk_factors.append("poor work environment")
        recommendations.append("- **Enhance Environment:** Evaluate workspace and team dynamics.")

    if user_input.get("WorkLifeBalance", 3) <= 2:
        risk_factors.append("poor work-life balance")
        recommendations.append("- **Promote Balance:** Offer flexible hours or remote work options.")

    if user_input.get("MonthlyIncome", 5000) < 3000:
        risk_factors.append("below-average compensation")
        recommendations.append("- **Review Compensation:** Benchmark salary against market rates.")

    if user_input.get("YearsSinceLastPromotion", 0) >= 5:
        risk_factors.append("career stagnation")
        recommendations.append("- **Career Development:** Discuss promotion pathway, assign stretch projects.")

    if user_input.get("DistanceFromHome", 5) > 20:
        risk_factors.append("long commute")
        recommendations.append("- **Commute Solutions:** Offer remote/hybrid work or transport subsidies.")

    if probability >= 0.60:
        severity = "**HIGH ATTRITION RISK**"
    elif probability >= 0.30:
        severity = "**MODERATE ATTRITION RISK**"
    else:
        severity = "**LOW ATTRITION RISK**"

    summary = severity + "\n\n"
    if risk_factors:
        summary += f"Risk factors: **{', '.join(risk_factors)}**.\n\n"
        summary += "**Recommended Actions:**\n\n" + "\n".join(recommendations)
    else:
        summary += "No major risk factors detected. Continue current engagement strategies."

    return summary


def predict_single(user_input):
    """Run a prediction for a single employee."""
    model, scaler, metadata = load_model()
    feature_names = metadata["feature_names"]
    X = prepare_input(user_input, feature_names, scaler)

    prediction = model.predict(X)[0]
    try:
        probability = model.predict_proba(X)[0][1]
    except Exception:
        probability = float(prediction)

    risk_level, risk_color = get_risk_level(probability)
    recommendation = generate_recommendation(user_input, probability)

    return {
        "prediction": "Yes - Likely to Leave" if prediction == 1 else "No - Likely to Stay",
        "probability": round(probability * 100, 2),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "recommendation": recommendation,
    }


def predict_batch(df):
    """Run predictions for a batch of employees from CSV."""
    model, scaler, metadata = load_model()
    feature_names = metadata["feature_names"]

    results = []
    for _, row in df.iterrows():
        user_input = row.to_dict()
        X = prepare_input(user_input, feature_names, scaler)
        pred = model.predict(X)[0]
        try:
            prob = model.predict_proba(X)[0][1]
        except Exception:
            prob = float(pred)
        level, _ = get_risk_level(prob)
        results.append({
            "Attrition_Prediction": "Yes" if pred == 1 else "No",
            "Attrition_Probability": round(prob * 100, 2),
            "Risk_Level": level,
        })

    result_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), result_df.reset_index(drop=True)], axis=1)
