"""
============================================================
 FEATURE ENGINEERING -- src/feature_engineering.py
============================================================
 Creates domain-informed features to boost model performance.
============================================================
"""

import pandas as pd
import numpy as np


def create_features(df):
    """Add engineered features to the DataFrame."""
    df = df.copy()

    # 1. Income per year of experience (undervalued employees leave)
    df["IncomePerYearWorked"] = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)

    # 2. Promotion gap (stagnation signal)
    df["PromotionGap"] = df["YearsAtCompany"] - df["YearsSinceLastPromotion"]

    # 3. Satisfaction index (holistic happiness)
    sat_cols = [c for c in ["EnvironmentSatisfaction", "JobSatisfaction", "WorkLifeBalance"]
                if c in df.columns]
    if sat_cols:
        df["SatisfactionIndex"] = df[sat_cols].mean(axis=1)

    # 4. Loyalty index (career inertia)
    df["LoyaltyIndex"] = df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)

    # 5. Overtime flag (numeric)
    if "OverTime" in df.columns and df["OverTime"].dtype == "object":
        df["OvertimeFlag"] = (df["OverTime"] == "Yes").astype(int)

    # 6. Engagement score (emotional investment)
    eng_cols = [c for c in ["JobInvolvement", "RelationshipSatisfaction"] if c in df.columns]
    if eng_cols:
        df["EngagementScore"] = df[eng_cols].mean(axis=1)

    # 7. Career stagnation flag (junior + no promotion)
    if all(c in df.columns for c in ["YearsSinceLastPromotion", "JobLevel"]):
        df["CareerStagnation"] = ((df["YearsSinceLastPromotion"] >= 5) &
                                   (df["JobLevel"] <= 2)).astype(int)

    # 8. Age group (binned)
    if "Age" in df.columns:
        df["AgeGroup"] = pd.cut(df["Age"], bins=[17, 25, 35, 45, 55, 65],
                                labels=["18-25", "26-35", "36-45", "46-55", "56+"]).astype(str)

    print("[FEATURES] Created 8 engineered features")
    return df
