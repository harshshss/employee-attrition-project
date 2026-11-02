"""
============================================================
 DATA PREPROCESSING -- src/preprocessing.py
============================================================
 Handles raw-data to model-ready transformation:
 Loading, Cleaning, Encoding, Scaling, Outlier Detection,
 Correlation Analysis, and EDA Plot Generation.
============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.utils import get_project_root, save_plot, print_section_header


def load_data(path):
    """Load the HR Analytics CSV into a DataFrame."""
    df = pd.read_csv(path)
    print(f"[LOAD] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def clean_data(df):
    """Remove duplicates, handle nulls, drop constant/ID columns."""
    initial_shape = df.shape
    df = df.drop_duplicates()
    if df.isnull().sum().sum() > 0:
        for col in df.columns:
            if df[col].dtype in ["float64", "int64"]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    drop_cols = [c for c in ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
                 if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")
    print(f"[CLEAN] {initial_shape} -> {df.shape}  |  Dropped: {drop_cols}")
    return df


def encode_target(df, target="Attrition"):
    """Binary-encode the target column (Yes=1, No=0)."""
    if target in df.columns and not pd.api.types.is_numeric_dtype(df[target]):
        df[target] = df[target].map({"Yes": 1, "No": 0})
        print(f"[ENCODE] Target '{target}' -> binary (Yes=1, No=0)")
    return df


def encode_features(df):
    """One-Hot Encode all remaining categorical features."""
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    encoding_map = {}
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
        for col in cat_cols:
            new_cols = [c for c in df.columns if c.startswith(col + "_")]
            encoding_map[col] = new_cols
        print(f"[ENCODE] One-Hot Encoded: {cat_cols}")
    return df, encoding_map


def scale_features(X):
    """Standardise numeric features to zero mean, unit variance."""
    scaler = StandardScaler()
    feature_names = X.columns.tolist()
    X_scaled = scaler.fit_transform(X)
    print(f"[SCALE] Scaled {len(feature_names)} features with StandardScaler")
    return X_scaled, scaler, feature_names


def detect_outliers(df, threshold=1.5):
    """Flag outliers using the IQR method."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    records = []
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - threshold * IQR, Q3 + threshold * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        records.append({"Feature": col, "Lower Bound": round(lower, 2),
                        "Upper Bound": round(upper, 2), "Outlier Count": n_outliers,
                        "Outlier %": round(n_outliers / len(df) * 100, 2)})
    return pd.DataFrame(records)


def generate_correlation_heatmap(df):
    """Create and save a correlation heatmap."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(18, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(230, 20, as_cmap=True),
                vmax=1, vmin=-1, center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.7}, annot=False, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=15)
    path = save_plot(fig, "correlation_heatmap.png")
    print("[PLOT] Correlation heatmap -> " + path)
    return path


def generate_distribution_plots(df):
    """Create distribution histograms for key numeric features."""
    features = [f for f in ["Age", "MonthlyIncome", "DistanceFromHome",
                            "YearsAtCompany", "TotalWorkingYears"] if f in df.columns]
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 5))
    if len(features) == 1:
        axes = [axes]
    for ax, feat in zip(axes, features):
        sns.histplot(df[feat], kde=True, ax=ax, color="#4C72B0", edgecolor="white")
        ax.set_title(feat, fontsize=13, fontweight="bold")
        ax.set_xlabel("")
    fig.suptitle("Feature Distributions", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = save_plot(fig, "feature_distributions.png")
    print("[PLOT] Distributions -> " + path)
    return [path]


def generate_attrition_charts(df):
    """Create attrition comparison charts for categorical features."""
    cat_features = [f for f in ["OverTime", "JobRole", "MaritalStatus",
                                "Department", "BusinessTravel"] if f in df.columns]
    paths = []
    for feat in cat_features:
        fig, ax = plt.subplots(figsize=(10, 5))
        ct = pd.crosstab(df[feat], df["Attrition"], normalize="index") * 100
        ct.plot(kind="bar", stacked=True, ax=ax,
                color=["#2ecc71", "#e74c3c"], edgecolor="white")
        ax.set_title(f"Attrition Rate by {feat}", fontsize=14, fontweight="bold")
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel("")
        ax.legend(["No", "Yes"], title="Attrition", loc="upper right")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        path = save_plot(fig, f"attrition_by_{feat.lower()}.png")
        paths.append(path)
    print(f"[PLOT] Attrition charts -> {len(paths)} saved")
    return paths


def run_preprocessing_pipeline(csv_path):
    """Execute the complete preprocessing pipeline end-to-end."""
    print_section_header("DATA PREPROCESSING PIPELINE")

    df_raw = load_data(csv_path)
    df_clean = clean_data(df_raw.copy())

    heatmap_path = generate_correlation_heatmap(df_clean)
    dist_paths = generate_distribution_plots(df_clean)
    attrition_paths = generate_attrition_charts(df_clean)
    outlier_report = detect_outliers(df_clean)

    df_encoded = encode_target(df_clean.copy())
    df_encoded, encoding_map = encode_features(df_encoded)

    X = df_encoded.drop(columns=["Attrition"])
    y = df_encoded["Attrition"]
    X_scaled, scaler, feature_names = scale_features(X)

    print_section_header("PREPROCESSING COMPLETE")
    print(f"  Features: {X_scaled.shape[1]}  |  Samples: {X_scaled.shape[0]}")

    return {
        "df_raw": df_raw, "df_clean": df_clean, "df_encoded": df_encoded,
        "X": X, "X_scaled": X_scaled, "y": y, "scaler": scaler,
        "feature_names": feature_names, "encoding_map": encoding_map,
        "outlier_report": outlier_report,
        "plot_paths": [heatmap_path] + dist_paths + attrition_paths,
    }
