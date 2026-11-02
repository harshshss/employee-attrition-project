"""
============================================================
 MODEL TRAINING -- src/train_model.py
============================================================
 Trains multiple classifiers, evaluates, compares, and saves
 the best model.
============================================================
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.utils import get_project_root, save_plot, print_section_header


def get_models():
    """Return a dictionary of model name -> untrained estimator."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42,
            class_weight="balanced", n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, use_label_encoder=False,
            eval_metric="logloss", scale_pos_weight=5)
    return models


def evaluate_model(model, X_test, y_test, model_name):
    """Compute all evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    except Exception:
        roc_auc = None

    metrics = {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "F1 Score": round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
    }
    if roc_auc is not None:
        metrics["ROC-AUC"] = round(roc_auc * 100, 2)
    return metrics


def plot_confusion_matrix(model, X_test, y_test, model_name):
    """Generate and save a confusion matrix heatmap."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Stay", "Leave"], yticklabels=["Stay", "Leave"],
                ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    return save_plot(fig, filename)


def plot_model_comparison(results_df):
    """Create a grouped bar chart comparing all models."""
    metrics = [m for m in ["Accuracy", "Precision", "Recall", "F1 Score"]
               if m in results_df.columns]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results_df))
    width = 0.18
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]

    for i, metric in enumerate(metrics):
        bars = ax.bar(x + i * width, results_df[metric], width,
                      label=metric, color=colors[i], edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=16, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df["Model"], rotation=15, ha="right")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return save_plot(fig, "model_comparison.png")


def plot_feature_importance(model, feature_names, model_name):
    """Plot top 15 most important features for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return ""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(indices)), importances[indices], color="#3498db", edgecolor="white")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top 15 Feature Importances - {model_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    filename = f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    return save_plot(fig, filename)


def run_training_pipeline(X_scaled, y, feature_names, scaler):
    """Execute the complete model training and evaluation pipeline."""
    print_section_header("MODEL TRAINING PIPELINE")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    print(f"[SPLIT] Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    models = get_models()
    results = []
    trained_models = {}
    confusion_paths = []
    importance_paths = []

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        print(f"    Accuracy: {metrics['Accuracy']}%  |  F1: {metrics['F1 Score']}%")

        cm_path = plot_confusion_matrix(model, X_test, y_test, name)
        confusion_paths.append(cm_path)

        fi_path = plot_feature_importance(model, feature_names, name)
        if fi_path:
            importance_paths.append(fi_path)

    results_df = pd.DataFrame(results)
    comparison_path = plot_model_comparison(results_df)

    best_idx = results_df["F1 Score"].idxmax()
    best_name = results_df.loc[best_idx, "Model"]
    best_model = trained_models[best_name]
    print(f"\n  * Best Model: {best_name} (F1: {results_df.loc[best_idx, 'F1 Score']}%)")

    root = get_project_root()
    model_dir = os.path.join(root, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "attrition_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    metadata_path = os.path.join(model_dir, "model_metadata.json")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "best_model": best_name,
        "feature_names": feature_names,
        "metrics": results_df.loc[best_idx].to_dict(),
        "all_results": results,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    report_path = os.path.join(root, "outputs", "reports", "model_results.csv")
    results_df.to_csv(report_path, index=False)

    print_section_header("TRAINING COMPLETE")
    print(f"  Best model saved -> {model_path}")
    print(f"  Scaler saved     -> {scaler_path}")

    return {
        "results_df": results_df, "best_model_name": best_name,
        "best_model": best_model, "trained_models": trained_models,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "model_path": model_path, "scaler_path": scaler_path,
        "comparison_path": comparison_path,
        "confusion_paths": confusion_paths,
        "importance_paths": importance_paths,
    }
