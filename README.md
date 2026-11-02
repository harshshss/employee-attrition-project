# 🏢 Employee Attrition Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-f7931e?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-006600?style=for-the-badge)

**An AI-powered machine learning system to predict and prevent employee turnover**

[Features](#-features) · [Installation](#-installation) · [Usage](#-usage) · [Model Performance](#-model-performance) · [Tech Stack](#-tech-stack)

</div>

---

## 📋 Overview

Employee attrition is one of the most expensive challenges facing modern organisations. This project builds a **complete end-to-end machine learning pipeline** that:

1. **Preprocesses** HR analytics data with professional-grade cleaning, encoding, and scaling
2. **Engineers** 8 domain-informed features to boost model performance
3. **Trains & compares** 5 classification algorithms (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost)
4. **Deploys** the best model via a beautiful, interactive Streamlit dashboard
5. **Provides** actionable retention recommendations for HR decision-makers

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 **Interactive EDA** | Plotly-powered charts with filtering and drill-down |
| 🤖 **Multi-Model Comparison** | 5 algorithms evaluated on Accuracy, Precision, Recall, F1, ROC-AUC |
| 🔮 **Real-Time Prediction** | Enter employee details → instant attrition probability |
| 📁 **Batch Prediction** | Upload CSV → predictions for all employees at once |
| 💡 **Smart Recommendations** | Context-aware retention advice based on risk factors |
| 🎨 **Modern Dashboard** | Dark-themed, responsive UI with KPI cards and animations |
| 📥 **Export Results** | Download prediction reports as CSV |

---

## 🛠️ Tech Stack

- **Language:** Python 3.9+
- **Data:** Pandas, NumPy
- **ML:** Scikit-learn, XGBoost
- **Visualisation:** Plotly, Matplotlib, Seaborn
- **Dashboard:** Streamlit
- **Serialisation:** Joblib

---

## 📁 Project Structure

```
Employee_Attrition_Project/
├── data/                    # Dataset
│   └── HR_Analytics.csv
├── models/                  # Trained model artifacts
│   ├── attrition_model.pkl
│   ├── scaler.pkl
│   └── model_metadata.json
├── src/                     # Source modules
│   ├── utils.py             # Shared utilities
│   ├── preprocessing.py     # Data cleaning & encoding
│   ├── feature_engineering.py
│   ├── train_model.py       # ML training pipeline
│   └── predict.py           # Prediction engine
├── app/
│   └── streamlit_app.py     # Dashboard
├── outputs/
│   ├── plots/               # Generated visualisations
│   └── reports/             # Model results CSV
├── main.py                  # Pipeline entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Employee_Attrition_Project.git
cd Employee_Attrition_Project

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### Step 1: Train the Models
```bash
python main.py
```
This will generate the dataset, preprocess data, engineer features, train 5 models, and save the best one.

### Step 2: Launch the Dashboard
```bash
streamlit run app/streamlit_app.py
```
Open `http://localhost:8501` in your browser.

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~82% | ~55% | ~60% | ~57% |
| Decision Tree | ~79% | ~48% | ~55% | ~51% |
| Random Forest | ~86% | ~62% | ~58% | ~60% |
| **XGBoost** | **~87%** | **~65%** | **~62%** | **~63%** |

> *Results may vary slightly due to random seed and synthetic data generation.*

---

## 🔮 Feature Engineering

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| IncomePerYearWorked | Income / Experience | Undervalued employees leave |
| PromotionGap | Years at company − years since promotion | Stagnation → attrition |
| SatisfactionIndex | Average of 3 satisfaction scores | Holistic happiness measure |
| LoyaltyIndex | Years at company / Total career | High loyalty = low risk |
| EngagementScore | Involvement + Relationship satisfaction | Emotional investment |
| CareerStagnation | No promotion in 5+ yrs & junior level | High flight risk flag |

---

## 🚀 Future Improvements

- [ ] SHAP explainability visualisations
- [ ] Deep learning (neural network) model
- [ ] FastAPI REST endpoint for production deployment
- [ ] Real-time database integration
- [ ] Automated retraining pipeline

---

## 👤 Author

**Your Name**
- 📧 Email: your.email@example.com
- 💼 LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- 🐙 GitHub: [Your GitHub](https://github.com/yourusername)

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

*Built with ❤️ for HR Analytics*

</div>
