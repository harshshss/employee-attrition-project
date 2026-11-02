# Interview Preparation Guide
## Employee Attrition Prediction System

---

## Common Interview Questions & Best Answers

### 1. "Tell me about your project."

> "I built an end-to-end Employee Attrition Prediction System that predicts whether
> an employee is likely to leave a company. I used the IBM HR Analytics dataset with
> 1,470 records and 35 features. The pipeline includes data preprocessing, feature
> engineering (8 custom features), and training 5 ML models. The best model — XGBoost
> — achieved 96.94% accuracy and 90.72% F1 Score. I also built an interactive
> Streamlit dashboard with real-time predictions, batch prediction via CSV upload,
> and automated retention recommendations."

### 2. "Why did you choose XGBoost as your best model?"

> "XGBoost consistently outperformed other models because it handles class imbalance
> well (only ~16% of employees actually leave), captures non-linear feature interactions,
> and has built-in regularisation to prevent overfitting. I compared it against Logistic
> Regression, Decision Tree, Random Forest, and Gradient Boosting using F1 Score as the
> primary metric — because in attrition prediction, catching true departures (recall)
> is as important as avoiding false alarms (precision)."

### 3. "Why F1 Score instead of Accuracy?"

> "The dataset is imbalanced — about 84% of employees stay and only 16% leave. A naive
> model predicting 'No' for everyone would get 84% accuracy but miss every actual departure.
> F1 Score is the harmonic mean of precision and recall, giving us a balanced view of
> both. In HR, a False Negative (missing someone who will leave) is very costly — it means
> losing talent you could have retained."

### 4. "What feature engineering did you do?"

> "I created 8 engineered features:
> - **IncomePerYearWorked** — flags undervalued employees
> - **PromotionGap** — detects career stagnation
> - **SatisfactionIndex** — composite happiness score
> - **LoyaltyIndex** — career inertia measure
> - **EngagementScore** — emotional investment
> - **CareerStagnation flag** — binary flag for junior employees stuck without promotion
> - **OvertimeFlag** — numeric conversion for calculations
> - **AgeGroup** — binned age for life-stage patterns"

### 5. "How did you handle class imbalance?"

> "I used class_weight='balanced' in Logistic Regression, Decision Tree, and Random
> Forest. For XGBoost, I used scale_pos_weight=5 to upweight the minority class. I also
> used stratified train-test splitting to preserve class ratios."

### 6. "What preprocessing steps did you perform?"

> "I cleaned the data by removing duplicates and handling missing values. I dropped
> constant columns (EmployeeCount, StandardHours, Over18) and the ID column. I used
> One-Hot Encoding for categorical features and StandardScaler for numerical features.
> I also performed IQR-based outlier detection."

### 7. "Why StandardScaler instead of MinMaxScaler?"

> "StandardScaler (zero mean, unit variance) works better with gradient-based models
> like Logistic Regression and XGBoost because they converge faster with standardised
> features. MinMaxScaler is sensitive to outliers, which HR data commonly contains
> (e.g., extreme salaries)."

---

## Resume Bullet Points

- Built an end-to-end Employee Attrition Prediction System using Python, Scikit-learn,
  XGBoost, and Streamlit, achieving 96.94% accuracy and 90.72% F1 Score
- Engineered 8 domain-informed features (IncomePerYearWorked, SatisfactionIndex,
  CareerStagnation) that improved model performance by capturing non-obvious attrition patterns
- Designed and deployed an interactive Streamlit dashboard with real-time predictions,
  batch CSV processing, and automated HR retention recommendations
- Compared 5 ML algorithms (Logistic Regression, Decision Tree, Random Forest, Gradient
  Boosting, XGBoost) using comprehensive metrics including ROC-AUC, Precision, Recall, and F1

---

## LinkedIn Project Description

**Employee Attrition Prediction System**

An AI-powered machine learning system that predicts employee turnover and provides
actionable retention strategies for HR teams.

Key Highlights:
- Trained and compared 5 classification models on IBM HR Analytics data (1,470 employees)
- Best model (XGBoost) achieved 96.94% accuracy and 90.72% F1 Score
- Built a professional Streamlit dashboard with interactive EDA, model comparison,
  real-time predictions, and batch processing
- Engineered 8 custom features including SatisfactionIndex and CareerStagnation flag
- Implemented automated recommendation engine for employee retention strategies

Tech: Python | Scikit-learn | XGBoost | Streamlit | Plotly | Pandas
