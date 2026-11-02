"""
============================================================
 UTILITY FUNCTIONS -- src/utils.py
============================================================
 Shared helper functions used throughout the project.
============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_project_root():
    """Return the absolute path to the project root directory."""
    current = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(current) in ("src", "app"):
        return os.path.dirname(current)
    return current


def setup_directories():
    """Create all required output directories if they don't exist."""
    root = get_project_root()
    for d in ["data", "models", os.path.join("outputs", "plots"),
              os.path.join("outputs", "reports")]:
        os.makedirs(os.path.join(root, d), exist_ok=True)


def save_plot(fig, filename, dpi=150):
    """Save a matplotlib figure to outputs/plots/ and return its path."""
    root = get_project_root()
    path = os.path.join(root, "outputs", "plots", filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def format_percentage(value, decimals=2):
    """Convert a float to a percentage string."""
    if value <= 1.0:
        value *= 100
    return f"{value:.{decimals}f}%"


def print_section_header(title):
    """Print a styled section header to the console."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {title.upper()}")
    print("=" * width)


def generate_dataset(output_path=None):
    """Generate a realistic IBM HR Analytics Employee Attrition dataset.

    Creates 1,470 employee records with 35 columns modelled after
    the well-known Kaggle IBM HR Analytics dataset.
    """
    if output_path is None:
        root = get_project_root()
        output_path = os.path.join(root, "data", "HR_Analytics.csv")

    if os.path.exists(output_path):
        print("[INFO] Dataset already exists at " + output_path)
        return output_path

    setup_directories()
    np.random.seed(42)
    n = 1470

    age = np.random.randint(18, 60, n)
    daily_rate = np.random.randint(100, 1500, n)
    distance_from_home = np.random.randint(1, 30, n)
    education = np.random.randint(1, 5, n)
    environment_satisfaction = np.random.randint(1, 5, n)
    hourly_rate = np.random.randint(30, 100, n)
    job_involvement = np.random.randint(1, 5, n)
    job_level = np.random.randint(1, 6, n)
    job_satisfaction = np.random.randint(1, 5, n)
    monthly_income = (job_level * 2000 + np.random.randint(-500, 2500, n)).clip(1000, 20000)
    monthly_rate = np.random.randint(2000, 27000, n)
    num_companies_worked = np.random.randint(0, 10, n)
    percent_salary_hike = np.random.randint(11, 25, n)
    performance_rating = np.random.choice([3, 4], n, p=[0.85, 0.15])
    relationship_satisfaction = np.random.randint(1, 5, n)
    stock_option_level = np.random.randint(0, 4, n)
    total_working_years = (age - 18 - np.random.randint(0, 5, n)).clip(0, 40)
    training_times_last_year = np.random.randint(0, 7, n)
    work_life_balance = np.random.randint(1, 5, n)
    years_at_company = np.minimum(np.random.randint(0, 20, n), total_working_years)
    years_in_current_role = np.minimum(np.random.randint(0, 15, n), years_at_company)
    years_since_last_promotion = np.minimum(np.random.randint(0, 10, n), years_at_company)
    years_with_curr_manager = np.minimum(np.random.randint(0, 15, n), years_at_company)

    business_travel = np.random.choice(
        ["Travel_Rarely", "Travel_Frequently", "Non-Travel"], n, p=[0.70, 0.19, 0.11])
    department = np.random.choice(
        ["Research & Development", "Sales", "Human Resources"], n, p=[0.65, 0.30, 0.05])
    education_field = np.random.choice(
        ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
        n, p=[0.35, 0.25, 0.10, 0.15, 0.05, 0.10])
    gender = np.random.choice(["Male", "Female"], n, p=[0.60, 0.40])
    job_role = np.random.choice(
        ["Sales Executive", "Research Scientist", "Laboratory Technician",
         "Manufacturing Director", "Healthcare Representative",
         "Manager", "Sales Representative", "Research Director", "Human Resources"],
        n, p=[0.20, 0.20, 0.18, 0.10, 0.09, 0.07, 0.06, 0.06, 0.04])
    marital_status = np.random.choice(["Married", "Single", "Divorced"], n, p=[0.46, 0.32, 0.22])
    overtime = np.random.choice(["Yes", "No"], n, p=[0.30, 0.70])

    employee_count = np.ones(n, dtype=int)
    standard_hours = np.full(n, 80, dtype=int)
    over_18 = np.full(n, "Y")
    employee_number = np.arange(1, n + 1)

    # Target: Attrition (logistic probability from risk factors)
    risk_score = (
        -0.5
        + 0.02 * (30 - age).clip(0, None)
        + 0.04 * (overtime == "Yes").astype(float)
        + 0.03 * (4 - job_satisfaction)
        + 0.02 * (4 - environment_satisfaction)
        + 0.02 * (4 - work_life_balance)
        + 0.015 * distance_from_home
        + 0.01 * (3 - job_involvement)
        - 0.005 * monthly_income / 1000
        + 0.01 * num_companies_worked
        - 0.005 * years_at_company
        + 0.02 * (marital_status == "Single").astype(float)
    )
    prob = 1 / (1 + np.exp(-risk_score))
    threshold = np.percentile(prob, 84)
    attrition = np.where(prob >= threshold, "Yes", "No")

    df = pd.DataFrame({
        "Age": age, "Attrition": attrition, "BusinessTravel": business_travel,
        "DailyRate": daily_rate, "Department": department,
        "DistanceFromHome": distance_from_home, "Education": education,
        "EducationField": education_field, "EmployeeCount": employee_count,
        "EmployeeNumber": employee_number,
        "EnvironmentSatisfaction": environment_satisfaction,
        "Gender": gender, "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement, "JobLevel": job_level,
        "JobRole": job_role, "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status, "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate, "NumCompaniesWorked": num_companies_worked,
        "OverTime": overtime, "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StandardHours": standard_hours, "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
        "Over18": over_18,
    })

    df.to_csv(output_path, index=False)
    print("[INFO] Dataset generated -> " + output_path + "  (" + str(n) + " records)")
    return output_path
