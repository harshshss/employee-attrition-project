"""
============================================================
 MAIN PIPELINE -- main.py
============================================================
 Entry point that orchestrates the complete ML pipeline:

   1. Generate / load dataset
   2. Run preprocessing pipeline
   3. Apply feature engineering
   4. Train and evaluate models
   5. Save the best model

 Usage:
   python main.py
============================================================
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import (
    setup_directories, generate_dataset, print_section_header,
)
from src.preprocessing import run_preprocessing_pipeline
from src.feature_engineering import create_features
from src.train_model import run_training_pipeline


def main():
    """Execute the full Employee Attrition Prediction pipeline."""
    print_section_header("EMPLOYEE ATTRITION PREDICTION SYSTEM")
    print("  Building complete ML pipeline...")

    # Step 0: Setup
    setup_directories()

    # Step 1: Dataset
    print_section_header("STEP 1: DATASET GENERATION")
    csv_path = generate_dataset()
    print("  Dataset ready at: " + csv_path)

    # Step 2: Preprocessing + EDA
    print_section_header("STEP 2: PREPROCESSING & EDA")
    pipeline_output = run_preprocessing_pipeline(csv_path)

    # Step 3: Feature Engineering (re-process with engineered features)
    print_section_header("STEP 3: FEATURE ENGINEERING")

    from src.preprocessing import (
        load_data, clean_data, encode_target,
        encode_features, scale_features,
    )

    df_raw = load_data(csv_path)
    df_clean = clean_data(df_raw.copy())
    df_featured = create_features(df_clean)
    df_encoded = encode_target(df_featured.copy())
    df_encoded, encoding_map = encode_features(df_encoded)

    target_col = "Attrition"
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    X_scaled, scaler, feature_names = scale_features(X)

    # Step 4: Model Training
    print_section_header("STEP 4: MODEL TRAINING")
    training_output = run_training_pipeline(X_scaled, y, feature_names, scaler)

    # Summary
    print_section_header("PIPELINE COMPLETE")
    print("  Best Model   : " + training_output["best_model_name"])
    print("  Model Saved  : " + training_output["model_path"])
    print("")
    print("  Next: streamlit run app/streamlit_app.py")

    return training_output


if __name__ == "__main__":
    main()
