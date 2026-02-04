import streamlit as st
import pandas as pd
import joblib
import os
import argparse
import sys
from config import config

# Import your custom preprocessing script
try:
    import preprocess
except ImportError as e:
    st.error(f"Error importing preprocess.py: {e}")
    st.stop()

# --- Helper Functions ---
@st.cache_resource
def load_artifacts(model_name):
    """
    Loads the model and the feature list.
    """
    model_path = os.path.join(config.MODEL_PATH,f'{config.VERSION_NAME}_{model_name}.joblib')
    if not os.path.exists(model_path):
        return None
    try:
        artifacts = joblib.load(model_path)
        return artifacts
    except Exception as e:
        return None

def run_preprocess_pipeline(df, args):
    """
    Runs the individual functions from preprocess.py based on provided arguments.
    """
    try:
        # 1. Drop Duplicates
        if args.drop_duplicate_rows:
            df = preprocess.remove_duplicates(df)
            
        # 2. Handle Missing Values
        if args.handle_missing_values:
            df = preprocess.impute_missing_values(df)
        
        # 3. Handle Column Types (Conversion + Encoding)
        # In preprocess.py, these are grouped under one flag
        if args.handle_column_types:
            df = preprocess.convert_date_columns(df)
            df = preprocess.convert_boolean_columns(df)
            df = preprocess.convert_ordinal_columns(df)
            df = preprocess.convert_numeric_columns(df)
            df = preprocess.encode_list_columns(df)
            df = preprocess.encode_categorical_columns(df)
        
        # 4. Handle Outliers
        if args.handle_outliers:
            df = preprocess.handle_outliers(df)
        
        # 5. Final Cleanup (Always runs in preprocess.py)
        # This keeps only numeric columns and fills NaNs with 0
        df = preprocess.prepare_final_dataset(df)
        
        return df
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.write("Stacktrace:", e) # Helpful for debugging
        return None

def main(args):
    # --- Page Setup ---
    st.set_page_config(page_title="AirBnB Price Predictor", layout="centered")
    st.title("🏠 AirBnB Price Predictor")
    
    # Display current configuration
    st.caption(f"Model: `{args.model}`")
    
    # Show active flags
    active_flags = [k for k, v in vars(args).items() if v is True and k != 'model']
    if active_flags:
        st.info(f"Active Preprocessing Steps: {', '.join(active_flags)}")
    else:
        st.warning("No preprocessing flags set! Data will be passed largely 'as-is' (only numeric fields kept).")

    # --- Load Model ---
    pipeline = load_artifacts(args.model)

    if pipeline is None:
        st.error(f"'{args.model}' model not found! Please run 'invoke train_{args.model}' first.")
        return

    # model = artifacts['model']
    # expected_features = artifacts['features']
    st.success("Model & Schema loaded successfully.")

    uploaded_file = st.file_uploader("Upload Raw AirBnB CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read CSV
            input_df = pd.read_csv(uploaded_file, engine='python')
            st.write("### Raw Data Preview")
            st.dataframe(input_df.head())

            if st.button("Predict"):
                with st.spinner('Preprocessing and Predicting...'):
                    
                    # 1. Run preprocessing with CLI arguments
                    # processed_df = run_preprocess_pipeline(input_df.copy(), args)
                    processed_df = preprocess.preprocess_v2(input_df.copy(), args)
                    st.info(f"columns {len(processed_df.columns)}, {processed_df.columns}")
                    if processed_df is not None:
                        st.success("CSV format not aligned to the original schema.")
                    # if processed_df is not None:
                    #     # 2. SCHEMA ALIGNMENT
                    #     # Add missing columns (fill with 0)
                    #     missing_cols = set(expected_features) - set(processed_df.columns)
                    #     if missing_cols:
                    #         for c in missing_cols:
                    #             processed_df[c] = 0
                        
                    #     # Remove extra columns
                    #     extra_cols = set(processed_df.columns) - set(expected_features)
                        
                    #     # Reorder to match training
                    #     try:
                    #         processed_df = processed_df[expected_features]
                    #     except KeyError as e:
                    #         st.error(f"Column alignment error: {e}")
                    #         st.stop()

                        # 3. Predict
                        try:
                            predictions = pipeline.predict(processed_df)
                            
                            output_df = pd.DataFrame(predictions, columns=['predicted'])
                            
                            st.write("### Predictions")
                            st.dataframe(output_df.head())
                            
                            # Download
                            csv = output_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Predictions",
                                data=csv,
                                file_name='predictions.csv',
                                mime='text/csv',
                            )
                        except Exception as e:
                            st.error(f"Prediction logic failed: {e}")
        except Exception as e:
            st.error(f"Error reading file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Streamlit App with specific preprocessing args")
    
    # Model arg
    parser.add_argument("--model", type=str, default="xgboost",choices=["linear", "ridge", "random_forest", "xgboost"], help="Path to model bundle")

    # Preprocess.py args (Defaults match preprocess.py)
    parser.add_argument("--drop-duplicate-rows"  ,    action='store_true', default=False)
    parser.add_argument("--handle-column-types"  ,    action='store_true', default=False)
    parser.add_argument("--handle-missing-values",  action='store_true', default=False)
    parser.add_argument("--handle-outliers"      ,        action='store_true', default=False)
    
    # Note: LLM Sentiment analysis omitted for inference app to avoid async complexity/API requirements
    
    try:
        args = parser.parse_args()
    except SystemExit as e:
        os._exit(e.code)
        
    main(args)