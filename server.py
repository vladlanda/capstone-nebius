import streamlit as st
import pandas as pd
import os
import argparse
import predict as predict_lib


@st.cache_resource
def cached_load_artifacts(model_name: str):
    return predict_lib.load_artifacts(model_name)


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
    if cached_load_artifacts(args.model) is None:
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

                    try:
                        output_df = predict_lib.apply_model(
                            input_df,
                            args.model,
                        )

                        st.write("### Predictions")
                        st.dataframe(output_df.head())

                        csv = output_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download Predictions",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
        except Exception as e:
            st.error(f"Error reading file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Streamlit App with specific preprocessing args")

    # Model arg
    parser.add_argument("--model", type=str, default="xgboost",choices=["linear", "ridge", "random_forest", "xgboost"], help="Path to model bundle")

    # Preprocess.py args (Defaults match preprocess.py)

    # Note: LLM Sentiment analysis omitted for inference app to avoid async complexity/API requirements

    try:
        args = parser.parse_args()
    except SystemExit as e:
        os._exit(e.code)

    main(args)
