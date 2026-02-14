import streamlit as st
import pandas as pd
import predict as predict_lib


@st.cache_resource
def cached_load_artifacts(model_name: str):
    return predict_lib.load_artifacts(model_name)


def main():
    model_name = "catboost"
    # --- Page Setup ---
    st.set_page_config(page_title="AirBnB Price Predictor", layout="centered")
    st.title("🏠 AirBnB Price Predictor")

    # Display current configuration
    st.caption(f"Model: `{model_name}`")

    # --- Load Model ---
    if cached_load_artifacts(model_name) is None:
        st.error(f"'{model_name}' model not found! Please run 'invoke train_{model_name}' first.")
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
                            model_name,
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
    main()
