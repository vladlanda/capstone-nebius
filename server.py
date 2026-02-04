import streamlit as st
import pandas as pd
import joblib
import os

# --- Configuration ---
MODEL_FILE = 'airbnb_model.pkl'

# --- Page Setup ---
st.set_page_config(page_title="AirBnB Price Predictor", layout="centered")

st.title("🏠 AirBnB Price Predictor")
st.markdown("""
This app accepts a CSV file with AirBnB listing details and outputs price predictions.
""")

# --- Helper Functions ---
@st.cache_resource
def load_model():
    """
    Loads the model from disk.
    Uses st.cache_resource to avoid reloading the model on every interaction.
    """
    if not os.path.exists(MODEL_FILE):
        return None
    try:
        model = joblib.load(MODEL_FILE)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def convert_df(df):
    """
    Converts a dataframe to CSV for download.
    """
    return df.to_csv(index=False).encode('utf-8')

# --- Main Logic ---

# 1. Load the Model
pipeline = load_model()

if pipeline is None:
    st.error(f"Model file '{MODEL_FILE}' not found! Please run 'generate_model.py' first or upload your own model to the directory.")
    st.stop() # Stop execution if model is missing

st.success("Model loaded successfully.")

# 2. File Uploader
uploaded_file = st.file_uploader("Upload your AirBnB CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the CSV
        input_df = pd.read_csv(uploaded_file)
        
        st.write("### Data Preview")
        st.dataframe(input_df.head())

        # 3. Generate Predictions
        # Note: We assume the pipeline handles preprocessing (encoding/scaling) internally.
        # If your pipeline expects raw numpy arrays, this needs adjustment.
        # Sklearn pipelines generally handle pandas DataFrames well if column names match training.
        
        if st.button("Generate Predictions"):
            with st.spinner('Predicting prices...'):
                try:
                    predictions = pipeline.predict(input_df)
                    
                    # 4. Format Output
                    # Requirements: Single column, same row order.
                    output_df = pd.DataFrame(predictions, columns=['predicted_price'])
                    
                    st.write("### Predictions Preview")
                    st.dataframe(output_df.head())
                    
                    # 5. Download Button
                    csv = convert_df(output_df)
                    
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name='airbnb_predictions.csv',
                        mime='text/csv',
                    )
                    
                except Exception as e:
                    st.error(f"Prediction failed. Error: {e}")
                    st.warning("Ensure your uploaded CSV columns match exactly what the model was trained on.")
                    
    except Exception as e:
        st.error(f"Error reading file: {e}")

else:
    st.info("Awaiting CSV file upload...")