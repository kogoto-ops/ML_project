import datetime
import io
import requests
import pandas as pd
import streamlit as st
import os

# Configuration
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8011")

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Used Car Auction Price Predictor")
st.markdown("Predict vehicle prices based on auction data")

# ---------- SIDEBAR: HEALTH CHECK ----------
with st.sidebar:
    st.header("🔧 System Status")
    if st.button("🔁 Check Backend Health"):
        try:
            response = requests.get(f"{BACKEND_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ Backend is healthy")
                st.json(response.json())
            else:
                st.error(f"❌ Backend error: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend. Make sure it's running on port 8000")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ---------- MAIN: SINGLE PREDICTION ----------
st.header("🔍 Single Vehicle Prediction")

with st.form("single_prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_year = datetime.datetime.now().year
        year = st.number_input("Year", min_value=1990, max_value=current_year, value=2015)
        condition = st.slider("Condition (1-50)", 1.0, 50.0, 35.0, 1.0)
        odometer = st.number_input("Odometer (miles)", min_value=0, max_value=300000, value=50000)
        
        make = st.text_input("Make", value="Ford")
        model = st.text_input("Model", value="Explorer")
        trim = st.text_input("Trim", value="XLT")
    
    with col2:
        body = st.selectbox(
            "Body Type",
            ["SUV", "Sedan", "Truck", "Wagon", "Coupe", "Convertible", "Van"]
        )
        transmission = st.selectbox(
            "Transmission",
            ["automatic", "manual"]
        )
        state = st.text_input("State (abbreviation)", value="tx")
        
        color = st.text_input("Color", value="black")
        interior = st.text_input("Interior Color", value="black")
    
    with col3:
        saledate = st.text_input(
            "Sale Date",
            value="Thu Mar 05 2015 01:00:00 GMT-0800 (PST)",
            help="Format: 'Thu Mar 05 2015 01:00:00 GMT-0800 (PST)'"
        )
        st.markdown("---")
        st.markdown("*Example date format shown above*")
    
    submitted = st.form_submit_button("💰 Predict Price", use_container_width=True)
    
    if submitted:
        payload = {
            "year": int(year),
            "condition": float(condition),
            "odometer": int(odometer),
            "make": make,
            "model": model,
            "trim": trim,
            "body": body,
            "transmission": transmission,
            "state": state,
            "color": color,
            "interior": interior,
            "saledate": saledate
        }
        
        with st.spinner("Predicting..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/predict",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    prediction = response.json()["prediction"]
                    st.success(f"## 💵 Predicted Selling Price: **${prediction:,.2f}**")
                else:
                    st.error(f"Prediction failed: {response.text}")
            
            except Exception as e:
                st.error(f"Request error: {e}")

# ---------- BATCH PREDICTION ----------
st.header("📁 Batch Prediction (CSV Upload)")
st.markdown("Upload a CSV file with the same columns as the training data (without sellingprice)")

# Use Streamlit's session state to persist results across execution loops safely
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="File must contain: year, make, model, trim, body, transmission, state, condition, odometer, color, interior, saledate"
)

if uploaded_file is not None:
    df_preview = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df_preview.head(), use_container_width=True)
    st.caption(f"Total rows: {len(df_preview)}")
    
    if st.button("🚀 Run Batch Prediction", use_container_width=True):
        uploaded_file.seek(0)
        files = {"file": uploaded_file}
        
        with st.spinner("Processing batch predictions..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/predict_batch",
                    files=files,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    predictions = result["predictions"]
                    
                    # SAFE INDEX ALIGNMENT MATCH
                    # If your API filters out rows internally, we preserve layout integrity:
                    df_output = df_preview.copy()
                    
                    if len(predictions) == len(df_output):
                        df_output["sellingprice"] = predictions
                    else:
                        # Fallback if rows were dropped due to empty date handling:
                        st.warning("⚠️ Row counts changed due to server validation cleaning. Appending predictions by sequential order.")
                        df_output = df_output.dropna(subset=['saledate']).copy()
                        df_output["sellingprice"] = predictions

                    # Cache directly inside the session state to survive clicks
                    st.session_state.batch_results = df_output
                    st.success(f"✅ Predictions complete! {result['count']} vehicles processed")
                    
                else:
                    st.error(f"Batch prediction failed: {response.text}")
                    st.session_state.batch_results = None
            
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.batch_results = None

    # Render results outside the button block to allow persistent downloads
    if st.session_state.batch_results is not None:
        st.subheader("📈 Prediction Results Preview")
        st.dataframe(st.session_state.batch_results.head(10), use_container_width=True)
        
        csv_output = st.session_state.batch_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV with Predictions",
            data=csv_output,
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Car Price Predictor | Uses Random Forest model | MAE: $1,861
    </div>
    """,
    unsafe_allow_html=True
)
