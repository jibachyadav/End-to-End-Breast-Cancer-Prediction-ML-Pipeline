"""
Breast Cancer Survival Prediction — Streamlit Frontend
"""

import streamlit as st
import requests
import os

st.set_page_config(
    page_title="Breast Cancer Survival Prediction",
    layout="centered"
)

st.title("Breast Cancer Survival Prediction")
st.divider()

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    race = st.selectbox("Race", ["White", "Black", "Other"])
    marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed", "Separated"])
    differentiate = st.selectbox("Differentiate", ["Well differentiated", "Moderately differentiated", "Poorly differentiated", "Undifferentiated"])
    a_stage = st.selectbox("A Stage", ["Regional", "Distant"])

with col2:
    tumor_size = st.number_input("Tumor Size (mm)", min_value=0.0, max_value=200.0, value=30.0)
    estrogen_status = st.selectbox("Estrogen Status", ["Positive", "Negative"])
    progesterone_status = st.selectbox("Progesterone Status", ["Positive", "Negative"])
    regional_node_examined = st.number_input("Regional Nodes Examined", min_value=0, max_value=100, value=12)
    regional_node_positive = st.number_input("Regional Nodes Positive", min_value=0, max_value=100, value=1)

st.divider()

if st.button("Predict Survival", use_container_width=True, type="primary"):

    if regional_node_positive > regional_node_examined:
        st.error("Regional Nodes Positive cannot be greater than Regional Nodes Examined!")
    else:
        patient_data = {
            "age"                    : age,
            "race"                   : race,
            "marital_status"         : marital_status,
            "differentiate"          : differentiate,
            "a_stage"                : a_stage,
            "tumor_size"             : tumor_size,
            "estrogen_status"        : estrogen_status,
            "progesterone_status"    : progesterone_status,
            "regional_node_examined" : regional_node_examined,
            "regional_node_positive" : regional_node_positive
        }

        try:
            with st.spinner("Analyzing patient data..."):
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json=patient_data
                )

            if response.status_code == 200:
                result = response.json()
                alive_pct = float(result["alive_prob"].replace("%", ""))

                st.divider()

                if result["prediction"] == "Alive":
                    color   = "#00ff88"
                    border  = "#00ff88"
                    status  = "ALIVE"
                    message = "Patient is likely to survive"
                else:
                    color   = "#ff4444"
                    border  = "#ff4444"
                    status  = "DEAD"
                    message = "Patient is likely to not survive"

                st.markdown(f"""
                <div style='background:#1a1a2e; border:1px solid {border}; border-radius:10px; padding:15px; font-family:Arial; box-shadow:0 0 10px {border}; max-width:420px; margin:auto;'>
                    <div style='text-align:center; border-bottom:1px solid {border}; padding-bottom:8px; margin-bottom:8px;'>
                        <p style='color:{color}; margin:0; font-size:10px; letter-spacing:2px;'>MEDICAL REPORT</p>
                        <h2 style='color:{color}; margin:3px 0; font-size:28px;'>{status}</h2>
                        <p style='color:#aaaaaa; margin:0; font-size:12px;'>{message}</p>
                    </div>
                    <div style='display:flex; justify-content:space-between; text-align:center; margin-bottom:8px;'>
                        <div style='flex:1;'>
                            <p style='color:#aaaaaa; margin:0; font-size:9px;'>CONFIDENCE</p>
                            <p style='color:{color}; margin:0; font-size:16px; font-weight:bold;'>{result['confidence']}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.caption(f"Survival Probability: {result['alive_prob']}")
                st.progress(int(alive_pct) / 100)

                if result.get("warning"):
                    st.warning(result["warning"])

            else:
                st.error(f"API Error: {response.status_code}")

        except Exception as e:
            st.error("Connection Error: Make sure FastAPI is running!")