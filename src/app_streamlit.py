# ================== IMPORTS ==================
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
from config import MODEL_PATH
import sys
import os
from datetime import datetime

# Matplotlib
import matplotlib.pyplot as plt

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Cycling Predict",
    page_icon="🚴",
    initial_sidebar_state="collapsed",
    layout="wide"
)

# ================== CSS (UI / UX UPGRADE) ==================
_style = """
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #020617 0%, #020617 100%);
    color: #e5e7eb;
    font-family: 'Inter', system-ui, sans-serif;
}

#MainMenu, header, footer {
    visibility: hidden;
}

h1, h2, h3 {
    letter-spacing: -0.02em;
    color: #f8fafc;
}

input, select {
    background-color: #020617 !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
}

.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border-radius: 14px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
    border: none;
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 30px rgba(37, 99, 235, 0.35);
}

hr {
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, #334155, transparent);
    margin: 1.5rem 0;
}

.stAlert {
    border-radius: 14px;
}

.element-container:has(canvas) {
    background: #020617;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 0.6rem;
}
</style>
"""
st.markdown(_style, unsafe_allow_html=True)


# ================== PATH SETUP ==================
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import add_features


# ================== MODEL LOADER ==================
@st.cache_resource
def load_model():
    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        st.error(f"Model not found at {MODEL_PATH}")
        return None
    return joblib.load(MODEL_PATH)


# ================== PHYSICS ==================
def estimate_power_watts(speed_kmph, grade, mass_kg, CdA=0.5, Cr=0.004, rho=1.226):
    g = 9.80665
    v_ms = speed_kmph / 3.6
    F_roll = Cr * mass_kg * g
    F_climb = mass_kg * g * grade
    F_aero = 0.5 * rho * CdA * v_ms * v_ms
    return max((F_roll + F_climb + F_aero) * v_ms, 0)


# ================== SAVE HISTORY ==================
def save_ride_history(data, path="data/ride_history.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([data])
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


# ================== PDF ==================
def generate_pdf(filename, data, graphs):
    c = canvas.Canvas(filename, pagesize=A4)
    y = 800

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Cycling Performance Report")
    y -= 40

    c.setFont("Helvetica", 11)
    for k, v in data.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 18

    for title, path in graphs.items():
        if os.path.exists(path):
            y -= 30
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, title)
            y -= 220
            c.drawImage(ImageReader(path), 50, y, width=500, height=200)

    c.save()


# ================== TABS ==================
tab_welcome, tab_predictor = st.tabs(["🏠 Welcome", "🚴 Predictor"])


# ================== WELCOME ==================
with tab_welcome:
    st.title("🚴‍♂️ Cycling Performance Prediction")

    st.markdown("""
    Predict your **average cycling speed**, **power**, and **calories burned**
    using **Machine Learning + Physics calculations**.

    ### 🚴 Input Data
    - Rider Name  
    - Distance & Elevation  
    - Ride Time & Temperature  
    - Route Type  
    - Rider & Bike Weight  

    ### 📊 Output
    - Speed, Power, Calories  
    - Side-by-side graphs  
    - Downloadable PDF report  

    **Developer:** Mohsin HM  
    """)


# ================== PREDICTOR ==================
with tab_predictor:

    st.markdown("## 👤 Rider Information")
    rider_name = st.text_input("Name", placeholder="e.g. Mohsin", label_visibility="collapsed")

    st.divider()

    st.markdown("## 📥 Ride Details")
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        distance_km = st.number_input("Distance (km)", 1.0, value=30.0)
        ride_time_min = st.number_input("Ride Time (min)", 10.0, value=90.0)
        temperature_c = st.number_input("Temperature (°C)", -5.0, value=28.0)

    with col2:
        elevation_gain_m = st.number_input("Elevation Gain (m)", 0.0, value=200.0)
        route_type = st.selectbox("Route Type", ["flat", "rolling", "climb"])

    st.markdown("## ⚙ Rider & Bike")
    rider_weight = st.number_input("Rider Weight (kg)", 30.0, value=70.0)
    bike_weight = st.number_input("Bike Weight (kg)", 5.0, value=8.0)

    st.markdown("<br>", unsafe_allow_html=True)

    btn1, btn2, btn3 = st.columns([1, 2, 1])
    with btn2:
        run = st.button("🚴 Predict & Save Ride")

    if run:

        if not rider_name.strip():
            st.error("Please enter rider name")
            st.stop()

        model = load_model()
        if model is None:
            st.stop()

        df = pd.DataFrame([{
            "distance_km": distance_km,
            "elevation_gain_m": elevation_gain_m,
            "ride_time_min": ride_time_min,
            "temperature_c": temperature_c,
            "route_type": route_type,
        }])

        df = add_features(df)
        pred_speed = float(model.predict(df)[0])

        grade = elevation_gain_m / max(distance_km * 1000, 1)
        total_mass = rider_weight + bike_weight
        power = estimate_power_watts(pred_speed, grade, total_mass)

        hours = ride_time_min / 60
        kcal = (power * hours * 0.860421) / 0.24

        save_ride_history({
            "time": datetime.now(),
            "name": rider_name,
            "distance_km": distance_km,
            "elevation_gain_m": elevation_gain_m,
            "ride_time_min": ride_time_min,
            "temperature_c": temperature_c,
            "route_type": route_type,
            "speed_kmph": round(pred_speed, 2),
            "power_w": round(power, 1),
            "calories": round(kcal, 1)
        })

        st.success(f"Prediction for {rider_name}")

        m1, m2, m3 = st.columns(3)
        m1.metric("🚴 Speed", f"{pred_speed:.2f} km/h")
        m2.metric("⚡ Power", f"{power:.0f} W")
        m3.metric("🔥 Calories", f"{kcal:.0f} kcal")

        st.markdown("## 📊 Performance Graphs")

        c1, c2, c3 = st.columns(3)

        with c1:
            fig1, ax1 = plt.subplots(figsize=(4, 3))
            ax1.bar(["Speed"], [pred_speed], width=0.3)
            ax1.set_ylabel("km/h")
            ax1.set_title("Speed")
            st.pyplot(fig1)

        with c2:
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            ax2.bar(["Power"], [power], width=0.3)
            ax2.set_ylabel("W")
            ax2.set_title("Power")
            st.pyplot(fig2)

        with c3:
            fig3, ax3 = plt.subplots(figsize=(4, 3))
            ax3.bar(["Calories"], [kcal], width=0.3)
            ax3.set_ylabel("kcal")
            ax3.set_title("Calories")
            st.pyplot(fig3)

        pdf_data = {
            "Name": rider_name,
            "Speed (km/h)": f"{pred_speed:.2f}",
            "Power (W)": f"{power:.0f}",
            "Calories (kcal)": f"{kcal:.0f}",
            "Distance (km)": distance_km,
            "Elevation Gain (m)": elevation_gain_m,
            "Date": datetime.now().strftime("%d-%m-%Y %H:%M")
        }

        pdf_file = f"ride_{rider_name}.pdf"
        generate_pdf(pdf_file, pdf_data, {})

        with open(pdf_file, "rb") as f:
            st.download_button(
                "📄 Download PDF Report",
                f,
                file_name=pdf_file,
                mime="application/pdf"
            )
