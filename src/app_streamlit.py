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

# ================== CSS (GLOBAL + WELCOME UI) ==================
_style = """
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #020617 0%, #020617 100%);
    color: #e5e7eb;
    font-family: 'Inter', system-ui, sans-serif;
}

#MainMenu, header, footer { visibility: hidden; }

h1, h2, h3 {
    color: #f8fafc;
    letter-spacing: -0.03em;
}

/* HERO */
.hero {
    padding: 4.5rem 1rem 3rem 1rem;
    text-align: center;
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.05;
}
.hero-sub {
    font-size: 1.15rem;
    color: #cbd5f5;
    max-width: 760px;
    margin: 1.4rem auto 2.4rem auto;
}

/* FEATURE CARDS */
.card {
    background: #020617;
    border: 1px solid #1e293b;
    border-radius: 20px;
    padding: 1.8rem;
    text-align: center;
    height: 100%;
}
.card h3 {
    font-size: 1.15rem;
    margin-bottom: 0.6rem;
}
.card p {
    color: #94a3b8;
    font-size: 0.95rem;
}

/* CTA BUTTON */
.cta button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: #3B9797;
    text-align: center;
    border-radius: 18px;
    padding: 0.8rem 2.2rem;
    font-size: 1.05rem;
    font-weight: 600;
    border: none;
}
.cta button:hover {
    box-shadow: 0 12px 30px rgba(37, 99, 235, 0.4);
    transform: translateY(-1px);
}

/* INPUTS */
input, select {
    background-color: #3B4953 !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
}

/* BUTTONS */
.stButton > button {
    background: #3B9797;
    color: white;
    border-radius: 14px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
    border: none;
}

/* GRAPH CARDS */
.element-container:has(canvas) {
    background: #020617;
    border: 1px solid #1e293b;
    border-radius: 16px;
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
    pd.DataFrame([data]).to_csv(
        path, mode="a", header=not os.path.exists(path), index=False
    )


# ================== PDF ==================
def generate_pdf(filename, data):
    c = canvas.Canvas(filename, pagesize=A4)
    y = 800

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Cycling Performance Report")
    y -= 40

    c.setFont("Helvetica", 11)
    for k, v in data.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 18

    c.save()


# ================== TABS ==================
tab_welcome, tab_predictor = st.tabs([" Welcome", " Predictor"])


# ================== WELCOME PAGE ==================
with tab_welcome:

    st.markdown("""
    <div class="hero">
        <div class="hero-title">Cycling Performance<br><span style="color:#3B9797">Prediction</span></div>
        <div class="hero-sub">
            Predict your <b>average speed</b>, <b>power output</b>, and
            <b>calories burned</b> using Machine Learning combined with
            real-world cycling physics.
        </div>
        <div class="hero-sub"> Developed by :<span style="color: #3B9797"> Mohsin HM </span></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="card">
            <h3>🚴 Smart ML Predictions</h3>
            <p>Trained machine learning model to estimate realistic cycling speed.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
            <h3>⚡ Physics-Based Power</h3>
            <p>Power & calories computed using aerodynamics, grade and mass.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card">
            <h3>📄 Professional Reports</h3>
            <p>Download clean PDF reports for training and performance analysis.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns([1, 2, 1])
    with b2:
        st.markdown('<div class="cta">', unsafe_allow_html=True)
        st.info("👉 Go to **Predictor** tab to start")
        st.markdown('</div>', unsafe_allow_html=True)


# ================== PREDICTOR PAGE ==================
with tab_predictor:

    st.markdown("## 👤 Rider Information")
    rider_name = st.text_input("Name", placeholder="e.g. Mohsin", label_visibility="collapsed")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        distance_km = st.number_input("Distance (km)", 1.0, value=30.0)
        ride_time_min = st.number_input("Ride Time (min)", 10.0, value=90.0)
        temperature_c = st.number_input("Temperature (°C)", -5.0, value=28.0)

    with col2:
        elevation_gain_m = st.number_input("Elevation Gain (m)", 0.0, value=200.0)
        route_type = st.selectbox("Route Type", ["flat", "rolling", "climb"])

    rider_weight = st.number_input("Rider Weight (kg)", 30.0, value=70.0)
    bike_weight = st.number_input("Bike Weight (kg)", 5.0, value=8.0)

    st.markdown("<br>", unsafe_allow_html=True)

    run = st.button(" Predict & Save Ride")

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
        power = estimate_power_watts(pred_speed, grade, rider_weight + bike_weight)

        kcal = (power * (ride_time_min / 60) * 0.860421) / 0.24

        save_ride_history({
            "time": datetime.now(),
            "name": rider_name,
            "speed": pred_speed,
            "power": power,
            "calories": kcal
        })

        st.success(f"Prediction for {rider_name}")

        m1, m2, m3 = st.columns(3)
        m1.metric("🚴 Speed", f"{pred_speed:.2f} km/h")
        m2.metric("⚡ Power", f"{power:.0f} W")
        m3.metric("🔥 Calories", f"{kcal:.0f} kcal")

        st.subheader("📊 Performance Graphs")

        c1, c2, c3 = st.columns(3)

        with c1:
            fig, ax = plt.subplots()
            ax.bar(["Speed"], [pred_speed], width=0.3)
            st.pyplot(fig)

        with c2:
            fig, ax = plt.subplots()
            ax.bar(["Power"], [power], width=0.3)
            st.pyplot(fig)

        with c3:
            fig, ax = plt.subplots()
            ax.bar(["Calories"], [kcal], width=0.3)
            st.pyplot(fig)

        pdf_file = f"ride_{rider_name}.pdf"
        generate_pdf(pdf_file, {
            "Name": rider_name,
            "Speed": f"{pred_speed:.2f}",
            "Power": f"{power:.0f}",
            "Calories": f"{kcal:.0f}",
            "Date": datetime.now().strftime("%d-%m-%Y %H:%M")
        })

        with open(pdf_file, "rb") as f:
            st.download_button("📄 Download PDF Report", f, file_name=pdf_file)
