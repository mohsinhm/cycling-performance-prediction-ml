# Cycling Performance Prediction (ML Regression)

## 📌 Overview
This project predicts a cyclist's **average speed (km/h)** for a ride based on:
- Distance (km)
- Elevation gain (m)
- Ride duration (minutes)
- Temperature (°C)
- Route type (flat / rolling / climb)

It is designed as a **portfolio project** for data science and machine learning, and can be extended with your own ride data (e.g., from Strava or cycling apps).

---

## 🧠 Problem Statement
Given past ride data, we want to **predict the average speed** of a new ride and understand:

- Which features most affect cycling performance.
- How terrain and distance influence expected speed.

---

## 🗂 Dataset
The sample dataset is in `data/raw/rides_raw_sample.csv` with columns:

- `distance_km`
- `elevation_gain_m`
- `ride_time_min`
- `temperature_c`
- `route_type` (flat/rolling/climb)
- `avg_speed_kmph` (target)

Replace or extend this file with your own ride data.

---

## 🛠 Tech Stack
- Python 3.x
- pandas, numpy
- scikit-learn
- joblib
- matplotlib (optional, for your own EDA)
- Streamlit (for the web app)

---

## 🏗 Project Structure
```bash
cycling-performance-prediction-ml/
├─ data/
│  ├─ raw/
│  │  └─ rides_raw_sample.csv
│  └─ processed/
├─ notebooks/
│  ├─ 01_exploratory_data_analysis.ipynb
│  └─ 02_model_training.ipynb
├─ src/
│  ├─ config.py
│  ├─ data_preprocessing.py
│  ├─ features.py
│  ├─ train_model.py
│  └─ app_streamlit.py
├─ models/
├─ reports/
│  ├─ figures/
│  └─ summary.md
├─ requirements.txt
└─ README.md
```

---

## ▶️ How to Run

1️⃣ **Create a virtual environment (optional but recommended)**

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

2️⃣ **Install dependencies**

```bash
pip install -r requirements.txt
```

3️⃣ **Check the sample data**

Confirm this file exists and inspect it:

```bash
data/raw/rides_raw_sample.csv
```

4️⃣ **Run preprocessing + training**

```bash
python -m src.data_preprocessing
python -m src.train_model
```

5️⃣ **Run the Streamlit app**

```bash
streamlit run src/app_streamlit.py
```

---

## 🚀 Future Ideas
- Add heart rate, power, wind speed.
- Integrate weather API.
- Build comparison of different bikes, routes, or training blocks.

---

## 👤 Author
**Mohsin HM** – B.Sc. Data Science | Road Cyclist | ML Enthusiast
