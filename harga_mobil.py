import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# ========== Tampilan Atas ==========
st.set_page_config(page_title="Prediksi Harga Mobil Bekas", layout="centered")
st.markdown("<h1 style='text-align: center; color: navy;'> Prediksi Harga Mobil Bekas</h1>", unsafe_allow_html=True)
st.write("Isi form di bawah ini untuk memprediksi harga mobil bekas berdasarkan data historis.")

# ========== Load & Latih Model ==========
@st.cache_data
def load_data():
    return pd.read_csv("datasetmobil.csv")

df = load_data()

X = df.drop("harga", axis=1)
y = df["harga"]

categorical_cols = ["model", "transmisi", "bahan_bakar"]
numerical_cols = ["tahun", "kilometer", "pajak", "mpg", "cc"]

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough"
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# ========== Evaluasi Model ==========
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2 = r2_score(y_test, y_pred)

# ========== Form Input ==========
with st.form("form_prediksi"):
    col1, col2 = st.columns(2)

    with col1:
        model = st.selectbox("Model Mobil", sorted(df["model"].unique()), index=0)
        tahun = st.number_input("Tahun Pembuatan", min_value=2000, max_value=2025, value=2021)
        transmisi = st.selectbox("Transmisi", df["transmisi"].unique())

    with col2:
        kilometer = st.number_input("Jarak Tempuh (km)", value=165188200)
        bahan_bakar = st.selectbox("Bahan Bakar", df["bahan_bakar"].unique())
        pajak = st.number_input("Pajak (Rp)", value=2500012)
    
    mpg = st.number_input("Konsumsi BBM (mpg)", value=40.3)
    cc = st.number_input("Kapasitas Mesin (cc)", value=1.5)

    submit = st.form_submit_button("🔍 Prediksi Harga Mobil")

# ========== Hasil Prediksi ==========
if submit:
    input_data = pd.DataFrame([{
        "model": model,
        "tahun": tahun,
        "transmisi": transmisi,
        "kilometer": kilometer,
        "bahan_bakar": bahan_bakar,
        "pajak": pajak,
        "mpg": mpg,
        "cc": cc
    }])

    prediksi = pipeline.predict(input_data)[0]
    st.markdown(f"<h2 style='color: green;'>Prediksi Harga Mobil: Rp{prediksi:,.0f}</h2>", unsafe_allow_html=True)
