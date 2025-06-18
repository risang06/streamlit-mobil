import streamlit as st
import pickle
import pandas as pd

# Load model pipeline
filename = 'harga_mobil.sav'
with open(filename, 'rb') as f:
    pipeline = pickle.load(f)

st.title("Prediksi Harga Mobil Bekas")

# Form input pengguna
model = st.selectbox("Model Mobil", ['Calya', 'Avanza', 'Xenia', 'Brio'])  # Tambahkan opsi sesuai dataset
tahun = st.number_input("Tahun Mobil", min_value=1990, max_value=2025, value=2021)
transmisi = st.selectbox("Transmisi", ['manual', 'automatic'])
kilometer = st.number_input("Kilometer Ditempuh", min_value=0, value=165188200)
bahan_bakar = st.selectbox("Bahan Bakar", ['bensin', 'diesel', 'hybrid', 'listrik'])  # Sesuaikan opsi
pajak = st.number_input("Pajak (Rupiah)", min_value=0, value=2500012)
mpg = st.number_input("MPG (Miles per Gallon)", min_value=0.0, value=40.3)
cc = st.number_input("Kapasitas Mesin (CC)", min_value=500.0, value=1500.0)

if st.button("Prediksi Harga"):
    input_data = pd.DataFrame([{
        'model': model,
        'tahun': tahun,
        'transmisi': transmisi,
        'kilometer': kilometer,
        'bahan_bakar': bahan_bakar,
        'pajak': pajak,
        'mpg': mpg,
        'cc': cc
    }])
    
    prediction = pipeline.predict(input_data)[0]
    st.success(f"Estimasi harga mobil (dalam Rupiah): Rp{prediction:,.0f}")
