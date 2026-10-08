import streamlit as st
import pandas as pd
import joblib

# 1. MEMUAT "OTAK" MESIN
try:
    model_rab = joblib.load('model_rab_bangunan.pkl')
    kolom_fitur = joblib.load('kolom_fitur.pkl')
except FileNotFoundError:
    st.error("File model belum ditemukan. Pastikan file .pkl ada di folder yang sama dengan app.py!")
    st.stop()

# 2. MENDESAIN TAMPILAN WEB
st.set_page_config(page_title="Prediksi RAB", page_icon="🏗️")
st.title("🏗️ Kalkulator Estimasi RAB Renovasi")
st.write("Masukkan spesifikasi proyek untuk mendapatkan estimasi harga secara instan.")

col1, col2 = st.columns(2)

with col1:
    luas_area = st.number_input("Luas Area (m²)", min_value=10, max_value=1000, value=50)
    jenis_pekerjaan = st.selectbox("Jenis Pekerjaan", ["Renovasi Total", "Interior Dapur", "Eksterior"])

with col2:
    lama_pengerjaan = st.number_input("Target Waktu (Hari)", min_value=7, max_value=365, value=30)
    gaya_interior = st.selectbox("Gaya Interior", ["Minimalis", "Industrial", "Skandinavia", "Klasik"])

st.markdown("---")

# 3. TOMBOL EKSEKUSI
if st.button("Hitung Estimasi Biaya", type="primary"):
    
    input_data = {
        'Luas_Area': luas_area,
        'Lama_Pengerjaan': lama_pengerjaan,
        'Jenis_Pekerjaan': jenis_pekerjaan,
        'Gaya_Interior': gaya_interior
    }
    df_input = pd.DataFrame([input_data])
    
    df_input_angka = pd.get_dummies(df_input, dtype=int)
    
    for col in kolom_fitur:
        if col not in df_input_angka.columns:
            df_input_angka[col] = 0
            
    df_input_angka = df_input_angka[kolom_fitur] 
    
    tebakan = model_rab.predict(df_input_angka)
    
    st.success(f"### 💰 Estimasi Total Biaya: Rp {tebakan[0]:,.0f}")
    st.caption("Catatan: Ini adalah estimasi berbasis Machine Learning dari data historis proyek sebelumnya.")