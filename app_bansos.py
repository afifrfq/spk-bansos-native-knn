import streamlit as st
import pandas as pd
import numpy as np
from logika_knn import prediksi_native, dapatkan_tetangga

# --- KONFIGURASI ---
st.set_page_config(page_title="SPK Bansos - Native", layout="centered")

# ============================================================
# 1. KONSTANTA & CONFIG
# ============================================================
MAX_PENGHASILAN = 2000000  # Pembagi Normalisasi Gaji
MAX_TANGGUNGAN = 10        # Pembagi Normalisasi Tanggungan
MAX_LISTRIK = 2200         # Pembagi Normalisasi Listrik
MAX_RUMAH = 4              # Pembagi Normalisasi Rumah (Skala 1-4)
MAX_ASET = 100             # Pembagi Normalisasi Aset

# ============================================================
# 2. LOAD DATA BERSIH
# ============================================================
@st.cache_data
def load_clean_data():
    try:
        # Membaca file hasil olahan Notebook
        df = pd.read_csv('data_bansos_bersih.csv')
        # Konversi ke List of Lists (Format Native)
        database = df.values.tolist()
        return df, database
    except FileNotFoundError:
        return None, None

df_clean, database_warga = load_clean_data()

if df_clean is None:
    st.error("❌ File 'data_bansos_bersih.csv' tidak ditemukan! Jalankan Notebook dulu.")
    st.stop()

# ============================================================
# 3. INTERFACE APLIKASI
# ============================================================
st.title("🏛️ Aplikasi Penentu Bansos")
st.success("Sistem Pendukung Keputusan (Metode KNN Native)")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Ekonomi")
    penghasilan = st.number_input("1. Penghasilan (Rp)", 0, 10000000, 500000, step=50000)
    tanggungan = st.number_input("2. Tanggungan (Org)", 0, 20, 3)
    aset = st.slider("3. Skor Aset (0-100)", 0, 100, 50)

with col2:
    st.subheader("Fasilitas Rumah")
    listrik = st.selectbox("4. Daya Listrik", [450, 900, 1300, 2200])
    
    # --- BAGIAN YANG DIUBAH (BAHASA LEBIH HALUS) ---
    # Skala 1-4 Tetap (Agar Akurasi Bagus), tapi Labelnya Sopan
    rumah_opsi = st.selectbox("5. Kondisi Rumah", 
                              ["Tidak Layak Huni (1)", 
                               "Kurang Layak (2)", 
                               "Cukup Layak (3)", 
                               "Sangat Layak (4)"])
    
    # Ambil angka dalam kurung untuk perhitungan
    # Contoh: "Tidak Layak Huni (1)" -> diambil angka 1
    rumah_angka = int(rumah_opsi.split("(")[1].replace(")", ""))

# ============================================================
# 4. PROSES PREDIKSI
# ============================================================
if st.button("🔍 PROSES PREDIKSI"):
    
    # --- RUMUS NORMALISASI MANUAL ---
    # Agar input user (Rupiah/Teks) setara dengan database (0-1)
    
    norm_gaji = penghasilan / MAX_PENGHASILAN
    norm_tanggung = tanggungan / MAX_TANGGUNGAN
    norm_listrik = listrik / MAX_LISTRIK
    norm_rumah = rumah_angka / MAX_RUMAH
    norm_aset = aset / MAX_ASET
    
    # Fungsi limit agar nilai tidak lebih dari 1.0 atau kurang dari 0.0
    def limit(n): return min(max(n, 0.0), 1.0)
    
    data_uji = [
        limit(norm_gaji), 
        limit(norm_tanggung), 
        limit(norm_listrik), 
        limit(norm_rumah), 
        limit(norm_aset)
    ]
    
    # --- HITUNG NATIVE ---
    k = 5
    hasil = prediksi_native(database_warga, data_uji, k)
    
    # --- HASIL ---
    st.markdown("---")
    st.subheader("Hasil Keputusan:")
    
    if hasil == 1:
        st.success("✅ Warga Tersebut **LAYAK** Menerima Bantuan")
        st.info("Keterangan: Profil ekonomi dan kondisi rumah memenuhi kriteria bantuan.")
    else:
        st.error("❌ Warga Tersebut **TIDAK LAYAK**")
        st.warning("Keterangan: Profil ekonomi dinilai mampu/sejahtera.")
        
    # --- TRANSPARANSI ---
    with st.expander("Lihat Detail Perhitungan (Native Code)"):
        st.write("Vector Input (Normalized):", [round(x,2) for x in data_uji])
        st.write(f"{k} Data Tetangga Terdekat (Most Similar):")
        
        tetangga = dapatkan_tetangga(database_warga, data_uji, k)
        # Tampilkan tabel tetangga dengan nama kolom yang jelas
        df_tetangga = pd.DataFrame(tetangga, columns=df_clean.columns)
        st.dataframe(df_tetangga)