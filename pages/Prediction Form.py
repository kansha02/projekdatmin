import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.title("🔮 Halaman 3: Prediksi Klaster Restoran Baru")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('semarang_resto_dataset.csv')

df = load_data()

# Pilih fitur untuk clustering
features = ['resto_rating', 'average_operation_hours', 'wifi_facility', 'toilet_facility', 'cash_payment_only']
X = df[features]

# Scaling data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering dengan KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
df['cluster'] = kmeans.predict(X_scaled)

# Input user
st.sidebar.header("Masukkan Kriteria Restoran")
resto_rating = st.sidebar.slider("Rating Restoran", 1.0, 5.0, 4.0)
operation_hours = st.sidebar.slider("Jam Operasi", 1.0, 24.0, 10.0)
wifi = st.sidebar.selectbox("Wifi Tersedia", [0, 1])
toilet = st.sidebar.selectbox("Ada Toilet", [0, 1])
cash_only = st.sidebar.selectbox("Hanya Tunai", [0, 1])

# Prediksi Klaster
if st.sidebar.button("Prediksi Klaster"):
    input_data = np.array([[resto_rating, operation_hours, wifi, toilet, cash_only]])
    user_scaled = scaler.transform(input_data)
    pred = kmeans.predict(user_scaled)
    
    st.success(f"Restoran ini diprediksi masuk ke Klaster: **{int(pred[0])}**")
    
    # Tampilkan contoh restoran dari klaster yang diprediksi
    st.subheader(f"Contoh Restoran di Klaster {int(pred[0])}:")
    cluster_restos = df[df['cluster'] == pred[0]][['resto_name', 'resto_type', 'resto_rating']].head(5)
    st.table(cluster_restos)
