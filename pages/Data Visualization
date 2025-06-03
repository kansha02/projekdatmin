
import streamlit as st 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📈 Halaman 1: Eksplorasi Data Restoran")

# Load dataset dari GitHub 
url = "https://raw.githubusercontent.com/fitripl02/streamlit/refs/heads/main/semarang_resto_dataset.csv"
df = pd.read_csv(url)

# Tampilkan dataset
st.subheader("🔍 Data Restoran")
st.dataframe(df.head())

# Tampilkan jumlah restoran
st.subheader("🔢 Jumlah Restoran dalam Dataset")
st.write(f"Total restoran: **{df.shape[0]}**")

# Statistik deskriptif
st.subheader("📊 Statistik Deskriptif")
st.write(df.describe())

# Visualisasi distribusi rating
st.subheader("⭐ Distribusi Rating Restoran")
fig, ax = plt.subplots()
sns.histplot(df["resto_rating"], bins=10, kde=True, ax=ax)
st.pyplot(fig)

# Bar chart jenis restoran
if "resto_type" in df.columns:
    st.subheader("🍽️ Jenis Restoran Paling Umum")
    st.bar_chart(df["resto_type"].value_counts().head(10))
