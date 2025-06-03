import streamlit as st

st.title("📚 Latar Belakang")

st.markdown("""
Di era digital saat ini, **industri kuliner dan restoran berkembang pesat**, khususnya di kota-kota besar seperti **Semarang**.  
Dengan banyaknya pilihan restoran yang tersedia, **pemilik usaha** maupun **pengguna** membutuhkan cara untuk memahami karakteristik dan kualitas restoran secara lebih **objektif dan berbasis data**.

---

### 🎯 Tujuan Analisis
Dashboard ini dibuat untuk:
- Melakukan **klasterisasi restoran** di Semarang berdasarkan beberapa fitur penting.
- Menyediakan **visualisasi interaktif** agar pengguna bisa memahami pola-pola tersembunyi dalam data.
- Memberikan **formulir prediksi**, sehingga pengguna dapat mengetahui klaster restoran baru berdasarkan input fitur tertentu.

---

### 🧾 Data yang Digunakan
Dataset berisi informasi mengenai restoran di Semarang, dengan fitur seperti:
- **Rating restoran**
- **Jam operasional**
- **Fasilitas (wifi, toilet)**
- **Metode pembayaran (tunai)**

---

### 🧠 Metode Analisis
Pendekatan yang digunakan dalam dashboard ini adalah:
- **EDA (Exploratory Data Analysis)**: Untuk memahami distribusi dan pola data
- **KMeans Clustering**: Untuk mengelompokkan restoran berdasarkan kemiripan fitur numerik
- **PCA (Principal Component Analysis)**: Untuk menyederhanakan data menjadi 2 dimensi guna keperluan visualisasi klaster

### Dibuat dengan cinta oleh kelompok 5
""")
