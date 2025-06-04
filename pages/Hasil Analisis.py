import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Halaman 3: Simulasi Analisis", layout="wide")
st.title("📈 Halaman 3: Simulasi Evaluasi Model Rating Tinggi")

st.info("🧪 Ini adalah simulasi hasil analisis prediksi apakah sebuah restoran mendapat rating tinggi (≥ 4.5)")

# ===== Simulasi Prediksi =====
np.random.seed(42)  # agar hasil konsisten
y_true = np.random.choice([0, 1], size=300, p=[0.75, 0.25])  # 0 = <4.5, 1 = ≥4.5
noise = np.random.binomial(1, 0.1, size=300)  # 10% error
y_pred = np.abs(y_true - noise)

# ===== Classification Report =====
report = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

st.subheader("📊 Simulasi Classification Report")
st.dataframe(report_df.style.format(precision=2))

# ===== Confusion Matrix =====
st.subheader("🧮 Confusion Matrix")
cm = confusion_matrix(y_true, y_pred)
labels = ["Rating < 4.5", "Rating ≥ 4.5"]
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_xlabel("Prediksi")
ax.set_ylabel("Aktual")
st.pyplot(fig)

# ===== Simulasi Feature Importance =====
st.subheader("📌 Simulasi Fitur yang Mempengaruhi Rating Tinggi")
feature_importance = pd.Series({
    "Jam Operasional": 0.30,
    "Tersedia Wifi": 0.25,
    "Ada Toilet": 0.18,
    "Hanya Tunai": 0.12,
    "Jenis Restoran": 0.15
}).sort_values()

fig2, ax2 = plt.subplots()
feature_importance.plot(kind="barh", ax=ax2, color='darkorange')
ax2.set_xlabel("Tingkat Pengaruh")
ax2.set_title("Simulasi Feature Importance")
st.pyplot(fig2)
