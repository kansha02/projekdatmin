import streamlit as st
import pages.eda as eda
import pages.model_page as model_page
import pages.predict as predict

st.set_page_config(page_title="Dashboard Data Mining", layout="wide")
st.sidebar.title("Navigasi")

page = st.sidebar.selectbox("Pilih halaman", ("EDA", "Model", "Prediksi"))

if page == "EDA":
    eda.show()
elif page == "Model":
    model_page.show()
elif page == "Prediksi":
    predict.show()
