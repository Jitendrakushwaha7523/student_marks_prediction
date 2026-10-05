import streamlit as st
from utils import load_data, load_metrics, load_model

st.set_page_config(page_title="Student Marks Predictor", page_icon="🎓", layout="wide")
load_model()
df, info = load_data(), load_metrics()

st.title("🎓 Student Marks Prediction System")
st.caption("A machine learning project that predicts final exam marks from study habits and academic history.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Students in dataset", len(df))
c2.metric("Input features", df.shape[1] - 1)
c3.metric("Best model", info["best"])
c4.metric("R² score", info["results"][info["best"]]["R2"])

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("📌 About the project")
    st.write(
        "Teachers often find out a student is struggling only after the exam. "
        "This system predicts marks **in advance**, so weak students can be helped early."
    )
    st.subheader("⚙️ How it works")
    st.markdown(
        "1. Student data is collected (study hours, attendance, etc.)\n"
        "2. Three regression models are trained and compared\n"
        "3. The best model is saved and used for predictions\n"
        "4. This web app lets anyone use it without coding"
    )
with right:
    st.subheader("🧭 Pages (see the left sidebar)")
    st.markdown(
        "- **Data Explorer** – view the dataset, charts and correlations\n"
        "- **Predict** – predict marks for one student with tips\n"
        "- **Model Performance** – compare models, accuracy and key factors\n"
        "- **Batch Prediction** – upload a CSV and predict a whole class\n"
        "- **What If** – see how changing one habit changes the marks"
    )
    st.info("Start with **Predict** in the sidebar 👈")
