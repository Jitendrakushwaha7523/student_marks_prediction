import pandas as pd, streamlit as st
from utils import load_model, load_data, grade, status

st.set_page_config(page_title="Batch Prediction", page_icon="📁", layout="wide")
st.title("📁 Batch Prediction (whole class)")
model, features = load_model()

st.write(f"Upload a CSV file with these columns: `{', '.join(features)}`")
sample = load_data()[features].head(10)
st.download_button("⬇️ Download sample template", sample.to_csv(index=False), "template.csv", "text/csv")

file = st.file_uploader("Upload CSV", type="csv")
use_sample = st.checkbox("No file? Try with sample students")
data = pd.read_csv(file) if file else (load_data().head(30).drop(columns="final_marks") if use_sample else None)

if data is not None:
    missing = [c for c in features if c not in data.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
    else:
        out = data.copy()
        out["predicted_marks"] = model.predict(out[features]).clip(0, 100).round(1)
        out["grade"] = out["predicted_marks"].apply(grade)
        out["result"] = out["predicted_marks"].apply(status)

        c1, c2, c3 = st.columns(3)
        c1.metric("Students", len(out))
        c2.metric("Average predicted marks", round(out["predicted_marks"].mean(), 1))
        c3.metric("At risk (below 50)", int((out["predicted_marks"] < 50).sum()))

        st.dataframe(out, width="stretch")
        st.write("**Grade distribution**")
        st.bar_chart(out["grade"].value_counts().reindex(["A+", "A", "B", "C", "D", "F"]).fillna(0))
        st.download_button("⬇️ Download results", out.to_csv(index=False), "predictions.csv", "text/csv")
