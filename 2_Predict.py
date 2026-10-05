import pandas as pd, streamlit as st
from utils import get_inputs, grade, status, tips, load_model, load_data, load_metrics

st.set_page_config(page_title="Predict", page_icon="🎯", layout="wide")
st.title("🎯 Predict a Student's Marks")
model, features = load_model()

name = st.text_input("Student name (optional)")
st.write("Adjust the details below:")
vals = get_inputs(st)

if st.button("Predict marks", type="primary"):
    pred = float(max(0, min(100, model.predict(pd.DataFrame([vals])[features])[0])))
    mae = load_metrics()["results"][load_metrics()["best"]]["MAE"]

    st.subheader(f"Result{' for ' + name if name else ''}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted marks", f"{pred:.1f} / 100")
    c2.metric("Expected grade", grade(pred))
    c3.metric("Result", status(pred))
    st.progress(int(pred))
    st.caption(f"Likely range: {max(0, pred - mae):.1f} – {min(100, pred + mae):.1f} (average model error ±{mae} marks)")

    if pred >= 75: st.success("Excellent performance expected! 🎉")
    elif pred >= 50: st.warning("Average performance. There is room to improve.")
    else: st.error("At risk. Needs extra support.")

    left, right = st.columns(2)
    with left:
        st.subheader("💡 Tips to improve")
        for t in tips(vals): st.write("•", t)
    with right:
        st.subheader("📏 Compared with class average")
        avg = load_data()[features].mean().round(1)
        st.dataframe(pd.DataFrame({"Student": pd.Series(vals)[features], "Class average": avg}),
                     width="stretch")
