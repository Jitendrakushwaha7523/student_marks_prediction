import json, os, joblib, pandas as pd, streamlit as st

st.set_page_config(page_title="Student Marks Predictor", page_icon="🎓")

if not os.path.exists("model.pkl"):
    st.error("Model not found. Run `python generate_data.py` then `python train_model.py` first.")
    st.stop()

bundle = joblib.load("model.pkl")
model, features = bundle["model"], bundle["features"]

st.title("🎓 Student Marks Predictor")
st.write("Enter a student's details in the sidebar to predict their final marks.")

st.sidebar.header("Student details")
inputs = {
    "study_hours":      st.sidebar.slider("Study hours per day", 0.0, 10.0, 4.0, 0.5),
    "attendance":       st.sidebar.slider("Attendance (%)", 50, 100, 80),
    "previous_marks":   st.sidebar.slider("Previous exam marks", 0, 100, 65),
    "assignment_score": st.sidebar.slider("Assignment score", 0, 100, 70),
    "sleep_hours":      st.sidebar.slider("Sleep hours per day", 4.0, 10.0, 7.0, 0.5),
    "extra_classes":    int(st.sidebar.checkbox("Attends extra classes / tuition")),
}
row = pd.DataFrame([inputs])[features]

def grade(m):
    return "A+" if m >= 90 else "A" if m >= 80 else "B" if m >= 70 else "C" if m >= 60 else "D" if m >= 50 else "F"

if st.button("Predict marks", type="primary"):
    pred = float(max(0, min(100, model.predict(row)[0])))
    c1, c2 = st.columns(2)
    c1.metric("Predicted marks", f"{pred:.1f} / 100")
    c2.metric("Expected grade", grade(pred))
    st.progress(int(pred))

    st.subheader("💡 Tips to improve")
    tips = []
    if inputs["study_hours"] < 3: tips.append("Study at least 3–4 hours daily.")
    if inputs["attendance"] < 75: tips.append("Raise attendance above 75%.")
    if inputs["sleep_hours"] < 6: tips.append("Get 7–8 hours of sleep for better focus.")
    if inputs["assignment_score"] < 60: tips.append("Put more effort into assignments.")
    if not tips: tips.append("Great habits! Keep it up.")
    for t in tips: st.write("•", t)

    if hasattr(model, "feature_importances_"):
        st.subheader("What matters most")
        imp = pd.Series(model.feature_importances_, index=features).sort_values()
        st.bar_chart(imp)

if os.path.exists("metrics.json"):
    with st.expander("Model performance"):
        info = json.load(open("metrics.json"))
        st.write(f"Best model: **{info['best']}**")
        st.table(pd.DataFrame(info["results"]).T)
