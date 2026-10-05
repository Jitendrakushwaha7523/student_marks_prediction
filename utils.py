"""Shared helpers used by every page."""
import json, os, joblib, pandas as pd, streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))   # works from any folder
def p(name): return os.path.join(BASE, name)

# name: (label, min, max, default, step)  -> slider settings
FEATURES = {
    "study_hours":      ("Study hours per day", 0.0, 10.0, 4.0, 0.5),
    "attendance":       ("Attendance (%)", 50, 100, 80, 1),
    "previous_marks":   ("Previous exam marks", 0, 100, 65, 1),
    "assignment_score": ("Assignment score", 0, 100, 70, 1),
    "sleep_hours":      ("Sleep hours per day", 4.0, 10.0, 7.0, 0.5),
}

@st.cache_resource
def load_model():
    if not os.path.exists(p("model.pkl")):
        st.error("Model not found. Run `python generate_data.py` and `python train_model.py` first.")
        st.stop()
    b = joblib.load(p("model.pkl"))
    return b["model"], b["features"]

@st.cache_data
def load_data():
    return pd.read_csv(p("students.csv"))

def load_metrics():
    return json.load(open(p("metrics.json")))

def grade(m):
    return "A+" if m >= 90 else "A" if m >= 80 else "B" if m >= 70 else "C" if m >= 60 else "D" if m >= 50 else "F"

def status(m):
    return "Pass" if m >= 40 else "Fail"

def get_inputs(box):
    """Draw all input widgets inside `box` (st, st.sidebar or a column) and return values."""
    vals = {}
    for key, (label, lo, hi, default, step) in FEATURES.items():
        vals[key] = box.slider(label, lo, hi, default, step)
    vals["extra_classes"] = int(box.checkbox("Attends extra classes / tuition"))
    return vals

def tips(v):
    t = []
    if v["study_hours"] < 3:       t.append("Study at least 3–4 hours daily.")
    if v["attendance"] < 75:       t.append("Raise attendance above 75%.")
    if v["sleep_hours"] < 6:       t.append("Sleep 7–8 hours for better focus and memory.")
    if v["assignment_score"] < 60: t.append("Put more effort into assignments.")
    if v["previous_marks"] < 50:   t.append("Revise basics from the previous exam topics.")
    return t or ["Great habits! Keep it up."]
