import numpy as np, pandas as pd, streamlit as st
from utils import FEATURES, get_inputs, load_model

st.set_page_config(page_title="What If", page_icon="🔮", layout="wide")
st.title("🔮 What-If Analysis")
st.write("Set a student's current habits on the left, then see how changing **one** habit affects the marks.")
model, features = load_model()

st.sidebar.header("Current student")
base = get_inputs(st.sidebar)

key = st.selectbox("Which factor do you want to change?", list(FEATURES), format_func=lambda k: FEATURES[k][0])
_, lo, hi, _, step = FEATURES[key]
grid = np.arange(lo, hi + step, step)
rows = pd.DataFrame([{**base, key: v} for v in grid])[features]
curve = pd.Series(model.predict(rows).clip(0, 100), index=grid, name="Predicted marks")

st.line_chart(curve)
now = float(model.predict(pd.DataFrame([base])[features])[0])
best = curve.max()
c1, c2 = st.columns(2)
c1.metric("Marks with current habits", f"{now:.1f}")
c2.metric(f"Best possible by changing {FEATURES[key][0]}", f"{best:.1f}", f"{best - now:+.1f}")
