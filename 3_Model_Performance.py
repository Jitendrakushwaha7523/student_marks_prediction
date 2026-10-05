import numpy as np, pandas as pd, streamlit as st
from sklearn.inspection import permutation_importance
from utils import load_model, load_data, load_metrics, p

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
st.title("📈 Model Performance")
model, features = load_model()
info = load_metrics()

st.subheader("Model comparison")
res = pd.DataFrame(info["results"]).T
c1, c2 = st.columns([1, 1])
c1.dataframe(res, width="stretch")
c1.success(f"Best model: **{info['best']}**")
c2.write("**R² score** (higher is better)")
c2.bar_chart(res["R2"])
st.caption("MAE = average error in marks (lower is better). R² = how much of the variation the model explains (max 1.0).")

st.divider()
left, right = st.columns(2)
preds = pd.read_csv(p("test_predictions.csv"))
with left:
    st.subheader("Actual vs predicted (test data)")
    st.scatter_chart(preds, x="actual", y="predicted")
    st.caption("Points close to a straight diagonal line mean accurate predictions.")
with right:
    st.subheader("Error distribution")
    err = preds["predicted"] - preds["actual"]
    counts, edges = np.histogram(err, bins=15)
    st.bar_chart(pd.DataFrame({"students": counts}, index=edges[:-1].round(1)))
    st.caption("Most errors should be close to 0.")

@st.cache_data
def importance():
    df = load_data()
    r = permutation_importance(model, df[features], df["final_marks"], n_repeats=5, random_state=42)
    return pd.Series(r.importances_mean, index=features).sort_values()

st.divider()
st.subheader("🔑 Which factors matter most?")
st.bar_chart(importance())
