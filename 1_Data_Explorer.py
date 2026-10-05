import numpy as np, pandas as pd, streamlit as st
from utils import load_data

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")
st.title("📊 Data Explorer")
df = load_data()

t1, t2, t3 = st.tabs(["Dataset", "Distributions", "Correlations"])

with t1:
    st.write(f"**{len(df)} students**, {df.shape[1]} columns")
    st.dataframe(df, width="stretch", height=300)
    st.subheader("Summary statistics")
    st.dataframe(df.describe().round(2), width="stretch")

with t2:
    col = st.selectbox("Choose a column", df.columns, index=len(df.columns) - 1)
    counts, edges = np.histogram(df[col], bins=15)
    st.write(f"**Distribution of {col}**")
    st.bar_chart(pd.DataFrame({"students": counts}, index=edges[:-1].round(1)))
    if col != "final_marks":
        st.write(f"**{col} vs final marks**")
        st.scatter_chart(df, x=col, y="final_marks")

with t3:
    corr = df.corr()
    st.write("**Correlation matrix** (green = strong positive link)")
    st.dataframe(corr.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1).format("{:.2f}"),
                 width="stretch")
    st.write("**What is linked to final marks?**")
    st.bar_chart(corr["final_marks"].drop("final_marks").sort_values())
