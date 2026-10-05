"""Trains 3 models, compares them, and saves the best one."""
import json, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("students.csv").dropna()
X, y = df.drop(columns="final_marks"), df["final_marks"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

results, best_name, best_r2, best_model = {}, None, -1, None
for name, m in models.items():
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    mae, r2 = mean_absolute_error(y_te, pred), r2_score(y_te, pred)
    results[name] = {"MAE": round(mae, 2), "R2": round(r2, 3)}
    print(f"{name:20s} MAE={mae:.2f}  R2={r2:.3f}")
    if r2 > best_r2:
        best_name, best_r2, best_model = name, r2, m

joblib.dump({"model": best_model, "features": list(X.columns)}, "model.pkl")
json.dump({"best": best_name, "results": results}, open("metrics.json", "w"), indent=2)
print(f"\nBest model: {best_name} (saved to model.pkl)")
