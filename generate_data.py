"""Creates a realistic sample dataset (students.csv).
Using real data? Replace students.csv with your own file that has the same columns."""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
n = 1000
df = pd.DataFrame({
    "study_hours":      rng.uniform(0, 10, n).round(1),
    "attendance":       rng.integers(50, 101, n),
    "previous_marks":   rng.integers(30, 100, n),
    "assignment_score": rng.integers(30, 101, n),
    "sleep_hours":      rng.uniform(4, 9, n).round(1),
    "extra_classes":    rng.integers(0, 2, n),
})
df["final_marks"] = (
    5 + 0.30 * df.previous_marks + 0.15 * df.assignment_score
    + 0.10 * df.attendance + 2.5 * df.study_hours
    + 1.0 * df.sleep_hours.clip(upper=8) + 3 * df.extra_classes
    + rng.normal(0, 3, n)
).clip(0, 100).round(1)

df.to_csv("students.csv", index=False)
print("Saved students.csv", df.shape)
