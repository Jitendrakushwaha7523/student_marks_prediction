# 🎓 Student Marks Prediction

## Problem Statement
Predict a student's final exam marks from study habits and academic history, so that
teachers and students can spot at-risk learners early and act before exams.

## Objectives
1. Build a regression model that predicts final marks (0–100).
2. Compare multiple algorithms and select the best one.
3. Provide a simple web interface anyone can use without coding.

## Features (Inputs)
study_hours, attendance, previous_marks, assignment_score, sleep_hours, extra_classes

## Target (Output)
final_marks

## Methodology
Data → Cleaning → Train/Test split (80/20) → Train Linear Regression, Random Forest,
Gradient Boosting → Evaluate (MAE, R²) → Save best model → Streamlit app

## How to Run
```
pip install -r requirements.txt
python generate_data.py     # creates students.csv (or use your own data)
python train_model.py       # trains and saves model.pkl
streamlit run app.py        # opens the web app
```

## Using Real Data
Replace students.csv with your own (same column names), e.g. the UCI
"Student Performance" dataset, then re-run train_model.py.

## Evaluation Metrics
- MAE: average error in marks (lower is better)
- R²: how much variation the model explains (closer to 1 is better)

## Future Scope
Pass/fail classification, SHAP explanations, class-wide CSV upload, database of students.
