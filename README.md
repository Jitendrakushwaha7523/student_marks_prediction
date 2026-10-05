# 🎓 Student Marks Prediction System

## Problem Statement
Predict a student's final exam marks from study habits and academic history so that
teachers can identify at-risk students early and help them before the exam.

## Objectives
1. Build regression models that predict final marks (0–100).
2. Compare multiple algorithms and select the best one.
3. Provide a multi-page web app that anyone can use without coding.

## Dataset
students.csv – 1000 students. Inputs: study_hours, attendance, previous_marks,
assignment_score, sleep_hours, extra_classes. Target: final_marks.
(Synthetic data. Replace with a real dataset such as UCI "Student Performance".)

## Methodology
Data -> Cleaning -> 80/20 split -> Linear Regression / Random Forest / Gradient Boosting
-> Evaluate (MAE, R²) -> Save best model -> Streamlit web app

## App Pages
| Page | What it does |
|------|--------------|
| Home | Project overview and key numbers |
| Data Explorer | Dataset, distributions, correlation matrix |
| Predict | Predict one student's marks, grade, tips, class comparison |
| Model Performance | Model comparison, actual vs predicted, error chart, feature importance |
| Batch Prediction | Upload CSV, predict whole class, download results |
| What If | See how changing one habit changes the marks |

## How to Run
```
pip install -r requirements.txt
python generate_data.py      # creates students.csv (skip if you have your own)
python train_model.py        # trains and saves the model
streamlit run app.py         # opens the web app
```

## Project Structure
```
app.py                  Home page
pages/                  The other 5 pages
utils.py                Shared helper functions
generate_data.py        Makes the sample dataset
train_model.py          Trains and saves the model
students.csv, model.pkl, metrics.json, test_predictions.csv
```

## Evaluation Metrics
MAE (average error in marks, lower is better) and R² (closer to 1 is better).

## Future Scope
Pass/fail classification, SHAP explanations, login for teachers, database storage,
real school data, deploy online with Streamlit Cloud.
