🎓 Student Score Predictor for Indian Students
A machine learning-powered Streamlit web app that predicts student academic scores based on demographic and academic factors such as gender, parental education, study time, and module completion.

👉 Live Demo: student-score-predictor-for-indian-students.streamlit.app

📌 Features
🔍 Predict student scores based on user input

📊 Compare performance of Linear Regression, Random Forest, and XGBoost

📈 Feature importance and model evaluation charts

📥 Upload your own CSV dataset

✅ Accuracy up to 86% with cross-validation

📁 Dataset Format
Required CSV columns:

Copy
Edit
gender, ethnicity, parental_education, reading_time,
module_1, module_2, module_3, module_4, module_5, score
Example row:

rust
Copy
Edit
Male, General, Bachelor's degree, 60, 80, 75, 70, 85, 90, 76
🚀 How to Run Locally
bash
Copy
Edit
git clone https://github.com/yourusername/student-score-predictor.git
cd student-score-predictor

python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

pip install -r requirements.txt
streamlit run student_score_predictor.py
🛠 Tech Stack
Python 3.11+

Streamlit

scikit-learn

XGBoost

Pandas, NumPy

Matplotlib, Seaborn

📊 Sample Model Evaluation
Model	MAE	RMSE	R² Score
Random Forest	14.70	17.51	~-0.10
XGBoost Regressor	14.24	17.10	~-0.05
Linear Regression	14.08	17.16	~-0.06

Accuracy may vary depending on dataset quality and size.

📈 Visual Output
Feature Importance (Random Forest)

Coefficients (Linear Regression)

Model Comparison Charts

📮 Contact
GitHub: @yourusername

Live App: student-score-predictor-for-indian-students.streamlit.app
