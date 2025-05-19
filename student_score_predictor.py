import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

st.title("🎓 Student Score Predictor for Indian Students")
st.markdown("""
This app predicts student academic scores based on demographic and educational features.
Upload your dataset (CSV) with columns: gender, ethnicity, parental_education, reading_time, module_1, module_2, module_3, module_4, module_5, and score.
""")

uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Sidebar for user input
    st.sidebar.header("Input Student Data for Prediction")

    def user_input_features():
        gender = st.sidebar.selectbox('Gender', data['gender'].unique())
        ethnicity = st.sidebar.selectbox('Ethnicity', data['ethnicity'].unique())
        parental_education = st.sidebar.selectbox('Parental Education Level', data['parental_education'].unique())
        reading_time = st.sidebar.slider('Reading Time (minutes per day)', 0, 180, 60)
        module_1 = st.sidebar.slider('Module 1 Completion (%)', 0, 100, 50)
        module_2 = st.sidebar.slider('Module 2 Completion (%)', 0, 100, 50)
        module_3 = st.sidebar.slider('Module 3 Completion (%)', 0, 100, 50)
        module_4 = st.sidebar.slider('Module 4 Completion (%)', 0, 100, 50)
        module_5 = st.sidebar.slider('Module 5 Completion (%)', 0, 100, 50)

        input_dict = {
            'gender': gender,
            'ethnicity': ethnicity,
            'parental_education': parental_education,
            'reading_time': reading_time,
            'module_1': module_1,
            'module_2': module_2,
            'module_3': module_3,
            'module_4': module_4,
            'module_5': module_5
        }
        return pd.DataFrame(input_dict, index=[0])

    input_df = user_input_features()

    # Encoding categorical features
    label_encoders = {}
    for col in ['gender', 'ethnicity', 'parental_education']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        input_df[col] = le.transform(input_df[col])
        label_encoders[col] = le

    if 'score' not in data.columns:
        st.error("Your dataset must contain a 'score' column as the target variable.")
    else:
        X = data.drop('score', axis=1)
        y = data['score']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Models
        rf_model = RandomForestRegressor(random_state=42)
        lr_model = LinearRegression()
        xgb_model = XGBRegressor(random_state=42, objective='reg:squarederror')

        # Training
        rf_model.fit(X_train, y_train)
        lr_model.fit(X_train, y_train)
        xgb_model.fit(X_train, y_train)

        # Predictions on user input
        prediction = rf_model.predict(input_df)
        st.subheader("Predicted Student Score (Random Forest)")
        st.metric(label="Expected Score", value=f"{prediction[0]:.2f}")

        # Predictions on test set
        rf_pred = rf_model.predict(X_test)
        lr_pred = lr_model.predict(X_test)
        xgb_pred = xgb_model.predict(X_test)

        st.subheader("Model Performance on Test Set")

        # Random Forest metrics
        st.write("**Random Forest Regressor**")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, rf_pred)):.2f}")
        st.write(f"MAE: {mean_absolute_error(y_test, rf_pred):.2f}")
        r2_rf = r2_score(y_test, rf_pred)
        st.write(f"R²: {r2_rf:.2f} ({r2_rf * 100:.2f}% accuracy)")

        # XGBoost metrics
        st.write("**XGBoost Regressor**")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, xgb_pred)):.2f}")
        st.write(f"MAE: {mean_absolute_error(y_test, xgb_pred):.2f}")
        r2_xgb = r2_score(y_test, xgb_pred)
        st.write(f"R²: {r2_xgb:.2f} ({r2_xgb * 100:.2f}% accuracy)")

        # Linear Regression metrics
        st.write("**Linear Regression**")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, lr_pred)):.2f}")
        st.write(f"MAE: {mean_absolute_error(y_test, lr_pred):.2f}")
        r2_lr = r2_score(y_test, lr_pred)
        st.write(f"R²: {r2_lr:.2f} ({r2_lr * 100:.2f}% accuracy)")

        # Cross-validation score for XGBoost (5-fold)
        cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='r2')
        st.write(f"XGBoost 5-Fold CV Mean R² Score: {cv_scores.mean():.2f}")

        # Feature Importance (Random Forest)
        st.subheader("🔍 Feature Importance (Random Forest)")
        importances = rf_model.feature_importances_
        feat_names = X.columns
        imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values(by="Importance", ascending=False)
        fig, ax = plt.subplots()
        sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax)
        ax.set_title("Feature Importance from Random Forest")
        st.pyplot(fig)

        # Linear Regression Coefficients
        st.subheader("🔍 Feature Coefficients (Linear Regression)")
        coef_df = pd.DataFrame({"Feature": feat_names, "Coefficient": lr_model.coef_}).sort_values(by="Coefficient", key=abs, ascending=False)
        fig2, ax2 = plt.subplots()
        sns.barplot(data=coef_df, x="Coefficient", y="Feature", ax=ax2)
        ax2.set_title("Linear Regression Coefficients")
        st.pyplot(fig2)

        # Additional Graph 1: Actual vs Predicted (XGBoost)
        st.subheader("📈 Actual vs Predicted (XGBoost)")
        fig3, ax3 = plt.subplots()
        ax3.scatter(y_test, xgb_pred, alpha=0.7, color="blue")
        ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax3.set_xlabel("Actual Scores")
        ax3.set_ylabel("Predicted Scores")
        ax3.set_title("Actual vs Predicted (XGBoost)")
        st.pyplot(fig3)

        # Additional Graph 2: Residual Plot (XGBoost)
        st.subheader("📉 Residuals Plot (XGBoost)")
        residuals = y_test - xgb_pred
        fig4, ax4 = plt.subplots()
        ax4.scatter(xgb_pred, residuals, alpha=0.7, color="green")
        ax4.axhline(0, linestyle='--', color='red')
        ax4.set_xlabel("Predicted Scores")
        ax4.set_ylabel("Residuals")
        ax4.set_title("Residual Plot (XGBoost)")
        st.pyplot(fig4)

else:
    st.warning("Please upload a dataset to begin analysis.")
