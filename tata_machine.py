import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained model
model_lgb = joblib.load('lgbm_model.pkl')

# Streamlit app
st.title("Machine Failure Prediction")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Detect file type and read accordingly
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine='openpyxl')

    # Keep original for later use
    df_original = df.copy()

    # Remove 'id' and 'Product ID'
    df_temp = df.drop(columns=['id', 'Product ID'], errors='ignore')

    # Identify categorical columns
    categorical_cols = df_temp.select_dtypes(include=['object']).columns

    # Encode categorical columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_temp[col] = le.fit_transform(df_temp[col])
        label_encoders[col] = le  # Store encoders for future use if needed

    # Remove special characters from column names
    df_temp.columns = df_temp.columns.str.replace(r"[\[\]<>]", "", regex=True)

    # Standardize numerical columns
    numerical_cols = ['Air temperature K', 'Process temperature K', 'Rotational speed rpm', 'Torque Nm', 'Tool wear min']
    scaler = StandardScaler()
    df_temp[numerical_cols] = scaler.fit_transform(df_temp[numerical_cols])

    # Make predictions
    y_pred = model_lgb.predict(df_temp)

    # Check if rows match before appending predictions
    if len(y_pred) == len(df_original):
        df_original['Machine failure'] = y_pred
        df_predict = df_original.copy()  # Final dataframe

        # Show results
        st.write("### Predictions")
        st.dataframe(df_predict)

        # Download options
        file_format = st.selectbox("Select file format:", ["CSV", "Excel"])

        if file_format == "CSV":
            csv = df_predict.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "predictions.csv", "text/csv")
        else:
            excel = df_predict.to_excel(index=False, engine='openpyxl')
            st.download_button("Download Excel", excel, "predictions.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.error(f"Row mismatch: y_pred = {len(y_pred)}, df = {len(df_original)}. Cannot proceed.")
