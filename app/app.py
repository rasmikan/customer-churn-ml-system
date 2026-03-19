import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Customer Churn ML System", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("C:/Users/HP/Desktop/projects/customer-churn-prediction/model.pkl")


model = load_model()

st.title("Customer Churn Prediction ML System")

tabs = st.tabs(["Prediction", "Batch Prediction", "Analytics"])

# =========================
# TAB 1 — SINGLE PREDICTION
# =========================

with tabs[0]:

    st.header("Predict Customer Churn")

    st.sidebar.header("Customer Information")

    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])

    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

    tenure = st.sidebar.number_input("Tenure (months)", 0, 72, 12)

    monthly_charges = st.sidebar.number_input(
        "Monthly Charges", 0.0, 200.0, 70.0
    )

    total_charges = st.sidebar.number_input(
        "Total Charges", 0.0, 10000.0, 2000.0
    )

    contract = st.sidebar.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )

    payment_method = st.sidebar.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    input_data = pd.DataFrame(
        {
            "gender": [gender],
            "SeniorCitizen": [0],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": ["Yes"],
            "MultipleLines": ["No"],
            "InternetService": ["DSL"],
            "OnlineSecurity": ["No"],
            "OnlineBackup": ["No"],
            "DeviceProtection": ["No"],
            "TechSupport": ["No"],
            "StreamingTV": ["No"],
            "StreamingMovies": ["No"],
            "Contract": [contract],
            "PaperlessBilling": ["Yes"],
            "PaymentMethod": [payment_method],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges],
            "avg_monthly_spend": [total_charges / (tenure + 1)],
        }
    )

    if st.button("Predict Churn"):

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        risk_score = probability * 100

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        col1.metric("Churn Probability", round(probability, 3))
        col2.metric("Customer Risk Score", round(risk_score, 1))

        st.progress(probability)

        if probability > 0.6:
            st.error("High Churn Risk")
            
            st.write("Recommended Retention Actions:")

            st.write("• Offer loyalty discount")
            st.write("• Promote yearly contract")
            st.write("• Provide premium customer support")
            

        elif probability > 0.3:
            st.warning("Medium Churn Risk")

        else:
            st.success("Low Churn Risk")

# =========================
# TAB 2 — BATCH PREDICTION
# =========================

with tabs[1]:

    st.header("Batch Customer Prediction")

    uploaded_file = st.file_uploader("Upload CSV file")

    if uploaded_file:

        data = pd.read_csv(uploaded_file)

        probabilities = model.predict_proba(data)[:, 1]

        data["Churn Probability"] = probabilities

        st.write(data)

        st.download_button(
            "Download Predictions",
            data.to_csv(index=False),
            "churn_predictions.csv",
        )

# =========================
# TAB 3 — ANALYTICS
# =========================

with tabs[2]:

    st.header("Customer Churn Analytics")

    df = pd.read_csv("C:\\Users\\HP\\Desktop\\projects\\customer-churn-prediction\\data\\raw\\Telco_Cusomer_Churn.csv")

    df["risk_level"] = pd.cut(
        df["Churn Probability"],
        bins=[0,0.3,0.6,1],
        labels=["Low","Medium","High"]
    )

    fig = px.pie(df, names="risk_level", title="Customer Risk Distribution")

    st.plotly_chart(fig)

    fig1 = px.histogram(df, x="tenure", color="Churn")

    fig2 = px.histogram(df, x="Contract", color="Churn")

    fig3 = px.histogram(df, x="MonthlyCharges", color="Churn")

    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    st.plotly_chart(fig3)