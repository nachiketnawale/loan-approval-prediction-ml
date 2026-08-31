import streamlit as st
import pandas as pd
import joblib


# --------------------------------
# 1. Load Model
# --------------------------------

model = joblib.load("loan_approval_model.pkl")


# --------------------------------
# 2. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

# --------------------------------
# Model Information Sidebar
# --------------------------------

with st.sidebar:

    st.header("📊 Model Information")

    st.write("**Algorithm:** Random Forest")

    st.metric(
        "Test Accuracy",
        "88.31%"
    )

    st.metric(
        "ROC-AUC",
        "0.888"
    )

    st.write("**Training Samples:** 304")

    st.write("**Testing Samples:** 77")

    st.divider()

    st.caption(
        "The model was trained on historical "
        "loan application data."
    )

    st.caption(
        "Predictions are for demonstration purposes "
        "and should not be treated as official banking decisions."
    )


# --------------------------------
# 3. Title
# --------------------------------

st.title("🏦 LoanWise")

st.subheader("AI-Powered Loan Approval Predictor")

st.write(
    "Enter applicant financial and personal information "
    "to estimate loan approval probability using our "
    "machine learning model."
)

st.divider()


# --------------------------------
# 4. Applicant Information
# --------------------------------

st.subheader("👤 Personal Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["0", "1", "2", "3+"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )


with col2:

    st.markdown(
        "<h3 style='margin-top: 0;'>💰 Financial Information</h3>",
        unsafe_allow_html=True
    )

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=5000
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0,
        value=0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0,
        value=150
    )

    loan_term = st.number_input(
        "Loan Term (months)",
        min_value=12,
        value=360
    )

    credit_history = st.selectbox(
        "Credit History",
        [1.0, 0.0]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )


# --------------------------------
# 5. Prediction Button
# --------------------------------

if st.button(
    "🔍 Predict Loan Approval",
    use_container_width=True
):

    # Create applicant DataFrame

    applicant = pd.DataFrame({
        "Gender": [gender],
        "Married": [married],
        "Dependents": [dependents],
        "Education": [education],
        "Self_Employed": [self_employed],
        "ApplicantIncome": [applicant_income],
        "CoapplicantIncome": [coapplicant_income],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_term],
        "Credit_History": [credit_history],
        "Property_Area": [property_area]
    })


    # --------------------------------
    # Prediction
    # --------------------------------

    prediction = model.predict(applicant)[0]

    probabilities = model.predict_proba(applicant)[0]

    approval_probability = probabilities[
        list(model.classes_).index("Y")
    ]


    # --------------------------------
    # Display Result
    # --------------------------------

    st.subheader("Prediction Result")


    if prediction == "Y":

        st.success(
            "✅ LOAN APPROVED"
        )

    else:

        st.error(
            "❌ LOAN REJECTED"
        )


# --------------------------------
# Approval Probability
# --------------------------------

st.metric(
    "Approval Probability",
    f"{approval_probability * 100:.2f}%"
)


# --------------------------------
# Probability Bar
# --------------------------------

st.progress(
    float(approval_probability)
)


# --------------------------------
# Risk Level
# --------------------------------

if approval_probability >= 0.75:

    risk_level = "LOW"
    risk_message = "🟢 Low Risk Indicator"

elif approval_probability >= 0.50:

    risk_level = "MODERATE"
    risk_message = "🟡 Moderate Risk Indicator"

else:

    risk_level = "HIGH"
    risk_message = "🔴 High Risk Indicator"


st.subheader("Model Risk Indicator")

st.write(
    f"**{risk_message}**"
)

st.caption(
    "Risk indicator is based only on the model's "
    "predicted approval probability and is not an "
    "official banking credit-risk assessment."
)