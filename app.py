import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np

# ==========================================
# 1. LOAD MODEL
# ==========================================

model = joblib.load("loan_approval_model.pkl")

# Extract pipeline components for explainability
preprocessor = model.named_steps["preprocessing"]
random_forest = model.named_steps["model"]


# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="LoanWise",
    page_icon="🏦",
    layout="centered"
)


# ==========================================
# 3. SIDEBAR - MODEL INFORMATION
# ==========================================

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


# ==========================================
# 4. HEADER
# ==========================================

st.title("🏦 LoanWise")

st.subheader(
    "AI-Powered Loan Approval Predictor"
)

st.write(
    "Enter applicant financial and personal information "
    "to estimate loan approval probability using our "
    "machine learning model."
)

st.divider()


# ==========================================
# 5. PERSONAL INFORMATION
# ==========================================

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


# ==========================================
# 6. FINANCIAL INFORMATION
# ==========================================

with col2:

    st.markdown(
        "### 💰 Financial Information"
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


# ==========================================
# 7. PREDICTION BUTTON
# ==========================================

if st.button(
    "🔍 Predict Loan Approval",
    use_container_width=True
):

    # --------------------------------------
    # Create Applicant DataFrame
    # --------------------------------------

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


    # ======================================
    # 8. MODEL PREDICTION
    # ======================================

    prediction = model.predict(
        applicant
    )[0]


    # ======================================
    # 9. APPROVAL PROBABILITY
    # ======================================

    probabilities = model.predict_proba(
        applicant
    )[0]

    approval_probability = probabilities[
        list(model.classes_).index("Y")
    ]


    # ======================================
    # 10. PREDICTION RESULT
    # ======================================

    st.divider()

    st.subheader("🎯 Prediction Result")


    if prediction == "Y":

        st.success(
            "✅ LOAN APPROVED"
        )

    else:

        st.error(
            "❌ LOAN REJECTED"
        )


    # ======================================
    # 11. APPROVAL PROBABILITY
    # ======================================

    st.metric(
        "Approval Probability",
        f"{approval_probability * 100:.2f}%"
    )


    st.progress(
        float(approval_probability)
    )


    # ======================================
    # 12. RISK INDICATOR
    # ======================================

    if approval_probability >= 0.75:

        risk_message = "🟢 Low Risk Indicator"

    elif approval_probability >= 0.50:

        risk_message = "🟡 Moderate Risk Indicator"

    else:

        risk_message = "🔴 High Risk Indicator"


    st.subheader(
        "Model Risk Indicator"
    )

    st.write(
        f"**{risk_message}**"
    )

    st.caption(
        "Risk indicator is based only on the model's "
        "predicted approval probability and is not an "
        "official banking credit-risk assessment."
    )


    # ======================================
    # 13. GLOBAL FEATURE IMPORTANCE
    # ======================================

    st.divider()

    st.subheader(
        "📊 Overall Model Feature Importance"
    )

    st.write(
        "These features were generally important "
        "to the Random Forest model."
    )


    preprocessor = model.named_steps[
        "preprocessing"
    ]

    random_forest = model.named_steps[
        "model"
    ]


    feature_names = (
        preprocessor
        .get_feature_names_out()
    )


    importances = (
        random_forest
        .feature_importances_
    )


    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importances

    })


    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
    )


    top_features = (
        importance_df
        .head(10)
    )


    st.bar_chart(
        top_features.set_index(
            "Feature"
        )
    )

# ======================================
# 14. SHAP INDIVIDUAL EXPLANATION
# ======================================

st.divider()

st.subheader(
    "🧠 Why Did the Model Predict This?"
)

st.write(
    "These features had the strongest influence "
    "on this individual prediction."
)


# --------------------------------------
# Transform applicant using preprocessing
# --------------------------------------

applicant_transformed = (
    preprocessor.transform(applicant)
)


# Convert sparse matrix to dense
if hasattr(
    applicant_transformed,
    "toarray"
):
    applicant_transformed = (
        applicant_transformed.toarray()
    )


# --------------------------------------
# Create SHAP explainer
# --------------------------------------

explainer = shap.TreeExplainer(
    random_forest
)


# --------------------------------------
# Calculate SHAP values
# --------------------------------------

shap_values = explainer.shap_values(
    applicant_transformed
)


# --------------------------------------
# Get feature names
# --------------------------------------

shap_feature_names = (
    preprocessor
    .get_feature_names_out()
)

# Make feature names user-friendly
shap_feature_names = [
    name
    .replace("numerical__", "")
    .replace("categorical__", "")
    .replace("onehot__", "")
    for name in shap_feature_names
]

n_features = len(
    shap_feature_names
)


# --------------------------------------
# Handle different SHAP versions
# --------------------------------------

if isinstance(shap_values, list):

    # Older SHAP versions
    shap_for_approval = np.asarray(
        shap_values[1]
    )

else:

    # Newer SHAP versions
    shap_for_approval = np.asarray(
        shap_values
    )


# --------------------------------------
# Remove applicant dimension
# --------------------------------------

if shap_for_approval.ndim == 3:

    # Example:
    # (1, features, classes)

    if shap_for_approval.shape[0] == 1:

        shap_for_approval = (
            shap_for_approval[0]
        )


# --------------------------------------
# Select approval class
# --------------------------------------

if shap_for_approval.ndim == 2:

    # Case:
    # (features, classes)

    if (
        shap_for_approval.shape[0]
        == n_features
    ):

        # Column 1 = approval class Y

        shap_for_approval = (
            shap_for_approval[:, 1]
        )

    # Case:
    # (classes, features)

    elif (
        shap_for_approval.shape[1]
        == n_features
    ):

        # Row 1 = approval class Y

        shap_for_approval = (
            shap_for_approval[1, :]
        )


# --------------------------------------
# Flatten final SHAP values
# --------------------------------------

shap_for_approval = np.asarray(
    shap_for_approval
).flatten()


# --------------------------------------
# Safety check
# --------------------------------------

if len(shap_for_approval) != n_features:

    st.error(
        f"SHAP feature mismatch: "
        f"{len(shap_for_approval)} values "
        f"for {n_features} features."
    )

else:

    # ----------------------------------
    # Create SHAP dataframe
    # ----------------------------------

    shap_df = pd.DataFrame({

        "Feature": shap_feature_names,

        "SHAP Value": shap_for_approval

    })


    # ----------------------------------
    # Calculate absolute impact
    # ----------------------------------

    shap_df["Absolute Impact"] = (
        shap_df["SHAP Value"].abs()
    )


    # ----------------------------------
    # Sort by importance
    # ----------------------------------

    shap_df = (
        shap_df
        .sort_values(
            by="Absolute Impact",
            ascending=False
        )
    )


    # ----------------------------------
    # Select top features
    # ----------------------------------

    top_shap = (
        shap_df
        .head(8)
    )


    # ----------------------------------
    # Display chart
    # ----------------------------------

    st.bar_chart(
        top_shap.set_index(
            "Feature"
        )["SHAP Value"]
    )


    st.caption(
        "Positive SHAP values push the prediction "
        "toward loan approval (Y), while negative "
        "values push the prediction toward rejection (N)."
    )