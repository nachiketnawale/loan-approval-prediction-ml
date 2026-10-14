import streamlit as st
import os
import pandas as pd
import joblib
import shap
import numpy as np


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="LoanWise",
    page_icon="🏦",
    layout="centered"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("loan_approval_model.pkl")

preprocessor = model.named_steps["preprocessing"]
random_forest = model.named_steps["model"]


# ============================================================
# HEADER
# ============================================================

st.title("🏦 LoanWise")

st.subheader(
    "AI-Powered Loan Approval Predictor"
)

st.write(
    "Enter the applicant's information below to estimate "
    "the probability of loan approval."
)

st.divider()


# ============================================================
# PERSONAL INFORMATION
# ============================================================

st.header("👤 Personal Information")

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


with col2:

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )


st.divider()


# ============================================================
# FINANCIAL INFORMATION
# ============================================================

st.header("💰 Financial Information")

col1, col2 = st.columns(2)

with col1:

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=5000,
        step=500
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0,
        value=0,
        step=500
    )

    loan_amount = st.number_input(
        "Loan Amount (in thousands)",
        min_value=0,
        value=150,
        step=10
    )


with col2:

    loan_term = st.number_input(
        "Loan Term (months)",
        min_value=12,
        value=360,
        step=12
    )

    credit_history = st.selectbox(
        "Credit History",
        [1.0, 0.0]
    )


st.divider()


# ============================================================
# APPLICANT DATA
# ============================================================

applicant = pd.DataFrame(
    {
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
        "Property_Area": [property_area],
    }
)


# ============================================================
# PREDICT
# ============================================================

st.header("🔍 Prediction")

predict_button = st.button(
    "Predict Loan Approval",
    use_container_width=True
)


if predict_button:

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(applicant)[0]

    probabilities = model.predict_proba(applicant)[0]

    approval_probability = probabilities[
        list(model.classes_).index("Y")
    ]

    approval_percentage = (
        approval_probability * 100
    )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    if prediction == "Y":

        st.success(
            "🎉 Loan Approved"
        )

        st.write(
            "The model predicts that this application "
            "is likely to be approved."
        )

    else:

        st.error(
            "❌ Loan Not Approved"
        )

        st.write(
            "The model predicts that this application "
            "is unlikely to be approved."
        )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    st.subheader(
        "Prediction Details"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Approval Probability",
            f"{approval_percentage:.2f}%"
        )

    with col2:

        if approval_probability >= 0.75:

            risk = "LOW"

        elif approval_probability >= 0.50:

            risk = "MODERATE"

        else:

            risk = "HIGH"

        st.metric(
            "Model Risk",
            risk
        )

    with col3:

        st.metric(
            "Algorithm",
            "Random Forest"
        )


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    st.write(
        "Approval Probability"
    )

    st.progress(
        float(approval_probability)
    )


    st.caption(
        "This probability is generated by the machine learning "
        "model."
    )


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.divider()

    st.header(
        "🧠 Why did the model make this prediction?"
    )

    st.write(
        "SHAP shows which features influenced this "
        "individual prediction."
    )


    applicant_transformed = (
        preprocessor.transform(applicant)
    )


    if hasattr(
        applicant_transformed,
        "toarray"
    ):

        applicant_transformed = (
            applicant_transformed.toarray()
        )


    explainer = shap.TreeExplainer(
        random_forest
    )


    shap_values = explainer.shap_values(
        applicant_transformed
    )


    # --------------------------------------------------------
    # HANDLE SHAP OUTPUT
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        shap_for_approval = (
            shap_values[1][0]
        )

    else:

        shap_array = np.asarray(
            shap_values
        )

        if shap_array.ndim == 3:

            shap_for_approval = (
                shap_array[0, :, 1]
            )

        elif shap_array.ndim == 2:

            if shap_array.shape[0] == 1:

                shap_for_approval = (
                    shap_array[0]
                )

            elif shap_array.shape[1] == 2:

                shap_for_approval = (
                    shap_array[:, 1]
                )

            else:

                shap_for_approval = (
                    shap_array[0]
                )

        else:

            shap_for_approval = (
                shap_array.flatten()
            )


    shap_for_approval = np.asarray(
        shap_for_approval
    ).flatten()


    # --------------------------------------------------------
    # FEATURE NAMES
    # --------------------------------------------------------

    feature_names = list(
        preprocessor.get_feature_names_out()
    )


    feature_names = [
        name
        .replace(
            "numerical__",
            ""
        )
        .replace(
            "categorical__",
            ""
        )
        .replace(
            "onehot__",
            ""
        )
        for name in feature_names
    ]


    # --------------------------------------------------------
    # SHAP TABLE
    # --------------------------------------------------------

    if len(shap_for_approval) == len(
        feature_names
    ):

        shap_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "SHAP Value": shap_for_approval
            }
        )

        shap_df["Impact"] = (
            shap_df["SHAP Value"].abs()
        )

        shap_df = shap_df.sort_values(
            "Impact",
            ascending=False
        ).head(8)


        st.bar_chart(
            shap_df.set_index(
                "Feature"
            )["SHAP Value"]
        )


    # ========================================================
    # GLOBAL FEATURE IMPORTANCE
    # ========================================================

    st.divider()

    st.header(
        "📊 Global Feature Importance"
    )

    st.write(
        "These features have the greatest overall influence "
        "on the Random Forest model."
    )


    importances = (
        random_forest.feature_importances_
    )


    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances
        }
    )


    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    ).head(10)


    st.bar_chart(
        importance_df.set_index(
            "Feature"
        )
    )



# ============================================================
# MODEL PERFORMANCE DASHBOARD
# ============================================================

st.divider()
st.header("📈 Model Performance Dashboard")

st.write(
    "Performance metrics are calculated on the held-out test set. "
    "Precision, recall, and F1-score below use loan approval (Y) "
    "as the positive class."
)

results_file = "model_evaluation_results.csv"

if os.path.exists(results_file):
    results = pd.read_csv(results_file)

    required_columns = {
        "Model", "Accuracy", "Precision", "Recall", "F1-Score"
    }

    if required_columns.issubset(results.columns):
        rf_results = results[results["Model"] == "Random Forest"]

        if not rf_results.empty:
            rf = rf_results.iloc[0]

            metric_col1, metric_col2 = st.columns(2)
            metric_col3, metric_col4 = st.columns(2)

            metric_col1.metric("Accuracy", f"{rf['Accuracy']:.2%}")
            metric_col2.metric("Precision", f"{rf['Precision']:.2%}")
            metric_col3.metric("Recall", f"{rf['Recall']:.2%}")
            metric_col4.metric("F1-Score", f"{rf['F1-Score']:.2%}")

            st.subheader("Compare Models")

            display_results = results.copy()
            metric_columns = [
                "Accuracy", "Precision", "Recall", "F1-Score"
            ]
            st.dataframe(
                display_results.style.format(
                    {column: "{:.2%}" for column in metric_columns}
                ),
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Metric Comparison")
            chart_data = results.set_index("Model")[metric_columns]
            st.bar_chart(chart_data, y=metric_columns)
        else:
            st.warning(
                "The evaluation file does not contain a Random Forest row. "
                "Run `python src/train_model.py` to regenerate it."
            )
    else:
        st.warning(
            "The evaluation CSV is missing required columns. "
            "Run `python src/train_model.py` using the updated training script."
        )
else:
    st.info(
        "Model evaluation results are not available yet. Run "
        "`python src/train_model.py` from the project root to create "
        "`model_evaluation_results.csv`, then refresh this page."
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🏦 LoanWise"
    )

    st.write(
        "AI-Powered Loan Approval Predictor"
    )

    st.divider()

    st.subheader(
        "🤖 Model Information"
    )

    st.write(
        "**Algorithm:** Random Forest"
    )

    st.write(
        "Test metrics are shown in the Model Performance Dashboard."
    )

    st.divider()

    st.subheader(
        "📊 Dataset"
    )

    st.write(
        "Training Samples: **304**"
    )

    st.write(
        "Testing Samples: **77**"
    )

    st.divider()

    
