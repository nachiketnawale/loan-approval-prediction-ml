import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


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
st.header("📊 Model Performance Dashboard")
st.write(
    "These scores come from the held-out test split saved by the training script. "
    "Precision, recall, and F1-score use loan approval (Y) as the positive class."
)

results_path = "model_evaluation_results.csv"
data_path = "data/loan_data.csv"

if os.path.exists(results_path):
    try:
        results = pd.read_csv(results_path)
        required_cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

        if not all(col in results.columns for col in required_cols):
            st.error(
                "The evaluation CSV does not contain the expected columns. "
                "Run `python src/save_model.py` to regenerate it."
            )
        else:
            rf_rows = results[results["Model"] == "Random Forest"]
            if rf_rows.empty:
                st.error("No Random Forest results found in the evaluation CSV.")
            else:
                rf = rf_rows.iloc[0]
                score1, score2 = st.columns(2)
                score3, score4 = st.columns(2)
                score1.metric(" Accuracy", f"{float(rf['Accuracy']):.2%}")
                score2.metric(" Precision", f"{float(rf['Precision']):.2%}")
                score3.metric(" Recall", f"{float(rf['Recall']):.2%}")
                score4.metric(" F1-Score", f"{float(rf['F1-Score']):.2%}")
                st.metric("ROC-AUC", f"{float(rf['ROC-AUC']):.4f}")

                st.subheader("Saved Evaluation Results")
                formatted_results = results.copy()
                for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
                    formatted_results[col] = formatted_results[col].map(
                        lambda value: f"{float(value):.2%}"
                    )
                formatted_results["ROC-AUC"] = formatted_results["ROC-AUC"].map(
                    lambda value: f"{float(value):.4f}"
                )
                st.dataframe(formatted_results, use_container_width=True, hide_index=True)

                st.subheader("Metric Comparison")
                chart_data = results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]]
                st.bar_chart(chart_data)

                # Recreate the exact split used in src/save_model.py and display the matrix.
                if os.path.exists(data_path):
                    evaluation_data = pd.read_csv(data_path).drop(columns=["Loan_ID"], errors="ignore")
                    X_eval = evaluation_data.drop(columns=["Loan_Status"])
                    y_eval = evaluation_data["Loan_Status"]
                    _, X_test_eval, _, y_test_eval = train_test_split(
                        X_eval,
                        y_eval,
                        test_size=0.20,
                        random_state=42,
                        stratify=y_eval
                    )
                    y_pred_eval = model.predict(X_test_eval)
                    cm = confusion_matrix(y_test_eval, y_pred_eval, labels=["N", "Y"])

                    st.subheader("Confusion Matrix")
                    st.caption("Rows are actual labels; columns are predicted labels.")
                    cm_df = pd.DataFrame(
                        cm,
                        index=["Actual: Rejected (N)", "Actual: Approved (Y)"],
                        columns=["Predicted: Rejected (N)", "Predicted: Approved (Y)"]
                    )
                    st.dataframe(cm_df, use_container_width=True)

                    fig, ax = plt.subplots(figsize=(6, 4))
                    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
                    ax.set_title("Loan Approval Confusion Matrix")
                    ax.set_xlabel("Predicted label")
                    ax.set_ylabel("Actual label")
                    ax.set_xticks([0, 1], labels=["Rejected (N)", "Approved (Y)"])
                    ax.set_yticks([0, 1], labels=["Rejected (N)", "Approved (Y)"])
                    threshold = cm.max() / 2 if cm.max() else 0
                    for row in range(cm.shape[0]):
                        for col in range(cm.shape[1]):
                            ax.text(
                                col, row, str(cm[row, col]),
                                ha="center", va="center",
                                color="white" if cm[row, col] > threshold else "black",
                                fontsize=13
                            )
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning("Dataset not found, so the confusion matrix could not be generated.")
    except Exception as error:
        st.error(f"Could not load evaluation results: {error}")
else:
    st.warning(
        "Evaluation results are missing. From the project root, run "
        "`python src/save_model.py` first, then refresh this page."
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
        "See the Model Performance Dashboard for current held-out test metrics."
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

    
