
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# --------------------------------
# 1. Load Dataset
# --------------------------------

data = pd.read_csv("data/loan_data.csv")
data = data.drop("Loan_ID", axis=1)

X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]


# --------------------------------
# 2. Define Features
# --------------------------------

categorical_features = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

numerical_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]


# --------------------------------
# 3. Build Preprocessing Pipelines
# --------------------------------

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_pipeline, categorical_features),
        ("numerical", numerical_pipeline, numerical_features)
    ]
)


# --------------------------------
# 4. Split Dataset BEFORE Training
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------
# 5. Create Random Forest Pipeline
# --------------------------------

model_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)


# --------------------------------
# 6. Train ONLY on Training Data
# --------------------------------

model_pipeline.fit(X_train, y_train)


# --------------------------------
# 7. Evaluate on Unseen Test Data
# --------------------------------

y_pred = model_pipeline.predict(X_test)

approved_index = list(
    model_pipeline.classes_
).index("Y")

y_probability = model_pipeline.predict_proba(
    X_test
)[:, approved_index]

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred,
    pos_label="Y",
    zero_division=0
)

recall = recall_score(
    y_test, y_pred,
    pos_label="Y",
    zero_division=0
)

f1 = f1_score(
    y_test, y_pred,
    pos_label="Y",
    zero_division=0
)

roc_auc = roc_auc_score(
    (y_test == "Y").astype(int),
    y_probability
)


# --------------------------------
# 8. Print Evaluation Results
# --------------------------------

print("\n========== RANDOM FOREST EVALUATION ==========")

print(f"Accuracy  : {accuracy:.4f} ({accuracy:.2%})")
print(f"Precision : {precision:.4f} ({precision:.2%})")
print(f"Recall    : {recall:.4f} ({recall:.2%})")
print(f"F1-Score  : {f1:.4f} ({f1:.2%})")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test, y_pred,
        labels=["N", "Y"]
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=["N", "Y"],
        target_names=["Rejected", "Approved"],
        zero_division=0
    )
)


# --------------------------------
# 9. Save Metrics for Streamlit UI
# --------------------------------

results = pd.DataFrame([
    {
        "Model": "Random Forest",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    }
])

results.to_csv(
    "model_evaluation_results.csv",
    index=False
)

print("\nEvaluation results saved to model_evaluation_results.csv")


# --------------------------------
# 10. Save the Trained Model
# --------------------------------

joblib.dump(
    model_pipeline,
    "loan_approval_model.pkl"
)

print("\nModel trained and saved successfully.")
print("Model file: loan_approval_model.pkl")
