
import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# --------------------------------
# 1. Load Dataset and define output paths
# --------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "loan_data.csv")
METRICS_PATH = os.path.join(BASE_DIR, "model_evaluation_results.csv")
CONFUSION_MATRIX_PATH = os.path.join(BASE_DIR, "confusion_matrix.csv")

data = pd.read_csv(DATA_PATH)


# --------------------------------
# 2. Remove Loan_ID
# --------------------------------

data = data.drop("Loan_ID", axis=1)


# --------------------------------
# 3. Separate Features and Target
# --------------------------------

X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]


# --------------------------------
# 4. Identify Columns
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
# 5. Categorical Preprocessing
# --------------------------------

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# --------------------------------
# 6. Numerical Preprocessing
# --------------------------------

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# --------------------------------
# 7. Combine Preprocessing
# --------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", categorical_pipeline, categorical_features),
        ("numerical", numerical_pipeline, numerical_features)
    ]
)


# --------------------------------
# 8. Train/Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------
# 9. Evaluation Function
# --------------------------------

def evaluate_model(model_name, y_true, y_pred, y_prob=None):

    accuracy = accuracy_score(y_true, y_pred)

    # Loan_Status = Y means loan approved
    precision = precision_score(
        y_true, y_pred,
        pos_label="Y",
        zero_division=0
    )

    recall = recall_score(
        y_true, y_pred,
        pos_label="Y",
        zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred,
        pos_label="Y",
        zero_division=0
    )

    print("\n" + "=" * 50)
    print(f"MODEL EVALUATION: {model_name}")
    print("=" * 50)

    print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1-Score  : {f1:.4f} ({f1 * 100:.2f}%)")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=["N", "Y"]))

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=["N", "Y"],
            target_names=["Rejected (N)", "Approved (Y)"],
            zero_division=0
        )
    )

    if y_prob is not None:
        roc_auc = roc_auc_score(
            (y_true == "Y").astype(int),
            y_prob
        )
        print(f"ROC-AUC   : {roc_auc:.4f} ({roc_auc * 100:.2f}%)")

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }


# --------------------------------
# 10. Logistic Regression
# --------------------------------

logistic_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ]
)

logistic_pipeline.fit(X_train, y_train)

y_pred_lr = logistic_pipeline.predict(X_test)

lr_metrics = evaluate_model(
    "Logistic Regression",
    y_test,
    y_pred_lr
)


# --------------------------------
# 11. Decision Tree
# --------------------------------

decision_tree_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", DecisionTreeClassifier(random_state=42))
    ]
)

decision_tree_pipeline.fit(X_train, y_train)

y_pred_tree = decision_tree_pipeline.predict(X_test)

tree_metrics = evaluate_model(
    "Decision Tree",
    y_test,
    y_pred_tree
)


# --------------------------------
# 12. Random Forest
# --------------------------------

random_forest_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ]
)

random_forest_pipeline.fit(X_train, y_train)

y_pred_rf = random_forest_pipeline.predict(X_test)


# --------------------------------
# 13. Random Forest Probabilities
# --------------------------------

# Find the probability column corresponding to "Y"
approved_index = list(
    random_forest_pipeline.named_steps["model"].classes_
).index("Y")

y_probability = random_forest_pipeline.predict_proba(
    X_test
)[:, approved_index]


# --------------------------------
# 14. Evaluate Random Forest
# --------------------------------

rf_metrics = evaluate_model(
    "Random Forest",
    y_test,
    y_pred_rf,
    y_probability
)


# --------------------------------
# 15. Compare All Models
# --------------------------------

results = pd.DataFrame([
    lr_metrics,
    tree_metrics,
    rf_metrics
])

print("\n\n" + "=" * 75)
print("MODEL COMPARISON")
print("=" * 75)

print(
    results.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1-Score": "{:.4f}".format
        }
    )
)

# Save results for future use
results.to_csv(METRICS_PATH, index=False)

print(f"\nEvaluation results saved to {METRICS_PATH}")

# --------------------------------
# 16. Save Random Forest Confusion Matrix
# --------------------------------

cm = confusion_matrix(y_test, y_pred_rf, labels=["N", "Y"])

cm_df = pd.DataFrame(
    cm,
    index=["Actual Rejected (N)", "Actual Approved (Y)"],
    columns=["Predicted Rejected (N)", "Predicted Approved (Y)"]
)

cm_df.to_csv(CONFUSION_MATRIX_PATH)

print("\nRandom Forest confusion matrix:")
print(cm_df)
print(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")
