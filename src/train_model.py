import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# --------------------------------
# 1. Load Dataset
# --------------------------------

data = pd.read_csv("data/loan_data.csv")


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
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


# --------------------------------
# 6. Numerical Preprocessing
# --------------------------------

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# --------------------------------
# 7. Combine Preprocessing
# --------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        )
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
# 9. Create Complete ML Pipeline
# --------------------------------

model_pipeline = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(max_iter=1000)
        )
    ]
)


# --------------------------------
# 10. Train Model
# --------------------------------

model_pipeline.fit(X_train, y_train)


# --------------------------------
# 11. Make Predictions
# --------------------------------

y_pred = model_pipeline.predict(X_test)


# 12. Evaluate Model


accuracy = accuracy_score(y_test, y_pred)

print("\nLogistic Regression Accuracy:")
print(accuracy)


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------
# 13. Decision Tree Model
# --------------------------------

decision_tree_pipeline = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor
        ),
        (
            "model",
            DecisionTreeClassifier(
                random_state=42
            )
        )
    ]
)


# --------------------------------
# 14. Train Decision Tree
# --------------------------------

decision_tree_pipeline.fit(X_train, y_train)


# --------------------------------
# 15. Make Predictions
# --------------------------------

y_pred_tree = decision_tree_pipeline.predict(X_test)


# --------------------------------
# 16. Evaluate Decision Tree
# --------------------------------

tree_accuracy = accuracy_score(
    y_test,
    y_pred_tree
)

print("\nDecision Tree Accuracy:")
print(tree_accuracy)

print("\nDecision Tree Classification Report:")
print(
    classification_report(
        y_test,
        y_pred_tree
    )
)


# --------------------------------
# 17. Random Forest Model
# --------------------------------

random_forest_pipeline = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor
        ),
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
# 18. Train Random Forest
# --------------------------------

random_forest_pipeline.fit(
    X_train,
    y_train
)


# --------------------------------
# 19. Make Predictions
# --------------------------------

y_pred_rf = random_forest_pipeline.predict(X_test)


# --------------------------------
# 20. Evaluate Random Forest
# --------------------------------

rf_accuracy = accuracy_score(
    y_test,
    y_pred_rf
)

print("\nRandom Forest Accuracy:")
print(rf_accuracy)

print("\nRandom Forest Classification Report:")
print(
    classification_report(
        y_test,
        y_pred_rf
    )
)