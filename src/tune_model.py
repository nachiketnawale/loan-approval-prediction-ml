import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------
# 1. Load Dataset
# --------------------------------

data = pd.read_csv("data/loan_data.csv")

data = data.drop("Loan_ID", axis=1)


# --------------------------------
# 2. Separate Features and Target
# --------------------------------

X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]


# --------------------------------
# 3. Define Columns
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
# 4. Categorical Pipeline
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
# 5. Numerical Pipeline
# --------------------------------

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# --------------------------------
# 6. Preprocessor
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
# 7. Train/Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------
# 8. Random Forest Pipeline
# --------------------------------

rf_pipeline = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                random_state=42
            )
        )
    ]
)


# --------------------------------
# 9. Hyperparameter Grid
# --------------------------------

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}


# --------------------------------
# 10. Grid Search
# --------------------------------

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)


# --------------------------------
# 11. Train Grid Search
# --------------------------------

print("Starting hyperparameter tuning...")

grid_search.fit(X_train, y_train)


# --------------------------------
# 12. Best Parameters
# --------------------------------

print("\nBest Parameters:")
print(grid_search.best_params_)


# --------------------------------
# 13. Best Cross-Validation Score
# --------------------------------

print("\nBest Cross-Validation Accuracy:")
print(grid_search.best_score_)


# --------------------------------
# 14. Test Set Evaluation
# --------------------------------

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nTest Accuracy:")
print(test_accuracy)


# --------------------------------
# 15. Classification Report
# --------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)