import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


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
# 3. Define Features
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
# 7. Create Full Pipeline
# --------------------------------

model_pipeline = Pipeline(
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
# 8. Train on Full Dataset
# --------------------------------

model_pipeline.fit(X, y)


# --------------------------------
# 9. Save Model
# --------------------------------

joblib.dump(
    model_pipeline,
    "loan_approval_model.pkl"
)

print("Model trained successfully.")

print("Model saved as:")
print("loan_approval_model.pkl")