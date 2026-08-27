import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# --------------------------------
# 1. Load Dataset
# --------------------------------

data = pd.read_csv("data/loan_data.csv")


# --------------------------------
# 2. Remove Loan_ID
# --------------------------------

data = data.drop("Loan_ID", axis=1)


# --------------------------------
# 3. Handle Missing Values
# --------------------------------

categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed"
]

for column in categorical_columns:
    data[column] = data[column].fillna(data[column].mode()[0])


numerical_columns = [
    "Loan_Amount_Term",
    "Credit_History"
]

for column in numerical_columns:
    data[column] = data[column].fillna(data[column].median())


# --------------------------------
# 4. Separate X and y
# --------------------------------

X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]


# --------------------------------
# 5. Categorical Features
# --------------------------------

categorical_features = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]


# --------------------------------
# 6. Encoding
# --------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


X_encoded = preprocessor.fit_transform(X)


# --------------------------------
# 7. Train/Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------
# 8. Create Model
# --------------------------------

model = LogisticRegression(max_iter=1000)


# --------------------------------
# 9. Train Model
# --------------------------------

model.fit(X_train, y_train)


# --------------------------------
# 10. Make Predictions
# --------------------------------

y_pred = model.predict(X_test)


# --------------------------------
# 11. Calculate Accuracy
# --------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)