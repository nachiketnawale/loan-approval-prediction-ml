import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer



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
# 4. Separate Features and Target
# --------------------------------

X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]


# --------------------------------
# 5. Identify Categorical Columns
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
# 6. Create Encoder
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


# --------------------------------
# 7. Transform Features
# --------------------------------

X_encoded = preprocessor.fit_transform(X)


# --------------------------------
# 8. Display Results
# --------------------------------

print("Original X shape:")
print(X.shape)

print("\nEncoded X shape:")
print(X_encoded.shape)

print("\nTarget shape:")
print(y.shape)

X_encoded = preprocessor.fit_transform(X)
# --------------------------------
# 9. Train/Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------
# 10. Display Split Results
# --------------------------------

print("\nTraining Features:")
print(X_train.shape)

print("\nTesting Features:")
print(X_test.shape)

print("\nTraining Target:")
print(y_train.shape)

print("\nTesting Target:")
print(y_test.shape)