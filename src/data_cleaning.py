import pandas as pd


# Load the dataset
data = pd.read_csv("data/loan_data.csv")

print("Original Shape:")
print(data.shape)


# Remove Loan_ID because it does not help predict loan approval
data = data.drop("Loan_ID", axis=1)


# Fill missing categorical values with the most common value
categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed"
]

for column in categorical_columns:
    data[column] = data[column].fillna(data[column].mode()[0])


# Fill missing numerical values with the median
numerical_columns = [
    "Loan_Amount_Term",
    "Credit_History"
]

for column in numerical_columns:
    data[column] = data[column].fillna(data[column].median())


# Check missing values after cleaning
print("\nMissing Values After Cleaning:")
print(data.isnull().sum())


# Display cleaned data
print("\nCleaned Data:")
print(data.head())


# Display new shape
print("\nFinal Shape:")
print(data.shape)