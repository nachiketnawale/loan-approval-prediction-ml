import pandas as pd

# Load the loan dataset
data = pd.read_csv("data/loan_data.csv")

# Display the first 5 rows
print(data.head())

# Display number of rows and columns
print("\nDataset Shape:")
print(data.shape)

# Display column names
print("\nColumn Names:")
print(data.columns.tolist())

# Check data types
print("\nData Types:")
print(data.dtypes)

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Check basic statistics
print("\nStatistical Summary:")
print(data.describe())