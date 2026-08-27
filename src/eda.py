import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
data = pd.read_csv("data/loan_data.csv")


# Remove Loan_ID
data = data.drop("Loan_ID", axis=1)


# Handle missing values
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
# 1. Loan Approval Distribution
# --------------------------------

print("Loan Status Counts:")
print(data["Loan_Status"].value_counts())

sns.countplot(x="Loan_Status", data=data)

plt.title("Loan Approval Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Number of Applicants")

plt.show()

# 2. Credit History vs Loan Status

print("\nCredit History vs Loan Status:")
print(pd.crosstab(data["Credit_History"], data["Loan_Status"]))

sns.countplot(
    x="Credit_History",
    hue="Loan_Status",
    data=data
)

plt.title("Credit History vs Loan Approval")
plt.xlabel("Credit History")
plt.ylabel("Number of Applicants")

plt.show()

approval_rate = pd.crosstab(
    data["Credit_History"],
    data["Loan_Status"],
    normalize="index"
) * 100

print("\nApproval Rate by Credit History:")
print(approval_rate)

# 3. Applicant Income vs Loan Status
# --------------------------------

print("\nAverage Applicant Income by Loan Status:")
print(data.groupby("Loan_Status")["ApplicantIncome"].mean())


sns.boxplot(
    x="Loan_Status",
    y="ApplicantIncome",
    data=data
)

plt.title("Applicant Income vs Loan Approval")
plt.xlabel("Loan Status")
plt.ylabel("Applicant Income")

plt.show()

# 4. Loan Amount vs Loan Status
# --------------------------------

print("\nAverage Loan Amount by Loan Status:")
print(data.groupby("Loan_Status")["LoanAmount"].mean())

sns.boxplot(
    x="Loan_Status",
    y="LoanAmount",
    data=data
)

plt.title("Loan Amount vs Loan Approval")
plt.xlabel("Loan Status")
plt.ylabel("Loan Amount")

plt.show()

# 5. Education vs Loan Approval
# --------------------------------

education_approval = pd.crosstab(
    data["Education"],
    data["Loan_Status"],
    normalize="index"
) * 100

print("\nApproval Rate by Education:")
print(education_approval)

sns.countplot(
    x="Education",
    hue="Loan_Status",
    data=data
)

plt.title("Education vs Loan Approval")
plt.xlabel("Education")
plt.ylabel("Number of Applicants")

plt.show()

# 6. Property Area vs Loan Approval
# --------------------------------

property_approval = pd.crosstab(
    data["Property_Area"],
    data["Loan_Status"],
    normalize="index"
) * 100

print("\nApproval Rate by Property Area:")
print(property_approval)

sns.countplot(
    x="Property_Area",
    hue="Loan_Status",
    data=data
)

plt.title("Property Area vs Loan Approval")
plt.xlabel("Property Area")
plt.ylabel("Number of Applicants")

plt.show()