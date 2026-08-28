import pandas as pd
import joblib


# --------------------------------
# 1. Load Saved Model
# --------------------------------

model = joblib.load("loan_approval_model.pkl")


# --------------------------------
# 2. Create New Applicant
# --------------------------------

applicant = pd.DataFrame({
    "Gender": ["Male"],
    "Married": ["Yes"],
    "Dependents": ["1"],
    "Education": ["Graduate"],
    "Self_Employed": ["No"],
    "ApplicantIncome": [5000],
    "CoapplicantIncome": [2000],
    "LoanAmount": [150],
    "Loan_Amount_Term": [360],
    "Credit_History": [1.0],
    "Property_Area": ["Urban"]
})


# --------------------------------
# 3. Make Prediction
# --------------------------------

prediction = model.predict(applicant)[0]


# --------------------------------
# 4. Get Approval Probability
# --------------------------------

probabilities = model.predict_proba(applicant)[0]

approval_probability = probabilities[
    list(model.classes_).index("Y")
]


# --------------------------------
# 5. Display Result
# --------------------------------

print("\n==============================")
print("     LOAN APPROVAL RESULT")
print("==============================")

print("\nPrediction:")

if prediction == "Y":
    print("LOAN APPROVED")
else:
    print("LOAN REJECTED")


print(
    f"\nApproval Probability: "
    f"{approval_probability * 100:.2f}%"
)

print("==============================")