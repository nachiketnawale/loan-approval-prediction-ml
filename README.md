#  LoanWise — AI-Powered Loan Approval Predictor

## 🚀 Live Demo

Try LoanWise here:

👉 **[Launch LoanWise](https://loan-approval-prediction-ml-hehhkphm4kyn4ltvucbnzs.streamlit.app/)**

The application provides:
- Loan approval prediction
- Approval probability
- Risk indication
- Global feature importance
- Individual SHAP explanations

LoanWise is a machine learning application that predicts whether a loan application is likely to be approved based on applicant financial and personal information.

The project uses a Random Forest classification model and provides both prediction probabilities and model explainability using SHAP.

##  Project Overview

The goal of this project is to build an end-to-end machine learning system for loan approval prediction.

The system covers:

- Data loading
- Data cleaning
- Exploratory Data Analysis
- Feature preprocessing
- Categorical feature encoding
- Model training
- Model comparison
- Hyperparameter tuning
- Model evaluation
- ROC-AUC analysis
- Model persistence
- Streamlit deployment
- SHAP-based model explainability

##  Machine Learning Models

Several classification algorithms were evaluated:

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 85.71% |
| Decision Tree | 76.62% |
| Random Forest | **88.31%** |

Random Forest was selected as the final model because it achieved the highest test accuracy among the evaluated models.

##  Model Performance

### Random Forest

**Test Accuracy:** 88.31%

**ROC-AUC:** 0.888

The model was trained and evaluated using a train-test split.

### Dataset Split

- Training samples: 304
- Testing samples: 77
- Original features: 11
- Encoded features: 20

##  Exploratory Data Analysis

The project includes analysis of relationships between applicant characteristics and loan approval.

Examples include:

- Loan amount vs. loan status
- Education vs. approval rate
- Property area vs. approval rate
- Credit history
- Applicant income
- Coapplicant income



## Machine Learning Pipeline

The project uses a preprocessing and classification pipeline.

```text
Raw Applicant Data
        ↓
Data Preprocessing
        ↓
Categorical Encoding
        ↓
Feature Transformation
        ↓
Random Forest Classifier
        ↓
Loan Approval Predictiongit