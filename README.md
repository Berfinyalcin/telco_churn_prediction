# TELCO CUSTOMER CHURN PEDICTION
This repository contains code for predicting customer churn for a fictional telecom company named Telco. The dataset includes information about customers' telecommunication services, such as home phone and internet services, and aims to identify which customers are likely to churn, stay, or sign up for services.

## Dataset Overview
The dataset (Telco-Customer-Churn.csv) includes the following columns:
- customerID: Unique identifier for each customer
- Gender: Customer's gender
- SeniorCitizen: Whether the customer is a senior citizen (1, 0)
- Partner: Whether the customer has a partner (Yes, No)
- Dependents: Whether the customer has dependents (Yes, No)
- tenure: Number of months the customer has stayed with the company
- PhoneService: Whether the customer has phone service (Yes, No)
- MultipleLines: Whether the customer has multiple lines (Yes, No, No phone service)
- InternetService: Customer's internet service provider (DSL, Fiber optic, No)
- OnlineSecurity: Whether the customer has online security (Yes, No, No internet service)
- OnlineBackup: Whether the customer has online backup (Yes, No, No internet service)
- DeviceProtection: Whether the customer has device protection (Yes, No, No internet service)
- TechSupport: Whether the customer has tech support (Yes, No, No internet service)
- StreamingTV: Whether the customer has streaming TV (Yes, No, No internet service)
- StreamingMovies: Whether the customer has streaming movies (Yes, No, No internet service)
- Contract: Customer's contract term (Month-to-month, One year, Two years)
- PaperlessBilling: Whether the customer has paperless billing (Yes, No)
- PaymentMethod: Customer's payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
- MonthlyCharges: Amount charged to the customer monthly
- TotalCharges: Total amount charged to the customer
- Churn: Whether the customer churned (Yes or No)

## Data Preprocessing
The code performs several preprocessing steps, including:

- Conversion of the 'TotalCharges' column to numeric types.
- Conversion of the 'Churn' column to binary values: 1 for "Yes" and 0 for "No".
- Exploratory Data Analysis (EDA) to understand the distribution of categorical and numerical features.
- Handling missing values and outliers.
- Feature extraction, creating new features based on existing columns.

## Feature Encoding
The code includes techniques for feature encoding:

- Label encoding for binary categorical columns.
- One-hot encoding for other categorical columns.

## Modeling
The following machine learning models are implemented for predicting customer churn:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- LightGBM
- CatBoost
The code uses cross-validation to evaluate the models' performance, including metrics such as accuracy, F1 score, and ROC-AUC.

## Hyperparameter Tuning
Grid search is employed to optimize hyperparameters for Random Forest, XGBoost, LightGBM, and CatBoost models.

## Feature Importance
The code includes functions to visualize feature importance for each model, providing insights into which features contribute most to the predictions.

## Results
The final models are evaluated using cross-validation, and their performance metrics are displayed.

