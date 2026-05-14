# Telco Customer Churn Prediction

Predicting which telecom customers will cancel their subscription using classical ML models.

## Problem

Customer churn costs telecom companies more than acquiring new ones. The goal is a model that flags high-risk customers so retention teams can intervene before they leave.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle.
- 7,043 customers, 21 features
- Target: `Churn` (Yes/No)
- Class balance: 26.6% churn rate (imbalanced)

## Approach

1. Cleaned `TotalCharges` (stored as text in the raw data; 11 rows with tenure=0 had empty values and were dropped).
2. EDA surfaced three strong signals: low tenure, month-to-month contracts, and high monthly charges all correlate with higher churn.
3. Built a scikit-learn pipeline with `StandardScaler` for numeric features and `OneHotEncoder` for categorical features. Pipeline keeps preprocessing inside cross-validation to prevent data leakage.
4. Compared three models with 5-fold stratified cross-validation, scored on ROC-AUC (better than accuracy for imbalanced classification).
5. Evaluated the best model on a held-out 20% test set.

## Results

| Model | CV ROC-AUC | Std |
|---|---|---|
| Logistic Regression | 0.846 | 0.005 |
| Random Forest | 0.830 | 0.008 |
| XGBoost | 0.821 | 0.008 |

Logistic regression won. On clean tabular data with mostly linear relationships, regularised linear models often beat tree-based ones, especially when the dataset is moderate in size.

### Test set performance (Logistic Regression)

| Metric | Score |
|---|---|
| Accuracy | 0.805 |
| Precision | 0.651 |
| Recall | 0.572 |
| F1 | 0.609 |
| ROC-AUC | 0.836 |

Recall on the churn class is 57%, meaning the model catches just over half of customers who will actually churn. For a retention use case this is the metric that matters most — false negatives (missed churners) cost more than false positives (extra retention calls). Threshold tuning could trade some precision for higher recall depending on the business cost ratio.

## Key churn drivers

Top features by absolute coefficient:
- Tenure (longer = lower churn risk)
- Contract type (two-year contracts dramatically reduce churn vs month-to-month)
- Internet service type (fiber optic customers churn more)
- Total charges
- Payment method (electronic check users churn more)

## How to run

1. Download the dataset from Kaggle (link above).
2. Open `telco_churn_prediction.ipynb` in Google Colab or Jupyter.
3. Update the CSV path in the first code cell to match your environment.
4. Run all cells.

## Tools

Python, pandas, scikit-learn, XGBoost, matplotlib, seaborn.
