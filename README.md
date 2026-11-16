# Clinical Machine Learning Benchmark: Heart Disease & Diabetes Progression

This repository contains the Python implementation and Jupyter notebooks for a graduate-level Machine Learning coursework project. The objective of this study is to evaluate standard data preprocessing pipelines and compare baseline supervised learning algorithms on two small, noisy clinical datasets.
 
**Context:** Master's Coursework Project (Data Science / Computer Science)

---

## Project Overview

Real-world medical datasets often contain missing values, mixed variable types, and limited sample sizes. Rather than focusing on heavy hyperparameter tuning, this project establishes clean, reproducible baselines to observe how different algorithm families behave under realistic data constraints:

1. **Cardiovascular Disease Classification (`heart_disease.ipynb`)**
   - **Dataset:** 305 patient records, 13 clinical predictors, binary target (`0` = no heart disease, `1` = heart disease).
   - **Preprocessing:** Median imputation for continuous variables, mode imputation for categorical variables, Label Encoding for binary features, One-Hot Encoding (`drop_first=True`) for multi-class symptoms (`Chest_Pain`), and Z-score standardization (`StandardScaler`) fitted strictly on the training set (`80/20` split).
   - **Models Evaluated:** K-Nearest Neighbors (KNN), Support Vector Machine (SVM), XGBoost, Decision Tree, Random Forest, and Logistic Regression.

2. **Diabetes Progression Regression (`diabetes_progression.ipynb`)**
   - **Dataset:** 454 patient records, 10 baseline physiological variables (`AGE`, `SEX`, `BMI`, `BP`, and serum measurements `S1`–`S6`), continuous progression target (`Y`).
   - **Preprocessing:** Mean imputation for numerical features, mode imputation for `SEX`, and Min-Max normalization (`MinMaxScaler`).
   - **Models Evaluated:** Lasso Regression, Ridge Regression, ElasticNet, and Stochastic Gradient Descent Regressor (`SGDRegressor`), evaluated via 5-fold cross-validation and a held-out test set.

---

## Summary of Baseline Results

### 1. Heart Disease Classification (Test Set, $N = 61$)
| Model | Test Accuracy | Macro F1-Score |
| :--- | :---: | :---: |
| **K-Nearest Neighbors (KNN)** | **63.93%** | **0.62** |
| Support Vector Machine (SVM) | 57.38% | 0.54 |
| XGBoost | 57.38% | 0.56 |
| Decision Tree | 55.74% | 0.55 |
| Random Forest | 50.82% | 0.48 |
| Logistic Regression | 47.54% | 0.45 |

### 2. Diabetes Progression Regression
| Model | Test MSE | Test $R^2$ | 5-Fold CV $R^2$ (Mean) |
| :--- | :---: | :---: | :---: |
| **Lasso ($\alpha = 0.01$)** | **3652.28** | **0.3316** | 0.4699 |
| Ridge ($\alpha = 0.01$) | 3652.66 | 0.3316 | 0.4695 |
| ElasticNet ($\alpha = 0.01$) | 3653.73 | 0.3314 | 0.4727 |
| SGDRegressor | 3677.82 | 0.3270 | **0.4756** |

---

## Repository Structure

```text
├── heart_disease_classification.ipynb   # Preprocessing & 6 classifiers benchmark
├── diabetes_regression.ipynb            # Preprocessing & 4 regularized linear models
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation
