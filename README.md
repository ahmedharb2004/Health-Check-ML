# ❤️ Heart Disease Analyzer

A Machine Learning web application built with Streamlit for analyzing heart disease datasets and comparing multiple classification models.

## 📌 Project Overview

This project provides an interactive interface for:

- 📊 Data Overview
- 🔍 Exploratory Data Analysis (EDA)
- 🤖 Machine Learning Model Comparison
- ⚖️ Handling Class Imbalance
- 🎯 Recall-focused training
- 🔧 Hyperparameter Tuning using GridSearchCV
- 🔄 Stratified Cross-Validation
- 📈 ROC Curve and Precision-Recall Curve
- 🎚️ Decision Threshold Tuning

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Imbalanced-learn (SMOTE)

## 🤖 Machine Learning Models

The application compares:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. Support Vector Machine (SVM)
6. K-Nearest Neighbors (KNN)
7. Gaussian Naive Bayes

## ⚖️ Class Imbalance

The project provides two approaches for handling class imbalance:

- `class_weight="balanced"`
- SMOTE (Synthetic Minority Oversampling Technique)

## 🔧 Hyperparameter Tuning

GridSearchCV is used to search for the best hyperparameters.

The tuning objective is:

```text
Recall
