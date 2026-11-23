# ❤️ Health Check ML

An interactive Streamlit web app for exploring a heart-disease dataset and comparing machine-learning models, with a focus on **recall**: catching as many true positive cases as possible.

> ⚠️ **Educational project.** This is a machine-learning learning project, not a medical device. It does not provide medical advice, diagnosis, or treatment.

## 🚀 Live Demo

**[Open the live app](https://health-check-ml.streamlit.app)** and click **"Try the bundled sample dataset"** to explore instantly, no file upload needed.

## ✨ What the app does

- 📊 **Data overview**: preview, missing values, statistics, and data types for any uploaded CSV
- 🔍 **Exploratory data analysis**: target distribution, correlation heatmap, and feature distributions (Plotly)
- ⚖️ **Class-imbalance handling**: `class_weight="balanced"` and optional SMOTE oversampling
- 🤖 **Model comparison**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN, and Naive Bayes
- 🔧 **Recall-focused tuning**: GridSearchCV scored directly on recall, plus cross-validated metrics
- 🎚️ **Decision-threshold control**: slide the classification threshold and watch the precision/recall trade-off live, with an F2-based "maximize recall" suggestion
- 📈 **Evaluation visuals**: ROC curves, precision-recall curves, confusion matrix, feature importance, and an overfitting-gap warning (cross-validation recall vs. test recall)

## 📊 Sample results (bundled `heart.csv`)

Measured by running the app's own default pipeline (balanced class weights + SMOTE + recall-scored grid search) on the bundled `heart.csv`. The app automatically removes duplicate rows (1,025 rows → 302 unique records), then applies a stratified 80/20 split (`random_state=42`): 241 training rows, 61 test rows, threshold 0.5.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Naive Bayes | 83.6% | 82.9% | **87.9%** | 85.3% | **0.887** |
| KNN (k=9) | 83.6% | 82.9% | **87.9%** | 85.3% | 0.862 |
| Logistic Regression | 80.3% | 80.0% | 84.8% | 82.4% | 0.882 |
| Random Forest | 82.0% | 84.4% | 81.8% | 83.1% | 0.883 |
| SVM (linear) | 78.7% | 81.2% | 78.8% | 80.0% | 0.873 |
| Gradient Boosting | 77.0% | 78.8% | 78.8% | 78.8% | 0.855 |
| Decision Tree | 80.3% | 86.2% | 75.8% | 80.6% | 0.860 |

Hyperparameters were tuned by stratified cross-validation on the training split only (`GridSearchCV` fit on `X_train`); the test set is used exactly once, for the metrics above, at the default 0.5 threshold.

These numbers reproduce exactly with the app's default settings. Change the dataset, options, or threshold in the app and the results update live.

## 🗂️ Bundled datasets

| File | Rows | Notes |
|---|---|---|
| `heart.csv` | 1,025 | Classic UCI heart-disease features, binary `target` column. The default sample dataset. |
| `heart_disease_uci.csv` | 921 | UCI Heart Disease (multi-site), mixed categorical/numeric features, `num` target |
| `heart_2020_cleaned.csv` | 319,796 | CDC BRFSS 2020 survey subset, `HeartDisease` Yes/No target (large, so training is slower) |

You can also upload any binary-classification CSV and select the target column inside the app.

## 🛠️ Run locally

```bash
git clone https://github.com/ahmedharb2004/Health-Check-ML.git
cd Health-Check-ML
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 and click **"Try the bundled sample dataset"**, or upload your own CSV.

## 🧰 Tech stack

Python · Streamlit · scikit-learn · imbalanced-learn (SMOTE) · Plotly · Pandas · NumPy

## 👤 Author

**Ahmed Harb**
- GitHub: [@ahmedharb2004](https://github.com/ahmedharb2004)
- LinkedIn: [Ahmed Harb](https://www.linkedin.com/in/ahmed-harb-735786322/)
