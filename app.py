import hashlib
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

# Optional dependency: SMOTE for oversampling the minority class.
# The app still works without it (falls back to class_weight="balanced" only).
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Health Check ML",
    page_icon="❤️",
    layout="wide"
)


# =========================================================
# CUSTOM STYLE
# =========================================================

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #E63946;
    text-align: center;
}

.sub-title {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}

div[data-testid="metric-container"] {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 10px;
    border: 1px solid #eee;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">❤️ Health Check ML</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Upload your dataset, explore it, and compare ML models '
    '(tuned for recall)</div>',
    unsafe_allow_html=True
)

st.caption(
    "⚠️ Educational machine-learning project for learning and demonstration. "
    "It is not a medical device and does not provide medical advice or diagnosis."
)


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

# Sample dataset: lets visitors try the full app instantly,
# without needing their own CSV file.
if uploaded_file is None:
    st.info("👆 Upload your own CSV file, or try the app instantly with the bundled sample dataset.")
    if not st.button("🫀 Try the bundled sample dataset (heart.csv)", type="primary"):
        st.stop()


# =========================================================
# READ DATA
# =========================================================

try:
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        df = pd.read_csv(uploaded_file)
    else:
        sample_path = Path(__file__).parent / "heart.csv"
        file_bytes = sample_path.read_bytes()
        df = pd.read_csv(sample_path)
        st.caption(
            "Using the bundled sample dataset: heart.csv "
            "(1,025 rows, UCI Heart Disease, target column: target)."
        )
except Exception as e:
    st.error(f"Error reading the CSV file: {e}")
    st.stop()


if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()


# =========================================================
# RESET STALE SESSION STATE ON NEW / RE-UPLOADED FILE
# =========================================================


uploaded_file_hash = hashlib.sha256(file_bytes).hexdigest()

if st.session_state.get("last_uploaded_hash") != uploaded_file_hash:
    _keys_to_clear = [
        "results_cv", "roc_data", "pr_data", "trained_models",
        "all_probs", "best_params_by_model", "y_test", "X_train",
        "numeric_features", "categorical_features", "fingerprint",
        "threshold", "all_predictions", "id_like_cols",
    ]
    for _k in _keys_to_clear:
        st.session_state.pop(_k, None)
    st.session_state["last_uploaded_hash"] = uploaded_file_hash


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs(
    ["📊 Data Overview", "🔍 EDA", "🤖 Model Comparison"]
)


# =========================================================
# TAB 1: DATA OVERVIEW  (unchanged)
# =========================================================

with tab1:

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Duplicates", int(df.duplicated().sum()))

    st.subheader("Missing Values")

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        st.success("✅ No missing values found.")
    else:
        missing_df = pd.DataFrame({
            "Column": missing.index,
            "Missing Values": missing.values
        })
        st.dataframe(missing_df, use_container_width=True)

    st.subheader("Statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    st.subheader("Data Types")
    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })
    st.dataframe(dtype_df, use_container_width=True)


# =========================================================
# TAB 2: EDA  (unchanged)
# =========================================================

with tab2:

    st.subheader("Exploratory Data Analysis")

    target_col = st.selectbox(
        "Select target column (disease label)",
        df.columns,
        index=len(df.columns) - 1
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Target Distribution")
        fig = px.pie(df, names=target_col, hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=np.number)

        if numeric_df.shape[1] >= 2:
            corr = numeric_df.corr()
            fig2 = px.imshow(
                corr, text_auto=".2f",
                color_continuous_scale="RdBu_r", aspect="auto"
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Not enough numerical columns for correlation analysis.")

    st.subheader("Feature Distribution")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) > 0:
        feature = st.selectbox("Choose a numerical feature", numeric_cols)
        fig3 = px.histogram(
            df, x=feature, color=target_col,
            marginal="box", barmode="overlay"
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No numerical features available.")


# =========================================================
# TAB 3: MODEL COMPARISON  (rebuilt for recall)
# =========================================================

with tab3:

    st.subheader("⚙️ Model Setup")

    col1, col2 = st.columns(2)

    with col1:
        target = st.selectbox(
            "Target column", df.columns,
            index=len(df.columns) - 1, key="model_target"
        )

    with col2:
        test_size = st.slider(
            "Test size (%)", min_value=10, max_value=40, value=20
        ) / 100

    # =====================================================
    # PREPARE X AND Y
    # =====================================================

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # =====================================================
    # AUTO FEATURE ENGINEERING: DROP ID-LIKE COLUMNS
    # =====================================================

    def detect_id_like_columns(frame: pd.DataFrame) -> list:
        id_like = []
        for col in frame.columns:
            name_hint = "id" in col.lower()
            all_unique = frame[col].nunique(dropna=True) == len(frame)
            if name_hint or all_unique:
                id_like.append(col)
        return id_like

    id_like_cols = detect_id_like_columns(X)

    if id_like_cols:
        X = X.drop(columns=id_like_cols)

    st.session_state["id_like_cols"] = id_like_cols

    # -----------------------------------------------------
    # REMOVE DUPLICATES
    # -----------------------------------------------------

    data = pd.concat([X, y], axis=1)
    duplicate_count = data.duplicated().sum()

    if duplicate_count > 0:
        data = data.drop_duplicates()
        X = data.drop(columns=[target])
        y = data[target]

    # =====================================================
    # TARGET ENCODING
    # =====================================================

    if y.isnull().sum() > 0:
        st.error(
            "The target column contains missing values. "
            "Please remove them before training."
        )
        st.stop()

    if y.dtype == "object":
        unique_values = y.unique()

        if len(unique_values) != 2:
            st.error(
                "The target column must contain exactly two classes "
                "for binary heart disease classification."
            )
            st.stop()

        mapping = {value: index for index, value in enumerate(unique_values)}
        y = y.map(mapping)

    else:
        # 0 = No Disease, 1..4 = Disease (classic heart-disease encoding)
        if set(y.dropna().unique()).issubset({0, 1, 2, 3, 4}):
            y = (y > 0).astype(int)
        else:
            unique_values = y.unique()

            if len(unique_values) != 2:
                st.error("The target column must contain exactly two classes.")
                st.stop()

            mapping = {
                value: index
                for index, value in enumerate(sorted(unique_values))
            }
            y = y.map(mapping)

    # =====================================================
    # CLASS BALANCE OVERVIEW
    # =====================================================

    st.subheader("⚖️ Class Balance")

    class_counts = y.value_counts().sort_index()
    imbalance_ratio = class_counts.max() / class_counts.min()

    cb1, cb2 = st.columns([1, 2])

    with cb1:
        st.dataframe(
            class_counts.rename("Count").rename_axis("Class").reset_index(),
            use_container_width=True
        )
        if imbalance_ratio >= 1.5:
            st.warning(
                f"⚠️ Classes are imbalanced (ratio ≈ {imbalance_ratio:.2f}:1). "
                "This can hurt recall on the minority class — "
                "consider enabling class balancing below."
            )
        else:
            st.success("✅ Classes are reasonably balanced.")

    with cb2:
        fig_balance = px.bar(
            class_counts.rename("Count").rename_axis("Class").reset_index(),
            x="Class", y="Count", color="Class", text="Count"
        )
        st.plotly_chart(fig_balance, use_container_width=True)

    # =====================================================
    # IDENTIFY FEATURE TYPES
    # =====================================================

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    # =====================================================
    # PREPROCESSING
    # =====================================================

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    # =====================================================
    # TRAIN / TEST SPLIT
    # =====================================================

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    except ValueError:
        st.error(
            "The dataset is too small or the target classes "
            "are not suitable for stratified splitting."
        )
        st.stop()

    # =====================================================
    # RECALL-FOCUSED TRAINING OPTIONS
    # =====================================================

    st.subheader("🎯 Recall-Focused Training Options")

    opt1, opt2, opt3 = st.columns(3)

    with opt1:
        use_class_weight = st.checkbox(
            "Balance classes (class_weight='balanced')",
            value=True,
            help="Makes the model penalize mistakes on the minority "
                 "(disease) class more heavily. Applies to Logistic "
                 "Regression, Decision Tree, Random Forest and SVM."
        )

    with opt2:
        use_smote = st.checkbox(
            "Oversample minority class (SMOTE)",
            value=HAS_SMOTE,
            disabled=not HAS_SMOTE,
            help=None if HAS_SMOTE else
                 "Requires 'imbalanced-learn'. Install it with: "
                 "pip install imbalanced-learn"
        )
        if not HAS_SMOTE:
            st.caption("⚠️ imbalanced-learn not installed — SMOTE disabled.")

    with opt3:
        use_tuning = st.checkbox(
            "Hyperparameter tuning (GridSearchCV, scoring=recall)",
            value=True,
            help="Searches a small grid of hyperparameters per model, "
                 "optimizing directly for recall via cross-validation. "
                 "Slower, but generally gives noticeably higher recall."
        )

    n_splits = int(min(5, class_counts.min()))
    n_splits = max(n_splits, 2)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    st.caption(
        f"Cross-validation folds in use: {n_splits} "
        f"(auto-reduced to fit the smallest class if needed)."
    )

    # =====================================================
    # MODELS  (defaults; class_weight applied where supported)
    # =====================================================

    def make_base_models(balanced: bool):
        cw = "balanced" if balanced else None
        return {
            "Logistic Regression": LogisticRegression(max_iter=2000, class_weight=cw),
            "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight=cw),
            "Random Forest": RandomForestClassifier(
                n_estimators=300, random_state=42, class_weight=cw
            ),
            # No class_weight support -> relies on SMOTE / threshold tuning instead
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "SVM": SVC(probability=True, random_state=42, class_weight=cw),
            # No class_weight support -> relies on SMOTE / threshold tuning instead
            "KNN": KNeighborsClassifier(n_neighbors=7),
            "Naive Bayes": GaussianNB(),
        }

    models = make_base_models(use_class_weight)

    # Small, fast hyperparameter grids, all scored on recall.
    param_grids = {
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 1, 10],
        },
        "Decision Tree": {
            # "None" (unlimited depth) removed on purpose: it lets the
            # tree grow until it memorizes the training set -> overfitting.
            "model__max_depth": [3, 5, 7, 10],
            "model__min_samples_split": [5, 10, 20],
            "model__min_samples_leaf": [2, 5, 10],
        },
        "Random Forest": {
            "model__n_estimators": [100, 200, 300],
            # Same reasoning: capped depth instead of unlimited.
            "model__max_depth": [5, 10, 15],
            "model__min_samples_leaf": [2, 5, 10],
        },
        "Gradient Boosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__max_depth": [2, 3, 4],
        },
        "SVM": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["rbf", "linear"],
        },
        "KNN": {
            "model__n_neighbors": [3, 5, 7, 9, 11],
            "model__weights": ["uniform", "distance"],
        },
        "Naive Bayes": {
            "model__var_smoothing": np.logspace(0, -9, 5).tolist(),
        },
    }

    # =====================================================
    # MODEL SELECTION
    # =====================================================

    selected_models = st.multiselect(
        "Select models to compare",
        list(models.keys()),
        default=list(models.keys())
    )

    run = st.button("🚀 Run Comparison", type="primary")

    # =====================================================
    # CONFIG FINGERPRINT
    # (threshold is NOT included: it is applied post-hoc
    #  to stored probabilities, no retraining needed)
    # =====================================================

    df_hash = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()

    fingerprint = (
        f"{df_hash}|{target}|{test_size}|{tuple(sorted(selected_models))}|"
        f"{use_class_weight}|{use_smote}|{use_tuning}"
    )

    def build_pipeline(name):
        steps = [("preprocessing", preprocessor)]

        if use_smote and HAS_SMOTE:
            steps.append(("smote", SMOTE(random_state=42)))
            steps.append(("model", models[name]))
            return ImbPipeline(steps=steps)

        steps.append(("model", models[name]))
        return Pipeline(steps=steps)

    # =====================================================
    # TRAIN MODELS + SAVE TO SESSION STATE
    # =====================================================

    if run:

        if not selected_models:
            st.warning("Please select at least one model.")

        else:
            results_cv = {}
            roc_data = {}
            pr_data = {}
            trained_models = {}
            all_probs = {}
            best_params_by_model = {}

            with st.spinner("Training models (this may take a while with tuning enabled)..."):

                for name in selected_models:

                    pipeline = build_pipeline(name)

                    if use_tuning:
                        grid = GridSearchCV(
                            pipeline,
                            param_grids.get(name, {}),
                            scoring="recall",
                            cv=cv,
                            n_jobs=-1,
                            refit=True
                        )
                        grid.fit(X_train, y_train)
                        best_pipeline = grid.best_estimator_
                        best_params_by_model[name] = grid.best_params_
                    else:
                        pipeline.fit(X_train, y_train)
                        best_pipeline = pipeline
                        best_params_by_model[name] = None

                    # Cross-validated estimate on the training set
                    # (robustness check, independent of the single test split)
                    cv_scores = cross_validate(
                        best_pipeline, X_train, y_train, cv=cv,
                        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
                        n_jobs=-1
                    )
                    results_cv[name] = {
                        metric: (
                            float(np.mean(cv_scores[f"test_{metric}"])),
                            float(np.std(cv_scores[f"test_{metric}"]))
                        )
                        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]
                    }

                    # Probabilities on the held-out test set
                    if hasattr(best_pipeline, "predict_proba"):
                        probs = best_pipeline.predict_proba(X_test)[:, 1]
                    else:
                        probs = best_pipeline.decision_function(X_test)

                    all_probs[name] = probs

                    fpr, tpr, _ = roc_curve(y_test, probs)
                    auc = roc_auc_score(y_test, probs)
                    roc_data[name] = (fpr, tpr, auc)

                    prec_curve, rec_curve, pr_thresh = precision_recall_curve(y_test, probs)
                    ap = average_precision_score(y_test, probs)
                    pr_data[name] = (prec_curve, rec_curve, pr_thresh, ap)

                    trained_models[name] = best_pipeline

            st.session_state["results_cv"] = results_cv
            st.session_state["roc_data"] = roc_data
            st.session_state["pr_data"] = pr_data
            st.session_state["trained_models"] = trained_models
            st.session_state["all_probs"] = all_probs
            st.session_state["best_params_by_model"] = best_params_by_model
            st.session_state["y_test"] = y_test
            st.session_state["X_train"] = X_train
            st.session_state["numeric_features"] = numeric_features
            st.session_state["categorical_features"] = categorical_features
            st.session_state["fingerprint"] = fingerprint

    # =====================================================
    # RENDER RESULTS FROM SESSION STATE
    # =====================================================

    stale = (
        "fingerprint" in st.session_state
        and st.session_state["fingerprint"] != fingerprint
    )

    if stale:
        st.info(
            "⚠️ The setup changed since the last run "
            "(data, target, test size or training options). "
            "Click **Run Comparison** again to refresh the results."
        )

    elif "results_cv" in st.session_state:

        y_test = st.session_state["y_test"]
        all_probs = st.session_state["all_probs"]

        # -------------------------------------------------
        # THRESHOLD CONTROL
        # -------------------------------------------------

        st.subheader("🎚️ Decision Threshold")

        if "threshold" not in st.session_state:
            st.session_state["threshold"] = 0.50

        def suggest_recall_threshold():
            precisions, recalls, pr_thresh, _ = st.session_state["pr_data"][
                st.session_state["suggest_model_select"]
            ]
            # Drop the last (precision=1, recall=0) point which has no threshold
            p, r = precisions[:-1], recalls[:-1]
            with np.errstate(divide="ignore", invalid="ignore"):
                beta = 2  # weigh recall higher than precision
                f_beta = (1 + beta**2) * (p * r) / np.where(
                    (beta**2 * p + r) == 0, 1, (beta**2 * p + r)
                )
            best_idx = int(np.nanargmax(f_beta))
            st.session_state["threshold"] = float(pr_thresh[best_idx])

        th_col1, th_col2 = st.columns([3, 1])

        with th_col1:
            threshold = st.slider(
                "Classify as 'disease' (positive) when predicted probability ≥ threshold. "
                "Lower it to catch more positive cases (↑ recall, may ↓ precision).",
                min_value=0.05, max_value=0.95, step=0.01,
                key="threshold"
            )

        with th_col2:
            st.selectbox(
                "Suggest threshold for", list(all_probs.keys()),
                key="suggest_model_select"
            )
            st.button(
                "🎯 Maximize recall (F2)",
                on_click=suggest_recall_threshold
            )

        threshold = st.session_state["threshold"]

        # -------------------------------------------------
        # BUILD RESULTS TABLE FOR CURRENT THRESHOLD
        # -------------------------------------------------

        results = []
        all_predictions = {}

        for name, probs in all_probs.items():
            preds = (probs >= threshold).astype(int)
            all_predictions[name] = preds

            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, preds),
                "Precision": precision_score(y_test, preds, zero_division=0),
                "Recall": recall_score(y_test, preds, zero_division=0),
                "F1-Score": f1_score(y_test, preds, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, probs),
            })

        st.session_state["all_predictions"] = all_predictions

        sort_metric = st.selectbox(
            "Rank models by",
            ["Recall", "F1-Score", "ROC-AUC", "Precision", "Accuracy"],
            index=0,
            help="Recall is selected by default: in disease screening, "
                 "missing a real case (false negative) is usually costlier "
                 "than a false alarm."
        )

        results_df = pd.DataFrame(results).sort_values(
            sort_metric, ascending=False
        ).reset_index(drop=True)

        # -------------------------------------------------
        # OVERFIT GAP: CV Recall (training) vs Test Recall
        # -------------------------------------------------
        # A large gap between the cross-validated training score and
        # the held-out test score is the clearest sign of overfitting.

        results_cv_map = st.session_state["results_cv"]
        results_df["CV Recall"] = results_df["Model"].map(
            lambda m: results_cv_map[m]["recall"][0]
        )
        results_df["Overfit Gap"] = (
            results_df["CV Recall"] - results_df["Recall"]
        )

        overfitting_models = results_df[results_df["Overfit Gap"] > 0.10]
        if not overfitting_models.empty:
            st.warning(
                "⚠️ These models have a large gap (> 10%) between their "
                "cross-validation (training) performance and test-set "
                "performance, which indicates overfitting: "
                f"**{', '.join(overfitting_models['Model'])}**"
            )

        # -------------------------------------------------
        # BEST MODEL
        # -------------------------------------------------

        best_model = results_df.iloc[0]

        st.success(
            f"🏆 Best Model by {sort_metric}: **{best_model['Model']}** "
            f"— Recall: {best_model['Recall']:.2%} | "
            f"Precision: {best_model['Precision']:.2%} | "
            f"F1: {best_model['F1-Score']:.2%} | "
            f"ROC-AUC: {best_model['ROC-AUC']:.2%} "
            f"(threshold = {threshold:.2f})"
        )

        best_params = st.session_state["best_params_by_model"].get(best_model["Model"])
        if best_params:
            st.caption(f"Best hyperparameters found: {best_params}")

        # -------------------------------------------------
        # RESULTS TABLE (held-out test set, current threshold)
        # -------------------------------------------------

        st.subheader("📋 Results Table (held-out test set)")

        st.dataframe(
            results_df.style.format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1-Score": "{:.2%}",
                "ROC-AUC": "{:.2%}",
                "CV Recall": "{:.2%}",
                "Overfit Gap": "{:+.2%}"
            }).background_gradient(subset=["Recall"], cmap="Reds")
              .background_gradient(subset=["Overfit Gap"], cmap="Oranges"),
            use_container_width=True
        )

        # -------------------------------------------------
        # CROSS-VALIDATED METRICS (training set, threshold-independent)
        # -------------------------------------------------

        st.subheader("🔁 Cross-Validated Metrics (training set)")
        st.caption(
            f"{n_splits}-fold stratified cross-validation on the training data, "
            "at the model's default 0.5 internal threshold — a more robust, "
            "less lucky/unlucky estimate than a single test split."
        )

        cv_rows = []
        for name, metrics in st.session_state["results_cv"].items():
            cv_rows.append({
                "Model": name,
                "CV Accuracy": f"{metrics['accuracy'][0]:.2%} ± {metrics['accuracy'][1]:.2%}",
                "CV Precision": f"{metrics['precision'][0]:.2%} ± {metrics['precision'][1]:.2%}",
                "CV Recall": f"{metrics['recall'][0]:.2%} ± {metrics['recall'][1]:.2%}",
                "CV F1": f"{metrics['f1'][0]:.2%} ± {metrics['f1'][1]:.2%}",
                "CV ROC-AUC": f"{metrics['roc_auc'][0]:.2%} ± {metrics['roc_auc'][1]:.2%}",
            })

        st.dataframe(pd.DataFrame(cv_rows), use_container_width=True)

        # -------------------------------------------------
        # ACCURACY / METRIC COMPARISON
        # -------------------------------------------------

        st.subheader(f"📊 {sort_metric} Comparison (test set)")

        fig4 = px.bar(
            results_df, x="Model", y=sort_metric, color=sort_metric,
            text_auto=".2%"
        )
        fig4.update_layout(yaxis_title=sort_metric, xaxis_title="Model")
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("📈 Multi-Metric Comparison (test set)")

        metrics_melted = results_df.melt(
            id_vars="Model",
            value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            var_name="Metric", value_name="Score"
        )
        fig5 = px.bar(
            metrics_melted, x="Model", y="Score", color="Metric", barmode="group"
        )
        fig5.update_layout(yaxis_title="Score", xaxis_title="Model")
        st.plotly_chart(fig5, use_container_width=True)

        # -------------------------------------------------
        # ROC CURVES
        # -------------------------------------------------

        st.subheader("🎯 ROC Curves")

        fig6 = go.Figure()
        for name, (fpr, tpr, auc) in st.session_state["roc_data"].items():
            fig6.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={auc:.2f})"))

        fig6.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], line=dict(dash="dash"), name="Random"
        ))
        fig6.update_layout(
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig6, use_container_width=True)

        # -------------------------------------------------
        # PRECISION-RECALL CURVES  (key view for recall trade-offs)
        # -------------------------------------------------

        st.subheader("🧭 Precision-Recall Curves")
        st.caption(
            "Shows the trade-off you control with the threshold slider above. "
            "The marker shows each model's current operating point."
        )

        fig_pr = go.Figure()
        for name, (prec_curve, rec_curve, pr_thresh, ap) in st.session_state["pr_data"].items():
            fig_pr.add_trace(go.Scatter(
                x=rec_curve, y=prec_curve, name=f"{name} (AP={ap:.2f})", mode="lines"
            ))
            # Mark current operating point at the chosen threshold
            idx = int(np.argmin(np.abs(pr_thresh - threshold))) if len(pr_thresh) else None
            if idx is not None:
                fig_pr.add_trace(go.Scatter(
                    x=[rec_curve[idx]], y=[prec_curve[idx]],
                    mode="markers", marker=dict(size=10, symbol="x"),
                    name=f"{name} @ {threshold:.2f}", showlegend=False
                ))

        fig_pr.update_layout(
            xaxis_title="Recall", yaxis_title="Precision",
            xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig_pr, use_container_width=True)

        # -------------------------------------------------
        # CONFUSION MATRIX
        # -------------------------------------------------

        st.subheader("🧮 Confusion Matrix")

        available_models = list(st.session_state["all_predictions"].keys())

        cm_model = st.selectbox(
            "Select model", available_models, key="cm_model_select"
        )

        if cm_model not in st.session_state["all_predictions"]:
            cm_model = available_models[0]

        y_true_cm = st.session_state["y_test"]
        preds_cm = st.session_state["all_predictions"][cm_model]

        cm = confusion_matrix(y_true_cm, preds_cm)

        fig7 = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            labels={"x": "Predicted", "y": "Actual"}
        )
        fig7.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig7, use_container_width=True)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            st.caption(
                f"False Negatives (missed disease cases): **{fn}** — "
                f"lower the threshold to reduce this at the cost of more "
                f"False Positives ({fp})."
            )

        # -------------------------------------------------
        # FEATURE IMPORTANCE (tree-based models)
        # -------------------------------------------------

        tree_models = {"Decision Tree", "Random Forest", "Gradient Boosting"}

        if cm_model in tree_models:
            st.subheader(f"🌳 Feature Importance — {cm_model}")

            fitted_pipeline = st.session_state["trained_models"][cm_model]
            fitted_preprocessor = fitted_pipeline.named_steps["preprocessing"]
            fitted_model = fitted_pipeline.named_steps["model"]

            num_feats = st.session_state["numeric_features"]
            cat_feats = st.session_state["categorical_features"]

            feature_names = list(num_feats)
            if cat_feats:
                encoder = fitted_preprocessor.named_transformers_["categorical"].named_steps["encoder"]
                feature_names += list(encoder.get_feature_names_out(cat_feats))

            importances = fitted_model.feature_importances_

            if len(importances) == len(feature_names):
                imp_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances
                }).sort_values("Importance", ascending=False).head(15)

                fig8 = px.bar(
                    imp_df, x="Importance", y="Feature",
                    orientation="h", color="Importance"
                )
                fig8.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig8, use_container_width=True)
            else:
                st.info("Feature importance shape mismatch — skipping plot.")
