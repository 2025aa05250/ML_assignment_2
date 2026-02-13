import streamlit as st
import pandas as pd
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

# ================================
# Streamlit App
# ================================
st.title("🌌 Stellar Classification Model Comparison")

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Clean data
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    id_columns = ["run_ID", "rerun_ID", "cam_col", "field_ID",
                  "spec_obj_ID", "plate", "MJD", "fiber_ID"]
    df.drop(columns=[col for col in id_columns if col in df.columns], inplace=True)

    # Encode target labels
    label_enc = LabelEncoder()
    df["class"] = label_enc.fit_transform(df["class"])

    X = df.drop(columns=["class"])
    y = df["class"]

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Feature selection
    selector = SelectKBest(score_func=f_classif, k="all")
    X_selected = selector.fit_transform(X_scaled, y)

    feature_scores = pd.DataFrame({
        "Feature": X.columns,
        "Score": selector.scores_
    }).sort_values(by="Score", ascending=False)

    st.subheader("🔎 Feature Importance Ranking")
    st.dataframe(feature_scores)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        try:
            y_prob = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
        except:
            auc = None

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": auc,
            "Precision": precision_score(y_test, y_pred, average="weighted"),
            "Recall": recall_score(y_test, y_pred, average="weighted"),
            "F1": f1_score(y_test, y_pred, average="weighted"),
            "MCC": matthews_corrcoef(y_test, y_pred)
        }
        results.append(metrics)

    results_df = pd.DataFrame(results)

    st.subheader("📊 Model Performance Comparison")
    st.dataframe(results_df)

    # Show metrics per model
    st.subheader("📌 Detailed Metrics")
    for idx, row in results_df.iterrows():
        st.markdown(f"### {row['Model']}")
        st.write(f"**Accuracy:** {row['Accuracy']:.4f}")
        if row['AUC'] is not None:
            st.write(f"**AUC:** {row['AUC']:.4f}")
        else:
            st.write("**AUC:** Not available")
        st.write(f"**Precision:** {row['Precision']:.4f}")
        st.write(f"**Recall:** {row['Recall']:.4f}")
        st.write(f"**F1 Score:** {row['F1']:.4f}")
        st.write(f"**MCC:** {row['MCC']:.4f}")
else:

    st.info("Please upload a dataset CSV file to proceed.")
