# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
    recall_score, f1_score, matthews_corrcoef,
    classification_report, confusion_matrix
)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Star Classification App")

# File uploader
uploaded_file = st.file_uploader("Upload your test dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Basic cleaning
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # Drop irrelevant identifiers if present
    id_columns = ["run_ID", "rerun_ID", "cam_col", "field_ID",
                  "spec_obj_ID", "plate", "MJD", "fiber_ID"]
    df = df.drop(columns=[col for col in id_columns if col in df.columns], errors="ignore")

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

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model dictionary
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
    }

    # Dropdown for model selection
    selected_model_name = st.selectbox("Select a model", list(models.keys()))
    model = models[selected_model_name]

    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    try:
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    except:
        auc = None

    st.subheader(f"Evaluation Metrics - {selected_model_name}")
    st.write(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    if auc is not None:
        st.write(f"AUC      : {auc:.4f}")
    else:
        st.write("AUC      : Not available")
    st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    st.write(f"Recall   : {recall_score(y_test, y_pred, average='weighted'):.4f}")
    st.write(f"F1 Score : {f1_score(y_test, y_pred, average='weighted'):.4f}")
    st.write(f"MCC      : {matthews_corrcoef(y_test, y_pred):.4f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_enc.classes_,
                yticklabels=label_enc.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=label_enc.classes_)
    st.text(report)
