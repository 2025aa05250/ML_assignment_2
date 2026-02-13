import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Heart Disease Prediction - ML Assignment")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.write(data.head())

    # Dropdown for model selection
    model_choice = st.selectbox(
        "Select Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        ]
    )

    # Map model names to .pkl files
    model_map = {
        "Logistic Regression": "model/logistic_regression.pkl",
        "Decision Tree": "model/decision_tree.pkl",
        "KNN": "model/knn.pkl",
        "Naive Bayes": "model/naive_bayes.pkl",
        "Random Forest": "model/random_forest.pkl",
        "XGBoost": "model/xgboost.pkl"
    }

    # Define categorical columns
    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

    # One-hot encode categorical variables
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Load expected feature names from training
    feature_names = joblib.load("model/features.pkl")

    # Align uploaded data with training features
    data_encoded = data_encoded.reindex(columns=feature_names, fill_value=0)

    # Scale numerical features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_encoded)

    # Load model
    model_path = model_map[model_choice]
    model = joblib.load(model_path)

    # Predictions
    predictions = model.predict(data_scaled)
    st.write("Predictions:", predictions)

    # If target column exists, show evaluation metrics
    if "target" in data.columns:
        y_true = data["target"]
        y_pred = predictions

        st.subheader("Evaluation Metrics")
        st.write("Accuracy:", accuracy_score(y_true, y_pred))
        st.write("Precision:", precision_score(y_true, y_pred))
        st.write("Recall:", recall_score(y_true, y_pred))
        st.write("F1 Score:", f1_score(y_true, y_pred))
        st.write("MCC:", matthews_corrcoef(y_true, y_pred))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
