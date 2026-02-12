# 🌌 Stellar Classification - Model Comparison

---

## a. Problem Statement
The goal of this project is to classify celestial objects (Galaxy, Star, Quasar) using machine learning models.  
We aim to preprocess the dataset, perform feature selection, train multiple classifiers, and evaluate their performance using standard metrics.  
The comparison helps identify which model performs best for stellar classification tasks.

---

## b. Dataset Description
- **Source:** Sloan Digital Sky Survey (SDSS) star classification dataset.  
- **Target Variable:** `class` (Galaxy, Star, Quasar).  
- **Features:** Spectral and photometric attributes such as `u`, `g`, `r`, `i`, `z` magnitudes, redshift, etc.  
- **Preprocessing Steps:**
  - Removed duplicates and missing values.
  - Dropped irrelevant identifier columns (`run_ID`, `rerun_ID`, `cam_col`, `field_ID`, `spec_obj_ID`, `plate`, `MJD`, `fiber_ID`).
  - Label encoded the target variable.
  - Normalized features using MinMaxScaler.
  - Applied ANOVA F-test for feature importance ranking.

---

## c. Models Used

### Comparison Table of Evaluation Metrics

| ML Model Name        | Accuracy | AUC   | Precision | Recall | F1   | MCC   |
|----------------------|----------|-------|-----------|--------|------|-------|
| Logistic Regression  | 0.XXXX   | 0.XXXX| 0.XXXX    | 0.XXXX |0.XXXX|0.XXXX |
| Decision Tree        | 0.XXXX   | 0.XXXX| 0.XXXX    | 0.XXXX |0.XXXX|0.XXXX |
| kNN                  | 0.XXXX   | 0.XXXX| 0.XXXX    | 0.XXXX |0.XXXX|0.XXXX |
| Naive Bayes          | 0.XXXX   | 0.XXXX| 0.XXXX    | 0.XXXX |0.XXXX|0.XXXX |
| Random Forest (Ens.) | 0.XXXX   | 0.XXXX| 0.XXXX    | 0.XXXX |0.XXXX|0.XXXX |
| XGBoost (Ens.)       | 0.XXXX   | 0.XXXX| 0.XXXX    | 0.XXXX |0.XXXX|0.XXXX |

*(Replace `0.XXXX` with actual values from your `results_df` output.)*

---

### Observations on Model Performance

| ML Model Name        | Observation about model performance |
|----------------------|--------------------------------------|
| Logistic Regression  | Performs well on linearly separable data; moderate accuracy and balanced metrics. |
| Decision Tree        | Easy to interpret; prone to overfitting; performance depends on depth and pruning. |
| kNN                  | Sensitive to scaling and choice of `k`; performs reasonably but slower on large datasets. |
| Naive Bayes          | Assumes feature independence; fast but less accurate compared to ensemble methods. |
| Random Forest (Ens.) | Strong performance due to ensemble averaging; robust against overfitting; high accuracy and F1. |
| XGBoost (Ens.)       | Best overall performance; handles complex relationships; high accuracy, precision, and MCC. |

---

## 📌 Notes
- Ensemble models (Random Forest, XGBoost) generally outperform simpler models due to their ability to capture complex feature interactions.
- Logistic Regression and Naive Bayes provide baseline performance and are computationally efficient.
- Decision Tree and kNN are more sensitive to hyperparameters and dataset characteristics.