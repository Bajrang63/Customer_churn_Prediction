# 📡 Telco Customer Churn Prediction & Interactive Dashboard

A professional-grade **End-to-End Machine Learning Pipeline and Business Intelligence (BI)** application designed to predict and analyze customer churn. 

This repository leverages the popular IBM Watson **Telco Customer Churn dataset** to perform robust data cleaning, feature engineering, exploratory data analysis (EDA), and machine learning modeling. We build and compare two primary models: **Logistic Regression** and a **Decision Tree Classifier**, deploying the final predictive engine into a stunning, dark-themed **Streamlit Web Application** for real-time customer churn scoring and actionable retention suggestions.

---

## 🚀 Key Features

* **Modular 11-Step Data Pipeline (`data_processing.py`)**: A highly structured, cleanly abstracted engineering framework representing all phases of a production ML lifecycle:
  1. **Data Loading & Inspection**: Standard ingestion of IBM dataset.
  2. **Data Preprocessing & Cleaning**: Graceful handling of missing values (blank `TotalCharges`) and dropping high-cardinality non-predictive identifiers.
  3. **Exploratory Data Analysis (EDA)**: Statistical summarization of numerical/categorical features against the target variable.
  4. **Feature Engineering & Encoding**: Binary encoding of yes/no flags, gender mapper, and one-hot encoding for multi-class variables.
  5. **Feature Selection**: Pearson correlation analysis to identify key drivers of churn.
  6. **Data Partitioning & Scaling**: Stratified train-test splitting to protect class ratios, followed by feature scaling (`StandardScaler`).
  7. **Logistic Regression Classifier**: Training an interpretable linear model with LBFGS.
  8. **Decision Tree Classifier**: Training a high-capacity non-linear estimator with depth constraints to prevent overfitting.
  9. **Accuracy & ROC-AUC Evaluation**: Measuring model precision, recall, F1-scores, and overall ROC-AUC curves.
  10. **Confusion Matrix & Curves**: Capturing false positives/negatives for cost-sensitive analysis.
  11. **Feature Importance & Coefficient Extraction**: Understanding the business signals that trigger customer churn.
* **Granular Script Steps (`step1_*.py` to `step11_*.py`)**: Eleven clean standalone scripts that mirror each phase of the pipeline, allowing researchers to run, study, and debug steps independently.
* **Unified Pipeline Entry-point (`customer_churn_prediction.py`)**: Executes the entire 11-step pipeline sequentially, generates all evaluation charts, and outputs clear metric comparisons.
* **Premium Interactive Streamlit App (`app.py`)**: A gorgeous, dark-themed dashboard featuring:
  * **🏠 Executive Overview KPI Grid**: Displays Total Customers, overall Churn Rate, Average Tenure, and Average Monthly Bill with Plotly performance summaries.
  * **📊 Interactive EDA Studio**: User-configurable histograms, bar charts, and heatmaps to inspect distributions of `tenure`, `MonthlyCharges`, `TotalCharges`, and other categories.
  * **🤖 Model Evaluation Console**: Comprehensive diagnostic visualizations (Confusion Matrices, interactive ROC curves, and side-by-side model comparison bars).
  * **🔮 Real-Time Customer Predictor**: High-fidelity sidebar inputs that update a gauge chart instantaneously, triggering tailored, actionable customer retention tips based on the individual's profile features.

---

## 📊 Business & Model Insights

Through our exploratory data analysis and model training, we uncovered key behavioral indicators of customer churn:

1. **Contract Structure**: Month-to-month contracts are highly prone to churn. Migrating these customers to one- or two-year contracts is the single most effective retention tactic.
2. **Internet Service & Tech Support**: Customers subscribed to Fiber Optic internet without Online Security or Tech Support add-ons churn at significantly elevated rates. Offering bundled support trials dramatically increases customer loyalty.
3. **Tenure Dynamics**: Newly acquired customers (low tenure months) are highly volatile. Customer risk decreases exponentially as tenure grows.
4. **Model Performance**:
   * **Logistic Regression** achieves a balanced accuracy of **~80.4%** and a **ROC-AUC of ~0.84**, offering highly interpretable coefficients.
   * **Decision Tree Classifier** provides comparable accuracy, showing high utility in classifying non-linear boundaries.

---

## 🛠️ Step-by-Step Installation & Setup

Set up your local development environment on Windows with these commands:

### 1. Install Dependencies
Run the standard package installer to fetch necessary data science, modeling, and visualization libraries:
```powershell
pip install -r requirements.txt
```

### 2. Execute the Main Machine Learning Pipeline
Run the entry point script to train both models, execute the evaluation pipeline, and generate all output plots:
```powershell
python customer_churn_prediction.py
```
This runs the end-to-end pipeline and saves a full suite of analytical plots in your workspace:
* `eda_churn_distribution.png`
* `eda_numerical_features.png`
* `eda_categorical_features.png`
* `feature_correlation.png`
* `confusion_matrices.png`
* `roc_curves.png`
* `feature_importances.png`
* `lr_coefficients.png`
* `model_comparison.png`

### 3. Launch the Interactive Dashboard
Spin up the Streamlit server to open the professional BI dashboard in your web browser:
```powershell
streamlit run app.py
```
The app will automatically serve on `http://localhost:8501`.

---

## 📁 File Structure

```
d:\DSProject1\
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # IBM Watson Telco Customer Churn dataset
├── data_processing.py                    # Reusable ML pipeline & class abstraction
├── customer_churn_prediction.py          # Sequential pipeline script runner
├── app.py                                # Sleek, interactive Streamlit application
│
├── step1_load_inspect.py                 # Ingestion & basic inspection script
├── step2_preprocess.py                  # Cleaning & missing value handling
├── step3_eda.py                         # Data visualization and stats
├── step4_encode.py                      # Categorical & binary encoding script
├── step5_feature_selection.py           # Pearson correlation selection script
├── step6_split_scale.py                 # Stratified partitioning & standard scaling
├── step7_logistic_regression.py         # Logistic Regression model training
├── step8_decision_tree.py               # Decision Tree model training
├── step9_evaluation.py                  # Accuracy, Precision, Recall, and F1 calculations
├── step10_cm_roc.py                     # ROC-AUC curves & Confusion Matrix generation
├── step11_feature_importance.py         # Decision Tree importances & LR coefficients
│
└── linkedin_post.txt                     # Draft content for project sharing
```
