import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
BINARY_COLS = [
    "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling", "MultipleLines", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies",
]
MULTI_COLS  = ["InternetService", "Contract", "PaymentMethod"]
NUM_FEATS   = ["tenure", "MonthlyCharges", "TotalCharges"]


# ═════════════════════════════════════════════════════════════
# STEP 1 — DATA LOADING & INSPECTION
# ═════════════════════════════════════════════════════════════
def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the Telco Customer Churn CSV and return a raw DataFrame.

    Returns
    -------
    df : pd.DataFrame  (raw, untouched)
    """
    df = pd.read_csv(filepath)
    return df


def inspect_data(df: pd.DataFrame) -> dict:
    """
    Return a summary dict for quick inspection.

    Keys
    ----
    shape, dtypes, head, missing, churn_dist, churn_rate_pct
    """
    return {
        "shape":         df.shape,
        "dtypes":        df.dtypes.to_dict(),
        "head":          df.head(),
        "missing":       df.isnull().sum().to_dict(),
        "churn_dist":    df["Churn"].value_counts().to_dict()
                         if "Churn" in df.columns else {},
        "churn_rate_pct": round(
            df["Churn"].value_counts(normalize=True).get("Yes", 0) * 100, 2
        ) if "Churn" in df.columns else None,
    }


# ═════════════════════════════════════════════════════════════
# STEP 2 — DATA PREPROCESSING & CLEANING
# ═════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame.

    Steps
    -----
    - Drop customerID (non-predictive identifier)
    - Convert TotalCharges from object → float (blank strings → NaN)
    - Drop rows where TotalCharges is NaN (~11 rows)

    Returns a cleaned copy.
    """
    df = df.copy()

    # Drop identifier
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    dropped = df["TotalCharges"].isnull().sum()
    df.dropna(subset=["TotalCharges"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"  [Preprocess] Dropped {dropped} rows with blank TotalCharges.")
    print(f"  [Preprocess] Clean dataset: {df.shape[0]} rows x {df.shape[1]} cols.")
    return df


# ═════════════════════════════════════════════════════════════
# STEP 3 — EXPLORATORY DATA ANALYSIS (EDA)
# ═════════════════════════════════════════════════════════════
def eda_summary(df: pd.DataFrame) -> dict:
    """
    Compute EDA statistics without plotting (plotting left to caller / Streamlit).

    Returns
    -------
    dict with keys:
        describe_num   — describe() on numerical cols
        churn_by       — dict[col] → churn-rate per category
        correlations   — Pearson corr of num cols with binary Churn
        insights       — list of human-readable insight strings
    """
    churn_binary = df["Churn"].map({"Yes": 1, "No": 0})

    # Numerical describe
    describe_num = df[NUM_FEATS].describe()

    # Churn rate per categorical value
    cat_cols = ["Contract", "InternetService", "PaymentMethod",
                "TechSupport", "OnlineSecurity", "SeniorCitizen"]
    churn_by = {}
    for col in cat_cols:
        if col in df.columns:
            ct = df.groupby(col)["Churn"].apply(
                lambda s: round((s == "Yes").mean() * 100, 1)
            ).to_dict()
            churn_by[col] = ct

    # Pearson correlation with Churn
    correlations = df[NUM_FEATS].corrwith(churn_binary).sort_values(ascending=False)

    # Key insights
    insights = [
        "Month-to-month contract customers churn at the highest rate.",
        "Fiber optic internet customers show significantly higher churn.",
        "Short-tenure (new) customers are far more likely to churn.",
        "Customers without TechSupport or OnlineSecurity churn more.",
        "TotalCharges is negatively correlated with churn (loyal = high spend).",
        "MonthlyCharges is positively correlated with churn.",
    ]

    return {
        "describe_num": describe_num,
        "churn_by":     churn_by,
        "correlations": correlations,
        "insights":     insights,
    }


# ═════════════════════════════════════════════════════════════
# STEP 4 — FEATURE ENGINEERING & ENCODING
# ═════════════════════════════════════════════════════════════
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical and binary features.

    Steps
    -----
    - Map target Churn: Yes→1, No→0
    - Binary-encode Yes/No columns (and gender)
    - One-hot encode multi-class columns (InternetService, Contract, PaymentMethod)
    - Cast bool columns to int

    Returns encoded DataFrame.
    """
    df = df.copy()

    # Target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Binary columns
    for col in BINARY_COLS:
        if col not in df.columns:
            continue
        if col == "gender":
            df[col] = df[col].map({"Female": 1, "Male": 0})
        else:
            df[col] = df[col].apply(lambda x: 1 if x == "Yes" else 0)

    # One-hot encode
    df = pd.get_dummies(df, columns=MULTI_COLS, drop_first=False)

    # Bool → int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"  [Encode] Encoded shape: {df.shape[0]} rows x {df.shape[1]} cols.")
    return df


# ═════════════════════════════════════════════════════════════
# STEP 5 — FEATURE SELECTION (Correlation)
# ═════════════════════════════════════════════════════════════
def select_features(df_encoded: pd.DataFrame,
                    top_n: int = None,
                    threshold: float = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Split X / y and compute feature correlations with target.

    Parameters
    ----------
    df_encoded : encoded DataFrame (output of encode_features)
    top_n      : keep only top N most correlated features (optional)
    threshold  : keep features with |corr| >= threshold (optional)

    Returns
    -------
    X           : feature DataFrame (all features, or filtered)
    y           : target Series
    corr_series : Pearson correlations of each feature with Churn
    """
    X = df_encoded.drop("Churn", axis=1)
    y = df_encoded["Churn"]

    corr_series = X.corrwith(y).sort_values(key=abs, ascending=False)

    if top_n is not None:
        selected = corr_series.head(top_n).index
        X = X[selected]
        print(f"  [Select] Kept top {top_n} features.")
    elif threshold is not None:
        selected = corr_series[corr_series.abs() >= threshold].index
        X = X[selected]
        print(f"  [Select] Kept {len(X.columns)} features with |corr| >= {threshold}.")
    else:
        print(f"  [Select] Using all {len(X.columns)} features.")

    return X, y, corr_series


# ═════════════════════════════════════════════════════════════
# STEP 6 — TRAIN / TEST SPLIT + SCALING
# ═════════════════════════════════════════════════════════════
def split_and_scale(X: pd.DataFrame, y: pd.Series,
                    test_size: float = 0.2,
                    random_state: int = 42
                    ) -> tuple:
    """
    Stratified train/test split + StandardScaler on numerical columns.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    num_in_X = [f for f in NUM_FEATS if f in X_train.columns]
    X_train[num_in_X] = scaler.fit_transform(X_train[num_in_X])
    X_test[num_in_X]  = scaler.transform(X_test[num_in_X])

    print(f"  [Split] Train: {X_train.shape[0]} | Test: {X_test.shape[0]} samples.")
    return X_train, X_test, y_train, y_test, scaler


# ═════════════════════════════════════════════════════════════
# STEP 7 — LOGISTIC REGRESSION
# ═════════════════════════════════════════════════════════════
def train_logistic_regression(X_train, y_train,
                               max_iter: int = 1000,
                               random_state: int = 42
                               ) -> LogisticRegression:
    """Train and return a Logistic Regression model."""
    model = LogisticRegression(max_iter=max_iter, random_state=random_state, solver="lbfgs")
    model.fit(X_train, y_train)
    print("  [Model] Logistic Regression trained.")
    return model


# ═════════════════════════════════════════════════════════════
# STEP 8 — DECISION TREE CLASSIFIER
# ═════════════════════════════════════════════════════════════
def train_decision_tree(X_train, y_train,
                         max_depth: int = 6,
                         min_samples_split: int = 10,
                         random_state: int = 42
                         ) -> DecisionTreeClassifier:
    """Train and return a Decision Tree model."""
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    print("  [Model] Decision Tree trained.")
    return model


# ═════════════════════════════════════════════════════════════
# STEP 9 — ACCURACY & ROC-AUC EVALUATION
# ═════════════════════════════════════════════════════════════
def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Evaluate a trained classifier.

    Returns
    -------
    dict with keys:
        name, accuracy, roc_auc, report_dict, report_str,
        predictions, probabilities
    """
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, preds)
    auc  = roc_auc_score(y_test, probs)
    rep  = classification_report(y_test, preds,
                                  target_names=["No Churn", "Churn"],
                                  output_dict=True)
    rep_str = classification_report(y_test, preds,
                                     target_names=["No Churn", "Churn"])

    print(f"  [Eval] {model_name} | Accuracy: {acc*100:.2f}% | ROC-AUC: {auc:.4f}")
    return {
        "name":          model_name,
        "accuracy":      acc,
        "roc_auc":       auc,
        "report_dict":   rep,
        "report_str":    rep_str,
        "predictions":   preds,
        "probabilities": probs,
    }


# ═════════════════════════════════════════════════════════════
# STEP 10 — CONFUSION MATRIX & ROC CURVE DATA
# ═════════════════════════════════════════════════════════════
def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Return sklearn confusion matrix array."""
    return confusion_matrix(y_true, y_pred)


def get_roc_curve_data(y_true, y_prob) -> tuple:
    """
    Compute ROC curve points.

    Returns
    -------
    fpr, tpr, thresholds (all np.ndarray)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return fpr, tpr, thresholds


# ═════════════════════════════════════════════════════════════
# STEP 11 — FEATURE IMPORTANCE ANALYSIS
# ═════════════════════════════════════════════════════════════
def get_feature_importances(model, feature_names,
                             top_n: int = 15) -> pd.DataFrame:
    """
    Extract feature importances (Decision Tree) or
    absolute coefficients (Logistic Regression).

    Returns
    -------
    pd.DataFrame with columns [Feature, Importance/Coefficient]
    sorted descending by absolute value, limited to top_n.
    """
    if isinstance(model, DecisionTreeClassifier):
        imp = pd.Series(model.feature_importances_, index=feature_names)
        df_imp = imp.sort_values(ascending=False).head(top_n).reset_index()
        df_imp.columns = ["Feature", "Importance"]
    elif isinstance(model, LogisticRegression):
        coef = pd.Series(model.coef_[0], index=feature_names)
        df_imp = coef.reindex(
            coef.abs().sort_values(ascending=False).index
        ).head(top_n).reset_index()
        df_imp.columns = ["Feature", "Coefficient"]
    else:
        raise ValueError("Model must be LogisticRegression or DecisionTreeClassifier.")

    return df_imp


# ═════════════════════════════════════════════════════════════
# PREDICT — Single Customer
# ═════════════════════════════════════════════════════════════
def predict_single(model, scaler, feature_columns: list, row_dict: dict) -> dict:
    """
    Predict churn for a single customer.

    Parameters
    ----------
    model          : trained sklearn model
    scaler         : fitted StandardScaler
    feature_columns: list of column names the model was trained on
    row_dict       : dict of {feature_name: value} for one customer

    Returns
    -------
    dict: prediction (0/1), probability (float), label (str)
    """
    df_row = pd.DataFrame([row_dict])
    for col in feature_columns:
        if col not in df_row.columns:
            df_row[col] = 0
    df_row = df_row[feature_columns].copy()

    num_in = [f for f in NUM_FEATS if f in df_row.columns]
    df_row[num_in] = scaler.transform(df_row[num_in])

    pred  = model.predict(df_row)[0]
    prob  = model.predict_proba(df_row)[0][1]
    label = "Churn" if pred == 1 else "No Churn"
    return {"prediction": int(pred), "probability": float(prob), "label": label}


# ═════════════════════════════════════════════════════════════
# PIPELINE CLASS — runs everything in sequence
# ═════════════════════════════════════════════════════════════
class ChurnPipeline:
    """
    End-to-end Customer Churn Prediction pipeline.

    Quick start
    -----------
        pipe = ChurnPipeline("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        pipe.run_all()
        print(pipe.metrics["lr"]["accuracy"])
    """

    def __init__(self, filepath: str):
        self.filepath = filepath

        # populated by run_all()
        self.df_raw     = None
        self.df_clean   = None
        self.df_encoded = None
        self.X          = None
        self.y          = None
        self.corr       = None
        self.X_train    = None
        self.X_test     = None
        self.y_train    = None
        self.y_test     = None
        self.scaler     = None
        self.lr_model   = None
        self.dt_model   = None
        self.metrics    = {}
        self.eda        = {}

    # ── individual steps ──────────────────────────────────────
    def step1_load(self):
        print("\n[STEP 1] Loading data...")
        self.df_raw = load_data(self.filepath)
        info = inspect_data(self.df_raw)
        print(f"  Shape     : {info['shape']}")
        print(f"  Churn rate: {info['churn_rate_pct']}%")

    def step2_preprocess(self):
        print("\n[STEP 2] Preprocessing & Cleaning...")
        self.df_clean = preprocess(self.df_raw)

    def step3_eda(self):
        print("\n[STEP 3] EDA...")
        self.eda = eda_summary(self.df_clean)
        print("  Key Insights:")
        for i in self.eda["insights"]:
            print(f"    - {i}")

    def step4_encode(self):
        print("\n[STEP 4] Feature Engineering & Encoding...")
        self.df_encoded = encode_features(self.df_clean)

    def step5_select(self, top_n=None, threshold=None):
        print("\n[STEP 5] Feature Selection...")
        self.X, self.y, self.corr = select_features(
            self.df_encoded, top_n=top_n, threshold=threshold
        )
        print(f"  Top correlated features:\n{self.corr.head(5).to_string()}")

    def step6_split(self):
        print("\n[STEP 6] Train/Test Split + Scaling...")
        self.X_train, self.X_test, self.y_train, self.y_test, self.scaler = \
            split_and_scale(self.X, self.y)

    def step7_logistic(self):
        print("\n[STEP 7] Training Logistic Regression...")
        self.lr_model = train_logistic_regression(self.X_train, self.y_train)

    def step8_tree(self):
        print("\n[STEP 8] Training Decision Tree...")
        self.dt_model = train_decision_tree(self.X_train, self.y_train)

    def step9_evaluate(self):
        print("\n[STEP 9] Evaluating Models...")
        self.metrics["lr"] = evaluate_model(self.lr_model, self.X_test,
                                             self.y_test, "Logistic Regression")
        self.metrics["dt"] = evaluate_model(self.dt_model, self.X_test,
                                             self.y_test, "Decision Tree")

    def step10_curves(self):
        print("\n[STEP 10] Computing CM & ROC Curve data...")
        for key, model in [("lr", self.lr_model), ("dt", self.dt_model)]:
            preds = self.metrics[key]["predictions"]
            probs = self.metrics[key]["probabilities"]
            self.metrics[key]["cm"]  = get_confusion_matrix(self.y_test, preds)
            fpr, tpr, _ = get_roc_curve_data(self.y_test, probs)
            self.metrics[key]["fpr"] = fpr
            self.metrics[key]["tpr"] = tpr
        print("  CM and ROC data stored in pipeline.metrics[<model>].")

    def step11_importance(self):
        print("\n[STEP 11] Feature Importance Analysis...")
        self.metrics["lr"]["importances"] = get_feature_importances(
            self.lr_model, self.X.columns)
        self.metrics["dt"]["importances"] = get_feature_importances(
            self.dt_model, self.X.columns)
        print("  Top 5 LR coefficients:")
        print(self.metrics["lr"]["importances"].head().to_string(index=False))

    def run_all(self, top_n=None, threshold=None):
        """Run the complete pipeline from loading to feature importance."""
        self.step1_load()
        self.step2_preprocess()
        self.step3_eda()
        self.step4_encode()
        self.step5_select(top_n=top_n, threshold=threshold)
        self.step6_split()
        self.step7_logistic()
        self.step8_tree()
        self.step9_evaluate()
        self.step10_curves()
        self.step11_importance()
        self._print_summary()

    def _print_summary(self):
        lr = self.metrics["lr"]
        dt = self.metrics["dt"]
        best = "Logistic Regression" if lr["roc_auc"] >= dt["roc_auc"] else "Decision Tree"
        print("\n" + "=" * 55)
        print("  FINAL SUMMARY")
        print("=" * 55)
        print(f"  Logistic Regression — Acc: {lr['accuracy']*100:.2f}%  AUC: {lr['roc_auc']:.4f}")
        print(f"  Decision Tree       — Acc: {dt['accuracy']*100:.2f}%  AUC: {dt['roc_auc']:.4f}")
        print(f"  Best Model : {best}")
        print("=" * 55)

    def predict(self, row_dict: dict, model: str = "lr") -> dict:
        """Predict for a single customer. model='lr' or 'dt'."""
        m = self.lr_model if model == "lr" else self.dt_model
        return predict_single(m, self.scaler, list(self.X.columns), row_dict)


# ─────────────────────────────────────────────────────────────
# Run directly for a quick test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipe = ChurnPipeline("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    pipe.run_all()
