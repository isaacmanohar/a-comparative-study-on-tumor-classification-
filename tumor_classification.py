"""
============================================================
  TUMOR CLASSIFICATION USING COMPARATIVE ML MODELS
  Dataset : Breast Cancer Wisconsin (sklearn built-in)
  Author  : ML Project
  Goal    : Compare multiple ML models to classify tumors
            as Benign or Malignant
============================================================
"""

# ---------------------------------------------------------
# 0. IMPORTS
# ---------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import os

# Output folder for saved plots
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. LOAD & PREPARE DATASET
# ---------------------------------------------------------
def load_and_prepare_data():
    """Load Breast Cancer dataset, return DataFrame, X, y."""
    print("\n" + "=" * 60)
    print("  STEP 1 : LOADING DATASET")
    print("=" * 60)

    raw = load_breast_cancer()
    df = pd.DataFrame(data=raw.data, columns=raw.feature_names)
    df["target"] = raw.target          # 0 = Malignant, 1 = Benign

    print(f"  Dataset Shape : {df.shape}")
    print(f"  Features      : {df.shape[1] - 1}")
    print(f"  Target Classes: {raw.target_names.tolist()}")
    print(f"\n  Class Distribution:")
    counts = df["target"].value_counts()
    print(f"    Benign    (1) - {counts[1]}")
    print(f"    Malignant (0) - {counts[0]}")

    # Check missing values
    missing = df.isnull().sum().sum()
    print(f"\n  Missing Values : {missing}  {'[OK] None found' if missing == 0 else '[X] Handling required'}")

    X = df.drop("target", axis=1)
    y = df["target"]
    return df, X, y, raw.feature_names, raw.target_names


# ---------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------
def preprocess_data(X, y):
    """Scale features and perform stratified 80-20 split."""
    print("\n" + "=" * 60)
    print("  STEP 2 : PREPROCESSING")
    print("=" * 60)

    # Stratified split (maintains class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"  Train size    : {X_train_sc.shape[0]} samples")
    print(f"  Test  size    : {X_test_sc.shape[0]} samples")
    print(f"  Train Benign / Malignant : "
          f"{(y_train == 1).sum()} / {(y_train == 0).sum()}")
    print(f"  Test  Benign / Malignant : "
          f"{(y_test == 1).sum()} / {(y_test == 0).sum()}")
    print("  [OK] StandardScaler applied")
    print("  [OK] Stratified split complete")

    return X_train_sc, X_test_sc, y_train, y_test, scaler, X_train, X_test


# ---------------------------------------------------------
# 3. DEFINE MODELS (default hyperparameters)
# ---------------------------------------------------------
def get_base_models():
    """Return a dict of base (un-tuned) models."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
        "SVM":                 SVC(probability=True, random_state=42),
        "KNN":                 KNeighborsClassifier(),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(random_state=42),
    }


# ---------------------------------------------------------
# 4. HYPERPARAMETER TUNING
# ---------------------------------------------------------
def tune_models(X_train, y_train):
    """GridSearchCV with 5-fold CV on SVM and Random Forest."""
    print("\n" + "=" * 60)
    print("  STEP 3 : HYPERPARAMETER TUNING  (GridSearchCV, 5-fold CV)")
    print("=" * 60)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # -- SVM ----------------------------------------------
    svm_params = {
        "C":      [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf"],
    }
    svm_grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        svm_params, cv=cv, scoring="f1", n_jobs=-1
    )
    svm_grid.fit(X_train, y_train)
    print(f"\n  SVM  Best Params : {svm_grid.best_params_}")
    print(f"       Best CV F1   : {svm_grid.best_score_:.4f}")

    # -- Random Forest ------------------------------------
    rf_params = {
        "n_estimators": [50, 100, 200],
        "max_depth":    [None, 5, 10, 20],
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params, cv=cv, scoring="f1", n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    print(f"\n  RF   Best Params : {rf_grid.best_params_}")
    print(f"       Best CV F1   : {rf_grid.best_score_:.4f}")

    tuned_models = {
        "SVM (Tuned)":           svm_grid.best_estimator_,
        "Random Forest (Tuned)": rf_grid.best_estimator_,
    }
    return tuned_models


# ---------------------------------------------------------
# 5. TRAIN & EVALUATE ALL MODELS
# ---------------------------------------------------------
def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    """Fit model and return a metrics dict."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall":    round(recall_score(y_test, y_pred),    4),
        "F1-Score":  round(f1_score(y_test, y_pred),        4),
    }


def train_all_models(base_models, tuned_models, X_train, X_test, y_train, y_test):
    """Train and evaluate all models; return results DataFrame."""
    print("\n" + "=" * 60)
    print("  STEP 4 : TRAINING & EVALUATING ALL MODELS")
    print("=" * 60)

    all_models = {**base_models, **tuned_models}
    results = []

    for name, model in all_models.items():
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(metrics)
        print(f"\n  [{name}]")
        print(f"    Accuracy  : {metrics['Accuracy']:.4f}")
        print(f"    Precision : {metrics['Precision']:.4f}")
        print(f"    Recall    : {metrics['Recall']:.4f}")
        print(f"    F1-Score  : {metrics['F1-Score']:.4f}")

    results_df = pd.DataFrame(results).set_index("Model")
    return results_df, all_models


# ---------------------------------------------------------
# 6. RESULTS TABLE & BEST MODEL
# ---------------------------------------------------------
def show_results_table(results_df):
    """Print clean comparison table and identify best model."""
    print("\n" + "=" * 60)
    print("  STEP 5 : MODEL COMPARISON TABLE")
    print("=" * 60)

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)

    print(f"\n{results_df.to_string()}\n")

    # Best model by F1 then Recall
    best_row   = results_df.sort_values(
        ["F1-Score", "Recall"], ascending=False
    ).iloc[0]
    best_name  = best_row.name

    print("-" * 60)
    print(f"  [BEST]  BEST MODEL  ->  {best_name}")
    print(f"      Accuracy   : {best_row['Accuracy']:.4f}")
    print(f"      Precision  : {best_row['Precision']:.4f}")
    print(f"      Recall     : {best_row['Recall']:.4f}")
    print(f"      F1-Score   : {best_row['F1-Score']:.4f}")
    print("-" * 60)
    print(f"\n  INSIGHT:")
    print(f"  '{best_name}' achieves the highest F1-Score and Recall.")
    print("  In medical diagnosis, RECALL is critical because missing a")
    print("  Malignant tumor (false negative) is far more dangerous than")
    print("  a false positive. High Recall = fewer missed cancers.")

    return best_name


# ---------------------------------------------------------
# 7. CONFUSION MATRIX  (best model)
# ---------------------------------------------------------
def plot_confusion_matrix(model, X_test, y_test, model_name, target_names):
    """Plot and save confusion matrix heatmap for best model."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        linewidths=0.5, linecolor="gray",
        ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Confusion Matrix saved -> {path}")

    print(f"\n  Classification Report ({model_name}):")
    print(classification_report(y_test, y_pred, target_names=list(target_names)))


# ---------------------------------------------------------
# 8. ROC CURVE  (best model)
# ---------------------------------------------------------
def plot_roc_curve(model, X_test, y_test, model_name):
    """Plot and save ROC curve for best model."""
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Guess")
    ax.fill_between(fpr, tpr, alpha=0.12, color="#4C72B0")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(f"ROC Curve - {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ROC Curve saved      -> {path}")
    print(f"  AUC Score            : {roc_auc:.4f}")


# ---------------------------------------------------------
# 9. BAR CHART  (all models comparison)
# ---------------------------------------------------------
def plot_model_comparison(results_df):
    """Bar chart comparing all models across 4 metrics."""
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
    n_models  = len(results_df)
    n_metrics = len(metrics_to_plot)
    x         = np.arange(n_models)
    width     = 0.18

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, results_df[metric], width,
                      label=metric, color=color, alpha=0.87, edgecolor="white")
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=7, rotation=45
            )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison - Accuracy, Precision, Recall, F1-Score",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df.index, rotation=20, ha="right", fontsize=10)
    ax.set_ylim([0.80, 1.02])
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Model Comparison chart saved -> {path}")


# ---------------------------------------------------------
# 10. FEATURE IMPORTANCE  (Random Forest)
# ---------------------------------------------------------
def plot_feature_importance(all_models, feature_names, top_n=15):
    """Plot top-N feature importances from Random Forest."""
    print("\n" + "=" * 60)
    print("  STEP 6 : FEATURE IMPORTANCE  (Random Forest)")
    print("=" * 60)

    # Pick tuned RF if available, else base RF
    rf_key = "Random Forest (Tuned)" if "Random Forest (Tuned)" in all_models \
             else "Random Forest"
    rf = all_models[rf_key]

    importances = rf.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    top_features = [(feature_names[i], importances[i]) for i in indices]

    print(f"\n  Top-{top_n} Important Features ({rf_key}):")
    print(f"  {'Rank':<5} {'Feature':<35} {'Importance':>10}")
    print("  " + "-" * 52)
    for rank, (feat, imp) in enumerate(top_features, 1):
        print(f"  {rank:<5} {feat:<35} {imp:>10.4f}")

    # Plot
    feats = [f for f, _ in top_features]
    imps  = [i for _, i in top_features]

    fig, ax = plt.subplots(figsize=(9, 6))
    palette = sns.color_palette("Blues_d", len(feats))
    bars = ax.barh(feats[::-1], imps[::-1], color=palette, edgecolor="white")
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(f"Top-{top_n} Feature Importances - {rf_key}",
                 fontsize=14, fontweight="bold")
    for bar, imp in zip(bars, imps[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{imp:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Feature Importance chart saved -> {path}")

    # Interpretation
    top3 = top_features[:3]
    print("\n  INTERPRETATION:")
    print(f"  The top 3 most influential features are:")
    for rank, (feat, imp) in enumerate(top3, 1):
        print(f"    {rank}. '{feat}'  (importance={imp:.4f})")
    print("\n  These features differentiate Benign from Malignant tumors most")
    print("  strongly. Higher-importance = stronger discriminative power.")
    print("  Features like 'worst radius', 'worst perimeter', and 'worst area'")
    print("  reflect tumor size at its worst recorded measurement, which")
    print("  clinically correlates with malignancy severity.")


# ---------------------------------------------------------
# 11. PREDICTION FUNCTION  (bonus)
# ---------------------------------------------------------
def predict_tumor(model, scaler, feature_names, sample_features=None):
    """
    Predict whether a tumor is Benign or Malignant.

    Parameters
    ----------
    model          : trained sklearn model
    scaler         : fitted StandardScaler
    feature_names  : list of feature names (30 features)
    sample_features: array-like of shape (30,)  - if None, uses a demo sample

    Returns
    -------
    str : 'Benign' or 'Malignant'
    """
    if sample_features is None:
        # Demo: first row from the dataset (known Malignant)
        raw = load_breast_cancer()
        sample_features = raw.data[0]

    sample = np.array(sample_features).reshape(1, -1)
    sample_sc = scaler.transform(sample)
    pred  = model.predict(sample_sc)[0]
    proba = model.predict_proba(sample_sc)[0]

    label = "Benign" if pred == 1 else "Malignant"
    confidence = proba[pred] * 100

    print("\n" + "=" * 60)
    print("  STEP 7 : PREDICTION DEMO")
    print("=" * 60)
    print(f"\n  Input Sample (first 5 features shown):")
    for fname, val in zip(feature_names[:5], sample_features[:5]):
        print(f"    {fname:<35} = {val:.4f}")
    print("    ...")
    print(f"\n  >>  Prediction  : {label}")
    print(f"  >>  Confidence  : {confidence:.2f}%")
    print(f"  >>  Probabilities: Malignant={proba[0]*100:.2f}%  |  "
          f"Benign={proba[1]*100:.2f}%")
    return label


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def main():
    print("\n" + "=" * 60)
    print("  TUMOR CLASSIFICATION - COMPARATIVE ML MODELS")
    print("  Breast Cancer Wisconsin Dataset")
    print("=" * 60)

    # 1. Load data
    df, X, y, feature_names, target_names = load_and_prepare_data()

    # 2. Preprocess
    X_train, X_test, y_train, y_test, scaler, X_train_raw, X_test_raw = \
        preprocess_data(X, y)

    # 3. Tune SVM + RF
    tuned_models = tune_models(X_train, y_train)

    # 4. Base models + train all
    base_models = get_base_models()
    results_df, all_models = train_all_models(
        base_models, tuned_models, X_train, X_test, y_train, y_test
    )

    # 5. Results table + best model
    best_name = show_results_table(results_df)

    # 6. Visualizations
    print("\n" + "=" * 60)
    print("  STEP 5b : GENERATING VISUALIZATIONS")
    print("=" * 60)
    plot_model_comparison(results_df)

    best_model = all_models[best_name]
    plot_confusion_matrix(best_model, X_test, y_test, best_name, target_names)
    plot_roc_curve(best_model, X_test, y_test, best_name)

    # 7. Feature importance
    plot_feature_importance(all_models, feature_names, top_n=15)

    # 8. Bonus: Prediction demo
    predict_tumor(best_model, scaler, list(feature_names))

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE  [OK]")
    print(f"  All plots saved in -> ./{OUTPUT_DIR}/")
    print("=" * 60 + "\n")


# ---------------------------------------------------------
if __name__ == "__main__":
    main()
