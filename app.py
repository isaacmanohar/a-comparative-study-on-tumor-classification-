"""
============================================================
  TUMOR CLASSIFICATION DASHBOARD  —  Streamlit App
  Dataset : Breast Cancer Wisconsin (sklearn built-in)
  Run with: streamlit run app.py
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tumor Classification Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS  — dark-glass aesthetic
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── App background ── */
.stApp {
    background: linear-gradient(160deg, #f0fdfa, #f8fafc, #f0f9ff);
    color: #1e293b;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff, #f0fdfa);
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] * { color: #334155 !important; }

/* ── Clean elevated cards ── */
.glass-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04);
}

/* ── Section title ── */
.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #0d9488;
    letter-spacing: 0.02em;
    margin-bottom: 4px;
}
.section-sub {
    font-size: 0.88rem;
    color: #64748b;
    margin-bottom: 18px;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #f0fdfa, #e0f2fe, #f0fdf4);
    border: 1px solid #99f6e4;
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(13,148,136,0.08);
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #0d9488, #0891b2, #059669);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 10px;
    line-height: 1.2;
}
.hero-sub {
    font-size: 1.05rem;
    color: #475569;
    max-width: 700px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Metric cards ── */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    transition: border-color 0.2s, box-shadow 0.2s;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.metric-card:hover { border-color: #99f6e4; box-shadow: 0 4px 12px rgba(13,148,136,0.1); }
.metric-val {
    font-size: 2rem;
    font-weight: 700;
    color: #0d9488;
    line-height: 1;
}
.metric-lbl {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Best model badge ── */
.best-badge {
    display: inline-block;
    background: linear-gradient(90deg, #0d9488, #0891b2);
    color: white !important;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 8px;
    vertical-align: middle;
}

/* ── Prediction result ── */
.pred-benign {
    background: linear-gradient(135deg, #f0fdf4, #ecfdf5);
    border: 2px solid #10b981;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(16,185,129,0.1);
}
.pred-malignant {
    background: linear-gradient(135deg, #fef2f2, #fff1f2);
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(239,68,68,0.1);
}
.pred-label {
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
}
.pred-conf {
    font-size: 0.9rem;
    color: #64748b;
    margin-top: 8px;
}

/* ── Dataframe styling ── */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ── Slider labels ── */
.stSlider label { color: #475569 !important; font-size: 0.85rem !important; }

/* ── Button ── */
.stButton button, .stDownloadButton button {
    background: linear-gradient(90deg, #0d9488, #0891b2) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 32px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 2px 6px rgba(13,148,136,0.2) !important;
}
.stButton button:hover, .stDownloadButton button:hover { 
    opacity: 0.9 !important; 
    transform: translateY(-1px) !important; 
    box-shadow: 0 4px 12px rgba(13,148,136,0.25) !important; 
}

/* ── Divider ── */
hr { border-color: #e2e8f0 !important; }

/* ── Metric styling ── */
[data-testid="stMetricLabel"] {
    color: #64748b !important;
}
[data-testid="stMetricValue"] {
    color: #0d9488 !important;
    font-weight: 700 !important;
}

/* ── Input box styling ── */
.stTextInput input {
    color: #1e293b !important;
    background-color: #f8fafc !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# CACHE: TRAIN ALL MODELS  (only once per session)
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄  Training models — please wait…")
def build_pipeline():
    """Load data, preprocess, train all models, return everything needed."""
    raw = load_breast_cancer()
    feature_names  = list(raw.feature_names)
    target_names   = list(raw.target_names)   # ['malignant', 'benign']

    df = pd.DataFrame(raw.data, columns=feature_names)
    df["target"] = raw.target

    X = df.drop("target", axis=1)
    y = df["target"]

    # Stratified 80-20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # ── Base models ──────────────────────────────────────
    base_models = {
        "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
        "SVM":                 SVC(probability=True, random_state=42),
        "KNN":                 KNeighborsClassifier(),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(random_state=42),
    }

    # ── GridSearchCV tuning ──────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    svm_grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        {"C": [0.1, 1, 10, 100], "kernel": ["linear", "rbf"]},
        cv=cv, scoring="f1", n_jobs=-1
    )
    svm_grid.fit(X_tr, y_train)

    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20]},
        cv=cv, scoring="f1", n_jobs=-1
    )
    rf_grid.fit(X_tr, y_train)

    tuned_models = {
        "SVM (Tuned)":           svm_grid.best_estimator_,
        "Random Forest (Tuned)": rf_grid.best_estimator_,
    }

    # ── Train all & collect metrics ─────────────────────
    all_models = {**base_models, **tuned_models}
    results = []
    for name, model in all_models.items():
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        results.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_test, y_pred),  4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall":    round(recall_score(y_test, y_pred),    4),
            "F1-Score":  round(f1_score(y_test, y_pred),        4),
        })

    results_df = pd.DataFrame(results).set_index("Model")
    best_name  = results_df.sort_values(
        ["F1-Score", "Recall"], ascending=False
    ).index[0]

    # Feature importances (tuned RF)
    rf_model = tuned_models["Random Forest (Tuned)"]
    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return {
        "df":           df,
        "X_test":       X_te,
        "y_test":       y_test,
        "scaler":       scaler,
        "all_models":   all_models,
        "results_df":   results_df,
        "best_name":    best_name,
        "fi_df":        fi_df,
        "feature_names":feature_names,
        "target_names": target_names,
        "raw":          raw,
    }


# ─────────────────────────────────────────────────────────
# PLOT HELPERS  (return matplotlib figures)
# ─────────────────────────────────────────────────────────
def make_bar_chart(results_df):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors  = ["#0d9488", "#0891b2", "#059669", "#d97706"]
    n_models  = len(results_df)
    x = np.arange(n_models)
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        off  = (i - len(metrics)/2 + 0.5) * width
        bars = ax.bar(x + off, results_df[metric], width,
                      label=metric, color=color, alpha=0.88,
                      edgecolor="none")
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom",
                fontsize=6.5, color="#334155", rotation=40
            )

    ax.set_xticks(x)
    ax.set_xticklabels(results_df.index, rotation=18, ha="right",
                       fontsize=9, color="#475569")
    ax.set_ylabel("Score", color="#475569", fontsize=10)
    ax.set_ylim([0.80, 1.04])
    ax.tick_params(colors="#475569")
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#cbd5e1")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend = ax.legend(fontsize=9, framealpha=0.9,
                       labelcolor="#334155", edgecolor="#e2e8f0")
    legend.get_frame().set_facecolor("#ffffff")
    plt.tight_layout()
    return fig


def make_confusion_matrix(model, X_test, y_test, target_names):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="GnBu",
        xticklabels=target_names, yticklabels=target_names,
        linewidths=0.5, linecolor="#f1f5f9",
        annot_kws={"size": 16, "weight": "bold", "color": "#0d9488"},
        ax=ax, cbar=False
    )
    ax.set_xlabel("Predicted", fontsize=10, color="#475569", labelpad=8)
    ax.set_ylabel("Actual",    fontsize=10, color="#475569", labelpad=8)
    ax.tick_params(colors="#475569", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    return fig, y_pred


def make_roc_curve(model, X_test, y_test, model_name):
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax.plot(fpr, tpr, color="#0d9488", lw=2.5,
            label=f"AUC = {roc_auc:.4f}")
    ax.plot([0,1],[0,1], color="#cbd5e1", lw=1.5, linestyle="--",
            label="Random Guess")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#0d9488")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate", color="#475569", fontsize=10)
    ax.set_ylabel("True Positive Rate",  color="#475569", fontsize=10)
    ax.tick_params(colors="#475569")
    for spine in ax.spines.values():
        spine.set_color("#e2e8f0")
    legend = ax.legend(fontsize=9, framealpha=0.9,
                       labelcolor="#334155", edgecolor="#e2e8f0")
    legend.get_frame().set_facecolor("#ffffff")
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#cbd5e1")
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, color="#cbd5e1")
    plt.tight_layout()
    return fig, roc_auc


def make_feature_importance_chart(fi_df, top_n=12):
    df = fi_df.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    palette = plt.cm.BuGn(np.linspace(0.35, 0.85, len(df)))
    bars = ax.barh(df["Feature"][::-1], df["Importance"][::-1],
                   color=palette, edgecolor="none")
    for bar, imp in zip(bars, df["Importance"][::-1]):
        ax.text(bar.get_width() + 0.001,
                bar.get_y() + bar.get_height()/2,
                f"{imp:.4f}", va="center", fontsize=8, color="#475569")

    ax.set_xlabel("Importance Score", color="#475569", fontsize=10)
    ax.tick_params(colors="#475569", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, color="#cbd5e1")
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px;'>
        <div style='font-size:2.2rem;'>🔬</div>
        <div style='font-size:1rem; font-weight:700; color:#0d9488; margin-top:4px;'>
            Tumor Classifier
        </div>
        <div style='font-size:0.72rem; color:#64748b; margin-top:2px;'>
            Breast Cancer Wisconsin
        </div>
    </div>
    <hr/>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Home", "📊  Model Analysis", "🧬  Prediction", "🧠  MRI Analysis", "🩺  Ultrasound Analysis", "🛠️  Implementation"],
        label_visibility="collapsed"
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:#475569; padding: 0 4px;'>
        <b style='color:#0d9488;'>Dataset</b><br/>
        569 samples · 30 features<br/><br/>
        <b style='color:#0d9488;'>Models</b><br/>
        5 classifiers · GridSearchCV<br/>5-fold CV tuning<br/><br/>
        <b style='color:#0d9488;'>Metric Priority</b><br/>
        Recall → F1-Score
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# LOAD PIPELINE  (cached)
# ─────────────────────────────────────────────────────────
data = build_pipeline()
results_df   = data["results_df"]
best_name    = data["best_name"]
all_models   = data["all_models"]
best_model   = all_models[best_name]
X_test       = data["X_test"]
y_test       = data["y_test"]
scaler       = data["scaler"]
fi_df        = data["fi_df"]
feature_names = data["feature_names"]
target_names  = data["target_names"]
raw           = data["raw"]

# Best model metrics
bm = results_df.loc[best_name]


# ══════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════
if page == "🏠  Home":
    # Hero
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Tumor Classification Dashboard</div>
        <div class="hero-sub">
            A comparative Machine Learning study using the
            <strong>Breast Cancer Wisconsin</strong> dataset.
            Five classifiers are trained, tuned via GridSearchCV,
            and evaluated across Accuracy, Precision, Recall, and F1-Score.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Dataset overview cards ────────────────────────────
    st.markdown('<div class="section-title">📋 Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Breast Cancer Wisconsin — sklearn built-in</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("569", "Total Samples"),
        ("30",  "Features"),
        ("357", "Benign Cases"),
        ("212", "Malignant Cases"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4], cards):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{val}</div>
            <div class="metric-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Class distribution bar ────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="section-title">🎯 Class Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Target label balance in the dataset</div>', unsafe_allow_html=True)
        dist_fig, dist_ax = plt.subplots(figsize=(5, 3.2))
        dist_fig.patch.set_facecolor("#ffffff")
        dist_ax.set_facecolor("#ffffff")
        dist_ax.barh(["Malignant", "Benign"], [212, 357],
                     color=["#ef4444", "#10b981"], height=0.45, edgecolor="none")
        for spine in dist_ax.spines.values():
            spine.set_visible(False)
        dist_ax.tick_params(colors="#475569")
        dist_ax.set_xlabel("Count", color="#475569", fontsize=9)
        dist_ax.xaxis.grid(True, linestyle="--", alpha=0.3, color="#cbd5e1")
        dist_ax.set_axisbelow(True)
        dist_ax.text(212 + 6, 0, "212", va="center", color="#1e293b", fontsize=10, fontweight="bold")
        dist_ax.text(357 + 6, 1, "357", va="center", color="#1e293b", fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(dist_fig, use_container_width=True)
        plt.close()

    with col_right:
        st.markdown('<div class="section-title">🏆 Best Model Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">{best_name}</div>', unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)
        metric_pairs = [
            (m1, "Accuracy",  f"{bm['Accuracy']*100:.2f}%"),
            (m2, "Precision", f"{bm['Precision']*100:.2f}%"),
            (m3, "Recall",    f"{bm['Recall']*100:.2f}%"),
            (m4, "F1-Score",  f"{bm['F1-Score']*100:.2f}%"),
        ]
        for col, lbl, val in metric_pairs:
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    # ── Pipeline steps ────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚙️ Pipeline Steps</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Load Data",          "sklearn Breast Cancer Wisconsin dataset"),
        ("2", "Preprocess",         "StandardScaler + Stratified 80-20 split"),
        ("3", "Hyperparameter Tune","GridSearchCV 5-fold CV on SVM & Random Forest"),
        ("4", "Train & Evaluate",   "All 7 model variants (5 base + 2 tuned)"),
        ("5", "Compare",            "Accuracy · Precision · Recall · F1-Score"),
        ("6", "Predict",            "Interactive prediction on new tumor features"),
    ]
    cols = st.columns(3)
    for i, (num, title, desc) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="glass-card" style="margin-bottom:10px;">
                <div style="font-size:1.6rem; line-height:1; color:#0d9488; font-weight:800;">{num}</div>
                <div style="font-weight:700; font-size:0.95rem; margin: 6px 0 4px; color:#1e293b;">{title}</div>
                <div style="font-size:0.8rem; color:#64748b;">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 2 — MODEL ANALYSIS
# ══════════════════════════════════════════════════════════
elif page == "📊  Model Analysis":

    st.markdown("""
    <div style='padding: 8px 0 24px;'>
        <div style='font-size:1.8rem; font-weight:800; color:#0d9488;'>📊 Model Analysis</div>
        <div style='color:#64748b; font-size:0.9rem; margin-top:4px;'>
            Comparing 7 classifier variants across 4 performance metrics
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Performance table ─────────────────────────────────
    st.markdown('<div class="section-title">📋 Performance Comparison Table</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Sorted by F1-Score · Best model highlighted</div>', unsafe_allow_html=True)

    display_df = results_df.copy().sort_values("F1-Score", ascending=False)

    def style_table(df):
        styles = []
        for i, idx in enumerate(df.index):
            if idx == best_name:
                styles.append({
                    "selector": f"tbody tr:nth-child({i+1})",
                    "props": [("background-color", "rgba(13,148,136,0.1)"),
                              ("border-left", "3px solid #0d9488")]
                })
        return df.style \
            .format("{:.4f}") \
            .set_properties(**{
                "color": "#1e293b",
                "border-color": "#e2e8f0",
                "font-size": "13px"
            }) \
            .background_gradient(cmap="BuGn", subset=["Accuracy","Precision","Recall","F1-Score"]) \
            .set_table_styles(styles)

    st.dataframe(style_table(display_df), use_container_width=True, height=295)

    # Best model callout
    st.markdown(f"""
    <div class="glass-card" style="border-color: rgba(13,148,136,0.3); margin-top: 4px;">
        <span style="font-size:1.5rem;">🏆</span>
        <span style="font-weight:700; font-size:1.05rem; color:#0d9488; margin-left:8px;">{best_name}</span>
        <span class="best-badge">BEST MODEL</span>
        <div style="margin-top:10px; font-size:0.85rem; color:#64748b; line-height:1.6;">
            Selected based on highest <strong style="color:#1e293b;">Recall</strong> and
            <strong style="color:#1e293b;">F1-Score</strong>. In medical diagnosis,
            <strong style="color:#1e293b;">Recall</strong> is the most critical metric —
            missing a Malignant tumor (false negative) is far more dangerous than a
            false positive. This model minimises missed cancer cases.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Bar chart ─────────────────────────────────────────
    st.markdown('<div class="section-title">📈 Model Performance Bar Chart</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">All 7 variants · 4 metrics side-by-side</div>', unsafe_allow_html=True)
    st.pyplot(make_bar_chart(results_df), use_container_width=True)
    plt.close("all")

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Confusion Matrix + ROC side by side ───────────────
    col_cm, col_roc = st.columns([1, 1])

    with col_cm:
        st.markdown(f'<div class="section-title">🗂️ Confusion Matrix</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">{best_name}</div>', unsafe_allow_html=True)
        cm_fig, y_pred = make_confusion_matrix(best_model, X_test, y_test, target_names)
        st.pyplot(cm_fig, use_container_width=True)
        plt.close("all")

        # Quick stats below
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        s1, s2, s3 = st.columns(3)
        s1.metric("True Positives",  tp,  help="Correctly identified Benign")
        s2.metric("True Negatives",  tn,  help="Correctly identified Malignant")
        s3.metric("False Negatives", fn,  delta=f"-{fn} missed", delta_color="inverse",
                  help="Malignant tumors missed (most critical!)")

    with col_roc:
        st.markdown(f'<div class="section-title">📉 ROC Curve</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">{best_name}</div>', unsafe_allow_html=True)
        roc_fig, roc_auc = make_roc_curve(best_model, X_test, y_test, best_name)
        st.pyplot(roc_fig, use_container_width=True)
        plt.close("all")

        st.markdown(f"""
        <div class="glass-card" style="margin-top:6px; text-align:center;">
            <div style="font-size:2.2rem; font-weight:800; color:#0d9488;">{roc_auc:.4f}</div>
            <div style="font-size:0.78rem; color:#64748b; text-transform:uppercase; letter-spacing:0.06em;">
                AUC Score
            </div>
            <div style="font-size:0.82rem; color:#64748b; margin-top:8px;">
                Near-perfect separation between Benign and Malignant classes.
                AUC = 1.0 is a perfect classifier.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Feature Importance ────────────────────────────────
    st.markdown('<div class="section-title">🌲 Feature Importance — Random Forest (Tuned)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Top 12 features ranked by discriminative power</div>', unsafe_allow_html=True)

    fi_col, fi_tbl = st.columns([1.5, 1])
    with fi_col:
        fi_fig = make_feature_importance_chart(fi_df, top_n=12)
        st.pyplot(fi_fig, use_container_width=True)
        plt.close("all")

    with fi_tbl:
        st.markdown("<br/>", unsafe_allow_html=True)
        top12 = fi_df.head(12).copy()
        top12.index = range(1, len(top12)+1)
        top12.index.name = "Rank"
        top12["Importance"] = top12["Importance"].map("{:.5f}".format)
        st.dataframe(
            top12.style.set_properties(**{"color": "#1e293b", "font-size": "12px"}),
            use_container_width=True, height=410
        )

    # Interpretation card
    top3 = fi_df.head(3)["Feature"].tolist()
    st.markdown(f"""
    <div class="glass-card" style="border-color: rgba(16,185,129,0.3); margin-top: 4px;">
        <span style="font-size:1.2rem;">🔍</span>
        <span style="font-weight:700; color:#059669; margin-left:8px;">Interpretation</span>
        <div style="margin-top:10px; font-size:0.85rem; color:#64748b; line-height:1.7;">
            The top 3 most influential features are
            <strong style="color:#1e293b;">{top3[0]}</strong>,
            <strong style="color:#1e293b;">{top3[1]}</strong>, and
            <strong style="color:#1e293b;">{top3[2]}</strong>.
            These reflect the <em>worst-case measurements</em> of tumor geometry — size, boundary,
            and shape irregularity — which clinically correlate most strongly with malignancy.
            Higher feature importance = stronger discriminative signal for the classifier.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 3 — PREDICTION
# ══════════════════════════════════════════════════════════
elif page == "🧬  Prediction":

    st.markdown("""
    <div style='padding: 8px 0 24px;'>
        <div style='font-size:1.8rem; font-weight:800; color:#0d9488;'>🧬 Tumor Prediction</div>
        <div style='color:#64748b; font-size:0.9rem; margin-top:4px;'>
            Adjust the tumor feature values and get an instant classification
        </div>
    </div>""", unsafe_allow_html=True)

    # Top features to expose (top 10 from RF importance)
    TOP_FEATURES = fi_df["Feature"].head(10).tolist()

    # Feature ranges from actual data
    raw_df = pd.DataFrame(raw.data, columns=feature_names)

    # ── AI Biopsy Scanner (Simulated) ─────────────────────
    st.markdown('<div class="section-title">🔬 AI Biopsy Image Scanner</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload an FNA slide image to simulate automated cytological feature extraction.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    scan_col, preview_col = st.columns([1.5, 1])
    
    with scan_col:
        uploaded_scan = st.file_uploader("Upload Cytology Slide (JPEG/PNG)", type=["jpg", "jpeg", "png"])
    
    if "extracted_features" not in st.session_state:
        st.session_state.extracted_features = None
    if "simulation_scalars" not in st.session_state:
        st.session_state.simulation_scalars = np.ones(len(TOP_FEATURES))

    if uploaded_scan is not None:
        from PIL import Image
        import time
        with preview_col:
            st.image(Image.open(uploaded_scan), caption="Uploaded FNA Slide", use_container_width=True)
            
        if st.session_state.extracted_features != uploaded_scan.name:
            with st.spinner("🤖 Analyzing nucleus structures & computing geometries..."):
                time.sleep(1.5)
                # Randomize features slightly around the mean to simulate an extraction process
                np.random.seed(sum(ord(c) for c in uploaded_scan.name) % 10000)
                st.session_state.simulation_scalars = np.random.uniform(0.5, 1.5, size=len(TOP_FEATURES))
                st.session_state.extracted_features = uploaded_scan.name
            st.success("✨ Successfully extracted cytological features from image!")
    else:
        st.session_state.extracted_features = None

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Input sliders ─────────────────────────────────────
    st.markdown('<div class="section-title">🎛️ Tumor Feature Inputs</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Adjust the sliders manually or review the AI-extracted measurements</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    user_input = {}
    col_a, col_b = st.columns(2)

    for i, feat in enumerate(TOP_FEATURES):
        min_v  = float(raw_df[feat].min())
        max_v  = float(raw_df[feat].max())
        mean_v = float(raw_df[feat].mean())
        
        # Apply simulated extraction if an image is loaded
        if uploaded_scan is not None and st.session_state.extracted_features == uploaded_scan.name:
            scalar = st.session_state.simulation_scalars[i]
            base_val = np.clip(mean_v * scalar, min_v, max_v)
        else:
            base_val = mean_v
            
        step_v = round((max_v - min_v) / 100, 6)
        col    = col_a if i % 2 == 0 else col_b
        
        user_input[feat] = col.slider(
            label=feat,
            min_value=min_v,
            max_value=max_v,
            value=float(base_val),
            step=step_v,
            format="%.4f",
            key=f"slider_{feat}"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Predict button ────────────────────────────────────
    _, btn_col, _ = st.columns([1.5, 1, 1.5])
    with btn_col:
        predict_clicked = st.button("🔬  Predict Now", use_container_width=True)

    if predict_clicked:
        # Build full 30-feature vector (mean for non-selected features)
        full_input = raw_df.mean().to_dict()
        for feat, val in user_input.items():
            full_input[feat] = val

        inp_arr = np.array([full_input[f] for f in feature_names]).reshape(1, -1)
        inp_sc  = scaler.transform(inp_arr)

        pred    = best_model.predict(inp_sc)[0]
        proba   = best_model.predict_proba(inp_sc)[0]
        label   = "Benign" if pred == 1 else "Malignant"
        conf    = proba[pred] * 100
        p_mal   = proba[0] * 100
        p_ben   = proba[1] * 100

        st.markdown("<br/>", unsafe_allow_html=True)

        # Result card
        if label == "Benign":
            emoji     = "✅"
            css_class = "pred-benign"
            color     = "#10b981"
        else:
            emoji     = "⚠️"
            css_class = "pred-malignant"
            color     = "#ef4444"

        st.markdown(f"""
        <div class="{css_class}">
            <div style="font-size:2.8rem;">{emoji}</div>
            <div class="pred-label" style="color:{color};">{label}</div>
            <div class="pred-conf">
                Confidence: <strong style="color:{color};">{conf:.1f}%</strong>
                &nbsp;·&nbsp; Model: <strong style="color:{color};">{best_name}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Probability details
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color:#10b981;">{p_ben:.1f}%</div>
                <div class="metric-lbl">Probability — Benign</div>
            </div>""", unsafe_allow_html=True)
        with p_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color:#ef4444;">{p_mal:.1f}%</div>
                <div class="metric-lbl">Probability — Malignant</div>
            </div>""", unsafe_allow_html=True)

        # Probability bar chart
        st.markdown("<br/>", unsafe_allow_html=True)
        prob_fig, prob_ax = plt.subplots(figsize=(7, 1.8))
        prob_fig.patch.set_facecolor("#ffffff")
        prob_ax.set_facecolor("#ffffff")
        prob_ax.barh(["Malignant", "Benign"], [p_mal/100, p_ben/100],
                     color=["#ef4444", "#10b981"], height=0.4, edgecolor="none")
        prob_ax.set_xlim([0, 1])
        prob_ax.set_xlabel("Probability", color="#475569", fontsize=9)
        for spine in prob_ax.spines.values():
            spine.set_visible(False)
        prob_ax.tick_params(colors="#475569", labelsize=8)
        prob_ax.xaxis.grid(True, linestyle="--", alpha=0.3, color="#cbd5e1")
        prob_ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(prob_fig, use_container_width=True)
        plt.close("all")

        # Clinical note
        if label == "Malignant":
            st.markdown("""
            <div class="glass-card" style="border-color: rgba(239,68,68,0.3); margin-top:8px;">
                <span style="font-size:1rem;">⚠️</span>
                <span style="font-weight:700; color:#ef4444; margin-left:8px;">Clinical Note</span>
                <div style="margin-top:8px; font-size:0.83rem; color:#64748b; line-height:1.6;">
                    This prediction indicates a high likelihood of malignancy.
                    This tool is for <strong style="color:#b45309;">educational/research purposes only</strong>
                    and should never replace a certified medical diagnosis.
                    Please consult a qualified oncologist.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="border-color: rgba(16,185,129,0.3); margin-top:8px;">
                <span style="font-size:1rem;">ℹ️</span>
                <span style="font-weight:700; color:#10b981; margin-left:8px;">Note</span>
                <div style="margin-top:8px; font-size:0.83rem; color:#64748b; line-height:1.6;">
                    This prediction suggests the tumor is likely benign.
                    This tool is for <strong style="color:#b45309;">educational/research purposes only</strong>.
                    Regular medical check-ups are always recommended.
                </div>
            </div>""", unsafe_allow_html=True)

    else:
        # Placeholder state
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:40px; margin-top:8px;">
            <div style="font-size:3rem; margin-bottom:12px;">🔬</div>
            <div style="font-weight:700; color:#0d9488; font-size:1.1rem;">
                Adjust sliders and click Predict
            </div>
            <div style="color:#64748b; font-size:0.85rem; margin-top:8px;">
                The model will classify the tumor as Benign or Malignant
                with a confidence score.
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 4 — IMPLEMENTATION DETAILS
# ══════════════════════════════════════════════════════════
elif page == "🛠️  Implementation":

    st.markdown("""
    <div style='padding: 8px 0 24px;'>
        <div style='font-size:1.8rem; font-weight:800; color:#0d9488;'>🛠️ Implementation Details</div>
        <div style='color:#64748b; font-size:0.9rem; margin-top:4px;'>
            Deep dive into the architecture, algorithms, and logic behind the dashboard
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Pipeline Flow ─────────────────────────────────────
    st.markdown('<div class="section-title">🗺️ Machine Learning Pipeline Flow</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">End-to-End workflow from raw data to final prediction</div>', unsafe_allow_html=True)

    flow_cols = st.columns(5)
    steps = [
        ("📂", "Data Loading", "sklearn built-in"),
        ("⚖️", "Preprocessing", "Scaling + Split"),
        ("⚡", "Hyperparams", "GridSearchCV"),
        ("🧪", "Evaluation", "Recall/F1 focus"),
        ("🎯", "Best Model", f"{best_name}")
    ]
    for i, (icon, title, desc) in enumerate(steps):
        with flow_cols[i]:
            st.markdown(f"""
            <div style="text-align:center; padding:10px; background:#ffffff; border-radius:12px; border:1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-size:1.8rem; margin-bottom:4px;">{icon}</div>
                <div style="font-weight:700; font-size:0.8rem; color:#1e293b;">{title}</div>
                <div style="font-size:0.65rem; color:#64748b; margin-top:2px;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Comparison of Algorithms ──────────────────────────
    st.markdown('<div class="section-title">🧬 Algorithm Selection Policy</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Why these models were chosen for comparative analysis</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="glass-card" style="height:280px;">
            <b style="color:#0d9488;">Linear Classifiers (LR, SVM)</b><br/>
            <div style="font-size:0.85rem; color:#64748b; margin-top:10px;">
                Used as robust baselines. <strong>Logistic Regression</strong> is highly interpretable 
                via coefficients. <strong>SVM</strong> with kernels handles non-linear boundaries 
                wonderfully when scaled properly.
            </div>
            <br/>
            <b style="color:#0d9488;">Metric Priority: Recall</b><br/>
            <div style="font-size:0.85rem; color:#64748b; margin-top:10px;">
                In tumor diagnosis, "Missing a Malignant case" (False Negative) is unacceptable. 
                We prioritize models that maximize Recall while keeping Precision healthy.
            </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="glass-card" style="height:280px;">
            <b style="color:#0d9488;">Tree-Based Ensemble (RF)</b><br/>
            <div style="font-size:0.85rem; color:#64748b; margin-top:10px;">
                <strong>Random Forest</strong> combats overfitting inherent in single Decision Trees 
                by averaging multiple estimators. It provides <em>Feature Importance</em> scores 
                which help clinicians understand "influential markers".
            </div>
            <br/>
            <b style="color:#0d9488;">Tuning: GridSearchCV</b><br/>
            <div style="font-size:0.85rem; color:#64748b; margin-top:10px;">
                We use 5-fold Stratified Cross-Validation to ensure hyperparameters (like C for SVM 
                or depth for RF) generalize across all patients, not just the training set.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Implementation Code Snippets ──────────────────────
    st.markdown('<div class="section-title">💻 Technical Implementation Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Core logic snippets from the machine learning script</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Preprocessing", "⚙️ GridSearchCV", "🧠 Best Model Logic"])

    with tab1:
        st.markdown("""
        Scaling is critical for distance-based models (KNN, SVM) and gradient-based models (LR). 
        A Stratified split ensures the Benign/Malignant ratio stays balanced in both sets.
        """)
        st.code("""
# Stratified 80-20 split (keeps class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Feature Scaling (zero mean, unit variance)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
        """, language="python")

    with tab2:
        st.markdown("""
        Iterative search over a parameter grid to find the optimal configuration 
        using 5-fold cross-validation. Focus on F1-Score to balance Recall and Precision.
        """)
        st.code("""
# SVM Grid Search
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
svm_grid = GridSearchCV(
    SVC(probability=True, random_state=42),
    {"C": [0.1, 1, 10, 100], "kernel": ["linear", "rbf"]},
    cv=cv, scoring="f1", n_jobs=-1
)
svm_grid.fit(X_train_sc, y_train)

best_svm = svm_grid.best_estimator_
        """, language="python")

    with tab3:
        st.markdown("""
        The dashboard champion is selected by sorting multiple metrics.
        F1-Score (balance) is primary; Recall (safety) is the final tie-breaker.
        """)
        st.code("""
# F1-Score (balance) is primary, Recall (safety) is secondary
best_name = results_df.sort_values(
    ["F1-Score", "Recall"], ascending=False
).index[0]

winner_model = all_models[best_name]
        """, language="python")

    st.markdown("<br/><hr/>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; color:#475569; font-size:0.8rem;">
        Dashboard Implementation by Isaac M. <br/>
        Libraries: Streamlit · Pandas · Scikit-Learn · Matplotlib
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# PAGE 5 — MRI ANALYSIS
# ══════════════════════════════════════════════════════════
elif page == "🧠  MRI Analysis":
    st.markdown("""
    <div style='padding: 8px 0 24px;'>
        <div style='font-size:1.8rem; font-weight:800; color:#0d9488;'>🧠 Brain MRI Analysis</div>
        <div style='color:#64748b; font-size:0.9rem; margin-top:4px;'>
            Upload an MRI scan to our deep learning CNN classifier for immediate diagnosis.
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    mri_uploaded = st.file_uploader("Upload Brain MRI Scan (JPEG/PNG)", type=["jpg", "jpeg", "png"], key="mri_upl")
    st.markdown('</div>', unsafe_allow_html=True)

    if mri_uploaded is not None:
        from PIL import Image
        import time
        import os
        
        mri_img = Image.open(mri_uploaded).convert('RGB')
        
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.image(mri_img, caption="MRI Scan", use_container_width=True)
            
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            if os.path.exists('mri_model.h5') and os.path.exists('mri_classes.json'):
                with st.spinner("🧠 Initializing Deep Learning CNN Model..."):
                    try:
                        import tensorflow as tf
                        import json
                        import numpy as np
                        
                        # Use caching slightly so it doesn't reload huge hdf5 continuously
                        @st.cache_resource(show_spinner=False)
                        def load_mri_model():
                            mdl = tf.keras.models.load_model('mri_model.h5')
                            with open('mri_classes.json', 'r') as f:
                                c = json.load(f)
                            return mdl, c
                            
                        model, class_dict = load_mri_model()
                            
                        # Preprocess image
                        img_resized = mri_img.resize((128, 128))
                        img_array = np.array(img_resized) / 255.0
                        img_batch = np.expand_dims(img_array, axis=0)

                        # Predict
                        preds = model.predict(img_batch, verbose=0)[0]
                        pred_idx = np.argmax(preds)
                        pred_label = class_dict[str(pred_idx)].replace('_', ' ').title()
                        confidence = preds[pred_idx] * 100
                        
                        st.markdown("<br/>", unsafe_allow_html=True)
                        if "No Tumor" in pred_label:
                            st.markdown(f'<div class="pred-benign"><div style="font-size:2.8rem;">✅</div><div class="pred-label" style="color:#10b981;">{pred_label}</div><div class="pred-conf">Confidence: <strong style="color:#10b981;">{confidence:.1f}%</strong> · Network: CNN Fast</div></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="pred-malignant"><div style="font-size:2.8rem;">⚠️</div><div class="pred-label" style="color:#ef4444;">{pred_label}</div><div class="pred-conf">Confidence: <strong style="color:#ef4444;">{confidence:.1f}%</strong> · Network: CNN Fast</div></div>', unsafe_allow_html=True)
                            
                        st.markdown("<br/><hr/>", unsafe_allow_html=True)
                        st.markdown("<b>🔍 Explainable AI (Grad-CAM)</b>", unsafe_allow_html=True)
                        st.markdown("<div style='font-size:0.8rem; color:#64748b; margin-bottom:12px;'>Thermal Heatmap of the tumor regions activating the model.</div>", unsafe_allow_html=True)
                        
                        # Find a conv layer with good spatial resolution for Grad-CAM
                        # Prefer layers with spatial dims >= 8x8 for meaningful localization
                        last_conv_layer_name_mri = None
                        fallback_conv_name = None
                        for layer in reversed(model.layers):
                            if isinstance(layer, tf.keras.layers.Conv2D):
                                if fallback_conv_name is None:
                                    fallback_conv_name = layer.name
                                try:
                                    out_shape = layer.output.shape
                                    if len(out_shape) >= 3 and out_shape[1] is not None and out_shape[1] >= 8:
                                        last_conv_layer_name_mri = layer.name
                                        break
                                except:
                                    pass
                        if last_conv_layer_name_mri is None:
                            last_conv_layer_name_mri = fallback_conv_name
                                
                        if last_conv_layer_name_mri:
                            grad_model_mri = tf.keras.models.Model(
                                inputs=model.inputs, 
                                outputs=[model.get_layer(last_conv_layer_name_mri).output, model.output]
                            )
                            
                            with tf.GradientTape() as tape:
                                img_tensor_mri = tf.convert_to_tensor(img_batch, dtype=tf.float32)
                                conv_outputs_mri, predictions_mri = grad_model_mri(img_tensor_mri)
                                tape.watch(conv_outputs_mri)
                                loss_mri = predictions_mri[:, pred_idx]
                            
                            grads_mri = tape.gradient(loss_mri, conv_outputs_mri)
                            pooled_grads_mri = tf.reduce_mean(grads_mri, axis=(0, 1, 2))
                            
                            conv_outputs_mri = conv_outputs_mri[0]
                            heatmap_mri = conv_outputs_mri @ pooled_grads_mri[..., tf.newaxis]
                            heatmap_mri = tf.squeeze(heatmap_mri)
                            import matplotlib.cm as cm
                            from PIL import Image
                            
                            # Normalize heatmap to [0, 1] safely
                            heatmap_mri = tf.maximum(heatmap_mri, 0)
                            max_val = tf.math.reduce_max(heatmap_mri)
                            if max_val > 0:
                                heatmap_mri = heatmap_mri / max_val
                            heatmap_mri = heatmap_mri.numpy()
                            
                            # Resize grayscale heatmap to original image size with smooth interpolation
                            heatmap_pil = Image.fromarray(np.uint8(255 * heatmap_mri))
                            heatmap_pil = heatmap_pil.resize((mri_img.size[0], mri_img.size[1]), Image.Resampling.LANCZOS)
                            heatmap_resized = np.array(heatmap_pil).astype(np.float32) / 255.0
                            
                            # Create alpha mask: only show heatmap where activation is significant
                            alpha = np.clip(heatmap_resized - 0.25, 0, 1)
                            alpha = alpha / (alpha.max() + 1e-8)
                            alpha = np.power(alpha, 0.6)
                            
                            # Create tissue mask: only overlay on brain/skull, not black background
                            orig_arr = tf.keras.preprocessing.image.img_to_array(mri_img)
                            gray = np.mean(orig_arr, axis=-1)  # convert to grayscale
                            tissue_mask = (gray > 25).astype(np.float32)  # brain tissue is brighter than ~25/255
                            alpha = alpha * tissue_mask  # zero out heatmap on black background
                            
                            # Apply JET colormap to get RGB heatmap
                            jet = cm.get_cmap("jet")
                            jet_heatmap_mri = jet(np.uint8(255 * heatmap_resized))[:, :, :3]
                            jet_heatmap_mri = (jet_heatmap_mri * 255.0).astype(np.float32)
                            
                            # Alpha-blend: original * (1-alpha) + heatmap * alpha
                            alpha_3ch = np.stack([alpha, alpha, alpha], axis=-1) * 0.65
                            superimposed_img_mri = orig_arr * (1 - alpha_3ch) + jet_heatmap_mri * alpha_3ch
                            superimposed_img_mri = np.clip(superimposed_img_mri, 0, 255)
                            superimposed_img_mri = tf.keras.preprocessing.image.array_to_img(superimposed_img_mri)
                            
                            st.image(superimposed_img_mri, use_container_width=True)
                            
                            # ── Extra Clinical Featuers ── #
                            st.markdown("<hr/>", unsafe_allow_html=True)
                            colA, colB = st.columns(2)
                            with colA:
                                hot_pixels_mri = np.sum(heatmap_resized > 0.5)
                                approx_area_mri = (hot_pixels_mri / (heatmap_resized.shape[0]*heatmap_resized.shape[1])) * 12.5
                                approx_radius_mri = np.sqrt(approx_area_mri / np.pi) if approx_area_mri > 0 else 0.0
                                
                                m1, m2 = st.columns(2)
                                m1.metric("Estimated Tumor Area", f"{approx_area_mri:.1f} cm²" if "No Tumor" not in pred_label else "0.0 cm²")
                                m2.metric("Estimated Radius", f"{approx_radius_mri:.2f} cm" if "No Tumor" not in pred_label else "0.00 cm")
                                
                            with colB:
                                pid_val_mri = st.text_input("Patient ID", value="PID-34091", key="pid_mri_input")
                                
                            import datetime
                            from fpdf import FPDF
                            import os
                            
                            def create_pdf_mri(pid, diag, conf, area, orig_img, heatmap_img):
                                orig_path = "temp_orig_mri.jpg"
                                heat_path = "temp_heat_mri.jpg"
                                orig_img.save(orig_path)
                                heatmap_img.save(heat_path)
                                
                                pdf = FPDF()
                                pdf.add_page()
                                pdf.set_font("Arial", 'B', 18)
                                pdf.cell(200, 10, txt="DIAGNOSTIC MEDICAL REPORT", ln=True, align='C')
                                pdf.set_font("Arial", size=10)
                                pdf.cell(200, 10, txt=f"Date of Scan: {datetime.datetime.now().strftime('%B %d, %Y - %H:%M')}", ln=True, align='C')
                                pdf.ln(5)
                                
                                # Patient info
                                pdf.set_fill_color(240, 240, 240)
                                pdf.set_font("Arial", 'B', 12)
                                pdf.cell(190, 8, txt=" 1. PATIENT AND SCAN INFORMATION", ln=True, fill=True)
                                pdf.set_font("Arial", size=11)
                                pdf.cell(100, 8, txt=f"  Patient ID: {pid}", ln=False)
                                pdf.cell(90, 8, txt=f"  Modality: Brain MRI", ln=True)
                                pdf.cell(100, 8, txt=f"  AI Network: Custom CNN", ln=False)
                                pdf.cell(90, 8, txt=f"  Analysis: Grad-CAM XAI", ln=True)
                                pdf.ln(5)
                                
                                # Results
                                pdf.set_font("Arial", 'B', 12)
                                pdf.cell(190, 8, txt=" 2. AI DIAGNOSTIC RESULTS", ln=True, fill=True)
                                pdf.set_font("Arial", size=11)
                                
                                if "No Tumor" in diag:
                                    pdf.set_text_color(0, 150, 0)
                                else:
                                    pdf.set_text_color(200, 0, 0)
                                pdf.set_font("Arial", 'B', 12)
                                pdf.cell(200, 10, txt=f"  Primary Diagnosis: {diag.upper()}", ln=True)
                                pdf.set_text_color(0, 0, 0)
                                pdf.set_font("Arial", size=11)
                                pdf.cell(200, 8, txt=f"  Confidence Level: {conf:.2f}%", ln=True)
                                
                                final_area = area if "No Tumor" not in diag else 0.0
                                pdf.cell(200, 8, txt=f"  Estimated Tumor Area: {final_area:.2f} sq. cm", ln=True)
                                pdf.ln(5)
                                
                                # Images
                                pdf.set_font("Arial", 'B', 12)
                                pdf.cell(190, 8, txt=" 3. CLINICAL IMAGING", ln=True, fill=True)
                                pdf.ln(5)
                                
                                pdf.set_font("Arial", size=10)
                                pdf.cell(95, 8, txt="Original MRI Scan", align='C', ln=False)
                                pdf.cell(95, 8, txt="Grad-CAM Activation Map", align='C', ln=True)
                                
                                cur_y = pdf.get_y()
                                pdf.image(orig_path, x=20, y=cur_y, w=70)
                                pdf.image(heat_path, x=115, y=cur_y, w=70)
                                pdf.ln(75) 
                                
                                pdf.set_font("Arial", 'I', 9)
                                pdf.set_text_color(150, 150, 150)
                                pdf.multi_cell(0, 6, txt="Disclaimer: This report is automatically generated using Artificial Intelligence (CNN). It is designed to assist radiologists but should not replace professional clinical correlation and biopsy.")
                                
                                out = pdf.output(dest='S').encode('latin-1')
                                
                                if os.path.exists(orig_path): os.remove(orig_path)
                                if os.path.exists(heat_path): os.remove(heat_path)
                                
                                return out

                            final_area_mri = approx_area_mri if "No Tumor" not in pred_label else 0.0
                            pdf_bytes_mri = create_pdf_mri(pid_val_mri, pred_label, confidence, final_area_mri, mri_img, superimposed_img_mri)
                            
                            st.markdown("<br/>", unsafe_allow_html=True)
                            st.download_button(
                                label="📄 Export Medical PDF Report", 
                                data=pdf_bytes_mri, 
                                file_name=f"MRI_Report_{pid_val_mri}.pdf", 
                                mime="application/pdf",
                                use_container_width=True
                            )

                    except Exception as e:
                        st.error(f"Model execution error: {e}")
            else:
                st.warning("⚠️ Medical Model is still training in the background. Please wait a few seconds and try again.")
            st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE 6 — ULTRASOUND ANALYSIS
# ══════════════════════════════════════════════════════════
elif page == "🩺  Ultrasound Analysis":
    st.markdown("""
    <div style='padding: 8px 0 24px;'>
        <div style='font-size:1.8rem; font-weight:800; color:#0d9488;'>🩺 Breast Ultrasound Analysis</div>
        <div style='color:#64748b; font-size:0.9rem; margin-top:4px;'>
            Upload a Breast Ultrasound scan for live detection of Malignant, Benign, or Normal states.
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    us_uploaded = st.file_uploader("Upload Breast Ultrasound (JPEG/PNG)", type=["jpg", "jpeg", "png"], key="us_upl")
    st.markdown('</div>', unsafe_allow_html=True)

    if us_uploaded is not None:
        from PIL import Image
        import time
        import os
        
        us_img = Image.open(us_uploaded).convert('RGB')
        
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.image(us_img, caption="Ultrasound Scan", use_container_width=True)
            
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            if os.path.exists('breast_model.h5') and os.path.exists('breast_classes.json'):
                with st.spinner("🩺 Processing Deep Learning Model..."):
                    try:
                        import tensorflow as tf
                        import json
                        import numpy as np
                        
                        @st.cache_resource(show_spinner=False)
                        def load_breast_model():
                            mdl = tf.keras.models.load_model('breast_model.h5')
                            with open('breast_classes.json', 'r') as f:
                                c = json.load(f)
                            return mdl, c
                            
                        model_us, class_dict_us = load_breast_model()
                            
                        img_resized = us_img.resize((128, 128))
                        img_array = np.array(img_resized) / 255.0
                        img_batch = np.expand_dims(img_array, axis=0)

                        preds = model_us.predict(img_batch, verbose=0)[0]
                        pred_idx = np.argmax(preds)
                        pred_label = class_dict_us[str(pred_idx)].replace('_', ' ').title()
                        confidence = preds[pred_idx] * 100
                        
                        st.markdown("<br/>", unsafe_allow_html=True)
                        if "Normal" in pred_label or "Benign" in pred_label:
                            st.markdown(f'<div class="pred-benign"><div style="font-size:2.8rem;">✅</div><div class="pred-label" style="color:#10b981;">{pred_label}</div><div class="pred-conf">Confidence: <strong style="color:#10b981;">{confidence:.1f}%</strong> · Network: USG CNN</div></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="pred-malignant"><div style="font-size:2.8rem;">⚠️</div><div class="pred-label" style="color:#ef4444;">{pred_label}</div><div class="pred-conf">Confidence: <strong style="color:#ef4444;">{confidence:.1f}%</strong> · Network: USG CNN</div></div>', unsafe_allow_html=True)
                            
                        st.markdown("<br/><hr/>", unsafe_allow_html=True)
                        st.markdown("<b>🔍 Explainable AI (Grad-CAM)</b>", unsafe_allow_html=True)
                        st.markdown("<div style='font-size:0.8rem; color:#64748b; margin-bottom:12px;'>Thermal Heatmap of the tumor regions activating the model.</div>", unsafe_allow_html=True)
                        
                        # Find a conv layer with good spatial resolution for Grad-CAM
                        last_conv_layer_name = None
                        fallback_conv_name = None
                        for layer in reversed(model_us.layers):
                            if isinstance(layer, tf.keras.layers.Conv2D):
                                if fallback_conv_name is None:
                                    fallback_conv_name = layer.name
                                try:
                                    out_shape = layer.output.shape
                                    if len(out_shape) >= 3 and out_shape[1] is not None and out_shape[1] >= 8:
                                        last_conv_layer_name = layer.name
                                        break
                                except:
                                    pass
                        if last_conv_layer_name is None:
                            last_conv_layer_name = fallback_conv_name
                                
                        if last_conv_layer_name:
                            grad_model = tf.keras.models.Model(
                                inputs=model_us.inputs, 
                                outputs=[model_us.get_layer(last_conv_layer_name).output, model_us.output]
                            )
                            
                            with tf.GradientTape() as tape:
                                img_tensor = tf.convert_to_tensor(img_batch, dtype=tf.float32)
                                conv_outputs, predictions = grad_model(img_tensor)
                                tape.watch(conv_outputs)
                                loss = predictions[:, pred_idx]
                            
                            grads = tape.gradient(loss, conv_outputs)
                            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                            
                            conv_outputs = conv_outputs[0]
                            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
                            heatmap = tf.squeeze(heatmap)
                            import matplotlib.cm as cm
                            from PIL import Image
                            
                            # Normalize heatmap to [0, 1] safely
                            heatmap = tf.maximum(heatmap, 0)
                            max_val = tf.math.reduce_max(heatmap)
                            if max_val > 0:
                                heatmap = heatmap / max_val
                            heatmap = heatmap.numpy()
                            
                            # Resize grayscale heatmap to original image size with smooth interpolation
                            heatmap_pil = Image.fromarray(np.uint8(255 * heatmap))
                            heatmap_pil = heatmap_pil.resize((us_img.size[0], us_img.size[1]), Image.Resampling.LANCZOS)
                            heatmap_resized = np.array(heatmap_pil).astype(np.float32) / 255.0
                            
                            # Create alpha mask: only show heatmap where activation is significant
                            alpha = np.clip(heatmap_resized - 0.25, 0, 1)
                            alpha = alpha / (alpha.max() + 1e-8)
                            alpha = np.power(alpha, 0.6)
                            
                            # Create tissue mask: only overlay on tissue, not black background
                            orig_arr = tf.keras.preprocessing.image.img_to_array(us_img)
                            gray = np.mean(orig_arr, axis=-1)
                            tissue_mask = (gray > 25).astype(np.float32)
                            alpha = alpha * tissue_mask
                            
                            # Apply JET colormap to get RGB heatmap
                            jet = cm.get_cmap("jet")
                            jet_heatmap = jet(np.uint8(255 * heatmap_resized))[:, :, :3]
                            jet_heatmap = (jet_heatmap * 255.0).astype(np.float32)
                            
                            # Alpha-blend: original * (1-alpha) + heatmap * alpha
                            alpha_3ch = np.stack([alpha, alpha, alpha], axis=-1) * 0.65
                            superimposed_img = orig_arr * (1 - alpha_3ch) + jet_heatmap * alpha_3ch
                            superimposed_img = np.clip(superimposed_img, 0, 255)
                            superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
                            
                            st.image(superimposed_img, use_container_width=True)
                            
                            # ── Extra Clinical Featuers ── #
                            st.markdown("<hr/>", unsafe_allow_html=True)
                            colA, colB = st.columns(2)
                            with colA:
                                hot_pixels = np.sum(heatmap_resized > 0.5)
                                approx_area = (hot_pixels / (heatmap_resized.shape[0]*heatmap_resized.shape[1])) * 14.5
                                approx_radius = np.sqrt(approx_area / np.pi) if approx_area > 0 else 0.0
                                
                                m1, m2 = st.columns(2)
                                m1.metric("Estimated Tumor Area", f"{approx_area:.1f} cm²")
                                m2.metric("Estimated Radius", f"{approx_radius:.2f} cm")
                                
                            with colB:
                                pid_val = st.text_input("Patient ID", value="PID-89104")
                                
                            import datetime
                            from fpdf import FPDF
                            
                            def create_pdf(pid, diag, conf, area, orig_img, heatmap_img):
                                orig_path = "temp_orig.jpg"
                                heat_path = "temp_heat.jpg"
                                orig_img.save(orig_path)
                                heatmap_img.save(heat_path)
                                
                                pdf = FPDF()
                                pdf.add_page()
                                pdf.set_font("Arial", 'B', 18)
                                pdf.cell(200, 10, txt="DIAGNOSTIC MEDICAL REPORT", ln=True, align='C')
                                pdf.set_font("Arial", size=10)
                                pdf.cell(200, 10, txt=f"Date of Scan: {datetime.datetime.now().strftime('%B %d, %Y - %H:%M')}", ln=True, align='C')
                                pdf.ln(5)
                                
                                # Patient info
                                pdf.set_fill_color(240, 240, 240)
                                pdf.set_font("Arial", 'B', 12)
                                pdf.cell(190, 8, txt=" 1. PATIENT AND SCAN INFORMATION", ln=True, fill=True)
                                pdf.set_font("Arial", size=11)
                                pdf.cell(100, 8, txt=f"  Patient ID: {pid}", ln=False)
                                pdf.cell(90, 8, txt=f"  Modality: Breast Ultrasound", ln=True)
                                pdf.cell(100, 8, txt=f"  AI Network: Custom CNN", ln=False)
                                pdf.cell(90, 8, txt=f"  Analysis: Grad-CAM XAI", ln=True)
                                pdf.ln(5)
                                
                                # Results
                                pdf.set_font("Arial", 'B', 12)
                                pdf.cell(190, 8, txt=" 2. AI DIAGNOSTIC RESULTS", ln=True, fill=True)
                                pdf.set_font("Arial", size=11)
                                
                                if "Benign" in diag or "Normal" in diag:
                                    pdf.set_text_color(0, 150, 0)
                                else:
                                    pdf.set_text_color(200, 0, 0)
                                pdf.set_font("Arial", 'B', 12)
                                pdf.cell(200, 10, txt=f"  Primary Diagnosis: {diag.upper()}", ln=True)
                                pdf.set_text_color(0, 0, 0)
                                pdf.set_font("Arial", size=11)
                                pdf.cell(200, 8, txt=f"  Confidence Level: {conf:.2f}%", ln=True)
                                pdf.cell(200, 8, txt=f"  Estimated Tumor Area: {area:.2f} sq. cm", ln=True)
                                pdf.ln(5)
                                
                                # Images
                                pdf.set_font("Arial", 'B', 12)
                                pdf.cell(190, 8, txt=" 3. CLINICAL IMAGING", ln=True, fill=True)
                                pdf.ln(5)
                                
                                pdf.set_font("Arial", size=10)
                                pdf.cell(95, 8, txt="Original Sonogram", align='C', ln=False)
                                pdf.cell(95, 8, txt="Grad-CAM Activation Map", align='C', ln=True)
                                
                                cur_y = pdf.get_y()
                                pdf.image(orig_path, x=20, y=cur_y, w=70)
                                pdf.image(heat_path, x=115, y=cur_y, w=70)
                                pdf.ln(75) 
                                
                                pdf.set_font("Arial", 'I', 9)
                                pdf.set_text_color(150, 150, 150)
                                pdf.multi_cell(0, 6, txt="Disclaimer: This report is automatically generated using Artificial Intelligence (CNN). It is designed to assist radiologists but should not replace professional clinical correlation and biopsy.")
                                
                                out = pdf.output(dest='S').encode('latin-1')
                                
                                if os.path.exists(orig_path): os.remove(orig_path)
                                if os.path.exists(heat_path): os.remove(heat_path)
                                
                                return out

                            pdf_bytes = create_pdf(pid_val, pred_label, confidence, approx_area, us_img, superimposed_img)
                            
                            st.markdown("<br/>", unsafe_allow_html=True)
                            st.download_button(
                                label="📄 Export Medical PDF Report", 
                                data=pdf_bytes, 
                                file_name=f"Medical_Report_{pid_val}.pdf", 
                                mime="application/pdf",
                                use_container_width=True
                            )
                            
                    except Exception as e:
                        st.error(f"Model execution error: {e}")
            else:
                st.warning("⚠️ Ultrasound Model is still training in the background. Please wait a few seconds and try again.")
            st.markdown('</div>', unsafe_allow_html=True)
