# A Comparative Study on Tumor Classification

A deep learning-powered Tumor Classification system for MRI and Ultrasound scans. Leveraging CNNs for high-accuracy diagnostics and Grad-CAM for Explainable AI (XAI) to visualize tumor localization. Features an interactive Streamlit dashboard for real-time medical image analysis, bridging the gap between AI and clinical interpretability.

---

## Models Compared
| Model | Tuned? |
|---|---|
| Logistic Regression | ✗ (default) |
| SVM | ✓ GridSearchCV |
| K-Nearest Neighbors | ✗ (default) |
| Decision Tree | ✗ (default) |
| Random Forest | ✓ GridSearchCV |

---

## Metrics Evaluated
- Accuracy · Precision · Recall · F1-Score
- Confusion Matrix (best model)
- ROC Curve + AUC Score

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the dashboard
```bash
streamlit run app.py
```

### 3. Outputs
All plots are saved to the `outputs/` folder:
- `model_comparison.png` — Bar chart of all models
- `confusion_matrix.png` — Heatmap for best model
- `roc_curve.png`        — ROC curve for best model
- `feature_importance.png` — Top-15 features from Random Forest

---

## Key Highlights
- **Stratified 80-20 split** preserves class balance
- **GridSearchCV (5-fold CV)** for hyperparameter tuning
- **Best model selected** by F1-Score + Recall
- **Explainable AI** — Grad-CAM heatmaps for tumor localization
