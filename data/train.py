import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# =========================================================
# 0. CONFIGURATION
# =========================================================
DATA_PATH = r"C:/Users/enock/Downloads/aihw/data/daily_summary_new.csv"
OUTPUT_DIR = r"C:/Users/enock/Downloads/aihw/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 1. LOAD DATA
# =========================================================
print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

print("\nLoaded dataset shape:", df.shape)
print(df.head())

# =========================================================
# 2. CREATE LABEL
# =========================================================
# Label = 1 if total_screen_time_min > 75th percentile
threshold = df["total_screen_time_min"].quantile(0.75)
df["high_usage_label"] = (df["total_screen_time_min"] > threshold).astype(int)

print("\n75th percentile threshold:", threshold)
print("\nLabel counts:")
print(df["high_usage_label"].value_counts())

# Save threshold info
with open(os.path.join(OUTPUT_DIR, "label_threshold.txt"), "w", encoding="utf-8") as f:
    f.write(f"75th percentile threshold for total_screen_time_min: {threshold}\n")

# =========================================================
# 3. FEATURE SELECTION
# =========================================================
# IMPORTANT:
# Since label is derived from total_screen_time_min,
# do NOT use total_screen_time_min as a feature.
# Also exclude raw text top_app names.
feature_cols = [
    "screen_interactive_min",
    "unlock_count_proxy",
    "top_app_1_min",
    "top_app_2_min",
    "top_app_3_min",
    "social_media_min",
    "messaging_min",
    "video_min",
    "browser_min",
    "productivity_min",
    "education_min",
    "gaming_min"
]

# Keep only columns that actually exist
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols].copy()
y = df["high_usage_label"].copy()

print("\nFeature columns used:")
print(feature_cols)

# =========================================================
# 4. VISUAL: CLASS BALANCE
# =========================================================
plt.figure(figsize=(6, 4))
sns.countplot(x="high_usage_label", data=df, palette="Set2")
plt.title("Class Balance: High Usage vs Normal Usage")
plt.xlabel("High Usage Label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_balance.png"), dpi=300)
plt.close()

# =========================================================
# 5. VISUAL: TOTAL SCREEN TIME DISTRIBUTION
# =========================================================
plt.figure(figsize=(8, 5))
sns.histplot(df["total_screen_time_min"], bins=25, kde=True, color="steelblue")
plt.axvline(threshold, color="red", linestyle="--", label=f"75th percentile = {threshold:.1f}")
plt.title("Distribution of Total Screen Time (Minutes)")
plt.xlabel("Total Screen Time (min)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "total_screen_time_histogram.png"), dpi=300)
plt.close()

# =========================================================
# 6. VISUAL: CORRELATION HEATMAP
# =========================================================
corr_cols = feature_cols + ["total_screen_time_min", "high_usage_label"]
corr_cols = [c for c in corr_cols if c in df.columns]
corr = df[corr_cols].corr(numeric_only=True)

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=False)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=300)
plt.close()

# =========================================================
# 7. DEFINE MODELS
# =========================================================
logreg = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

rf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ))
])

models = {
    "LogisticRegression": logreg,
    "RandomForest": rf
}

# =========================================================
# 8. CROSS-VALIDATION SETTINGS
# =========================================================
min_class_count = y.value_counts().min()
n_splits = min(5, min_class_count)

if n_splits < 2:
    raise ValueError("Not enough samples to perform cross-validation.")

cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

# =========================================================
# 9. EVALUATE MODELS
# =========================================================
print("\n====================")
print("BASE CROSS-VALIDATION RESULTS")
print("====================")

summary_rows = []

for model_name, model in models.items():
    print(f"\n--- Evaluating {model_name} ---")

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        error_score=np.nan
    )

    y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")
    y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y, y_proba)
    except Exception:
        auc = np.nan

    cm = confusion_matrix(y, y_pred)

    print("Mean CV Accuracy :", np.nanmean(cv_results["test_accuracy"]))
    print("Mean CV Precision:", np.nanmean(cv_results["test_precision"]))
    print("Mean CV Recall   :", np.nanmean(cv_results["test_recall"]))
    print("Mean CV F1       :", np.nanmean(cv_results["test_f1"]))
    print("Mean CV AUROC    :", np.nanmean(cv_results["test_roc_auc"]))
    print("Confusion Matrix :")
    print(cm)

    summary_rows.append({
        "Model": model_name,
        "CV_Accuracy": np.nanmean(cv_results["test_accuracy"]),
        "CV_Precision": np.nanmean(cv_results["test_precision"]),
        "CV_Recall": np.nanmean(cv_results["test_recall"]),
        "CV_F1": np.nanmean(cv_results["test_f1"]),
        "CV_AUROC": np.nanmean(cv_results["test_roc_auc"]),
        "Overall_Accuracy": acc,
        "Overall_Precision": prec,
        "Overall_Recall": rec,
        "Overall_F1": f1,
        "Overall_AUROC": auc
    })

    # -----------------------------------------------------
    # Save confusion matrix plot
    # -----------------------------------------------------
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name}.png"), dpi=300)
    plt.close()

    # -----------------------------------------------------
    # Save ROC curve
    # -----------------------------------------------------
    try:
        fpr, tpr, _ = roc_curve(y, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title(f"ROC Curve - {model_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"roc_curve_{model_name}.png"), dpi=300)
        plt.close()
    except Exception:
        pass

# Save summary table
results_df = pd.DataFrame(summary_rows)
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_results_summary.csv"), index=False)

print("\nSummary table:")
print(results_df)

# =========================================================
# 10. RANDOM FOREST FEATURE IMPORTANCE
# =========================================================
print("\n====================")
print("RANDOM FOREST FEATURE IMPORTANCE")
print("====================")

# Fit RF on full dataset to get feature importance
rf.fit(X, y)
rf_model = rf.named_steps["model"]

importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(feature_importance_df)

feature_importance_df.to_csv(
    os.path.join(OUTPUT_DIR, "random_forest_feature_importance.csv"),
    index=False
)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df, x="Importance", y="Feature", palette="viridis")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "random_forest_feature_importance.png"), dpi=300)
plt.close()

# =========================================================
# 11. TRAINING DATA SIZE EXPERIMENT
# =========================================================
print("\n====================")
print("TRAINING DATA SIZE EXPERIMENT")
print("====================")

fractions = [0.3, 0.5, 0.7, 1.0]
experiment_rows = []

for frac in fractions:
    n = max(10, int(len(df) * frac))
    sampled_df = df.sample(n=n, random_state=42)

    X_sub = sampled_df[feature_cols].copy()
    y_sub = sampled_df["high_usage_label"].copy()

    # Skip if too small or only one class
    if y_sub.nunique() < 2:
        print(f"Fraction {frac}: skipped (only one class)")
        continue

    min_class_sub = y_sub.value_counts().min()
    sub_splits = min(5, min_class_sub)

    if sub_splits < 2:
        print(f"Fraction {frac}: skipped (not enough samples per class)")
        continue

    cv_sub = StratifiedKFold(n_splits=sub_splits, shuffle=True, random_state=42)

    for model_name, model in models.items():
        cv_results = cross_validate(
            model,
            X_sub,
            y_sub,
            cv=cv_sub,
            scoring=scoring,
            return_train_score=False,
            error_score=np.nan
        )

        acc = np.nanmean(cv_results["test_accuracy"])
        f1 = np.nanmean(cv_results["test_f1"])
        auc = np.nanmean(cv_results["test_roc_auc"])

        print(f"Fraction={frac}, Model={model_name}, Acc={acc:.3f}, F1={f1:.3f}, AUROC={auc:.3f}")

        experiment_rows.append({
            "Fraction": frac,
            "Samples": n,
            "Model": model_name,
            "CV_Accuracy": acc,
            "CV_F1": f1,
            "CV_AUROC": auc
        })

experiment_df = pd.DataFrame(experiment_rows)
experiment_df.to_csv(os.path.join(OUTPUT_DIR, "training_size_experiment.csv"), index=False)

# Plot training size experiment
plt.figure(figsize=(8, 5))
sns.lineplot(data=experiment_df, x="Fraction", y="CV_F1", hue="Model", marker="o")
plt.title("Training Data Size Experiment (CV F1)")
plt.xlabel("Fraction of Dataset Used")
plt.ylabel("Cross-Validated F1")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_size_experiment_f1.png"), dpi=300)
plt.close()

# =========================================================
# 12. OPTIONAL: BOXPLOT OF FEATURES BY LABEL
# =========================================================
# Example boxplot for a few informative columns
plot_cols = [c for c in ["screen_interactive_min", "unlock_count_proxy", "social_media_min", "messaging_min"] if c in df.columns]

for col in plot_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="high_usage_label", y=col, data=df, palette="Set3")
    plt.title(f"{col} by Label")
    plt.xlabel("High Usage Label")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"boxplot_{col}.png"), dpi=300)
    plt.close()

print("\nSUCCESS: Training complete.")
print(f"All results and visuals saved in: {OUTPUT_DIR}")