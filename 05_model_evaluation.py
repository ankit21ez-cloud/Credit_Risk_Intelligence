
# ============================================================
# CREDIT RISK INTELLIGENCE
# 05 - MODEL EVALUATION
# ============================================================


# ------------------------------------------------------------
# STEP 1: IMPORT LIBRARIES
# ------------------------------------------------------------

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    balanced_accuracy_score,
    matthews_corrcoef
)


# ------------------------------------------------------------
# STEP 2: CREATE RESULTS FOLDER
# ------------------------------------------------------------

os.makedirs("results", exist_ok=True)

print("\n" + "=" * 60)
print("CREDIT RISK INTELLIGENCE - MODEL EVALUATION")
print("=" * 60)


# ------------------------------------------------------------
# STEP 3: LOAD BEST MODEL
# ------------------------------------------------------------

print("\nLoading Best Model...")

best_model_data = joblib.load("model/best_model.pkl")

model = best_model_data["model"]
model_name = best_model_data["model_name"]
threshold = best_model_data["threshold"]
feature_names = best_model_data["feature_names"]

print("\nBest Model Loaded Successfully!")
print("Model Name:", model_name)
print("Selected Threshold:", threshold)


# ------------------------------------------------------------
# STEP 4: LOAD VALIDATION DATASET
# ------------------------------------------------------------

print("\nLoading Validation Dataset...")

data = joblib.load("model/feature_engineering_output.pkl")

X_valid = data["X_valid"]
y_valid = data["y_valid"]

# Convert target into NumPy array
y_valid = np.asarray(y_valid).ravel()

print("\nValidation Dataset Loaded!")
print("Validation Features:", X_valid.shape)
print("Validation Target:", y_valid.shape)


# ------------------------------------------------------------
# STEP 5: PREDICT PROBABILITIES
# ------------------------------------------------------------

print("\nGenerating Predictions...")

# Get probability of default class (1)
y_probability = model.predict_proba(X_valid)[:, 1]

# Apply selected threshold
y_prediction = (y_probability >= threshold).astype(int)

print("Predictions Generated Successfully!")


# ------------------------------------------------------------
# STEP 6: BASIC MODEL EVALUATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("BASIC MODEL EVALUATION")
print("=" * 60)

accuracy = accuracy_score(y_valid, y_prediction)

precision = precision_score(
    y_valid, y_prediction, zero_division=0
)

recall = recall_score(
    y_valid, y_prediction, zero_division=0
)

f1 = f1_score(
    y_valid, y_prediction, zero_division=0
)

roc_auc = roc_auc_score(y_valid, y_probability)

pr_auc = average_precision_score(y_valid, y_probability)

print("\nAccuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))
print("PR-AUC:", round(pr_auc, 4))


# ------------------------------------------------------------
# STEP 7: ADVANCED EVALUATION METRICS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("ADVANCED EVALUATION METRICS")
print("=" * 60)

# Confusion Matrix
cm = confusion_matrix(y_valid, y_prediction)

TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TP = cm[1, 1]

# Specificity
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

# Balanced Accuracy
balanced_acc = balanced_accuracy_score(
    y_valid, y_prediction
)

# Matthews Correlation Coefficient
mcc = matthews_corrcoef(y_valid, y_prediction)

print("\nTrue Negative:", TN)
print("False Positive:", FP)
print("False Negative:", FN)
print("True Positive:", TP)

print("\nSpecificity:", round(specificity, 4))
print("Balanced Accuracy:", round(balanced_acc, 4))
print("Matthews Correlation Coefficient:", round(mcc, 4))


# ------------------------------------------------------------
# STEP 8: CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_valid,
    y_prediction,
    target_names=["Non-Default", "Default"],
    zero_division=0
)

print(report)

# Save classification report
with open("results/classification_report.txt", "w") as file:
    file.write(report)


# ------------------------------------------------------------
# STEP 9: SAVE EVALUATION METRICS
# ------------------------------------------------------------

evaluation_results = {
    "Model": model_name,
    "Threshold": threshold,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "ROC_AUC": roc_auc,
    "PR_AUC": pr_auc,
    "Specificity": specificity,
    "Balanced_Accuracy": balanced_acc,
    "MCC": mcc,
    "True_Negative": TN,
    "False_Positive": FP,
    "False_Negative": FN,
    "True_Positive": TP
}

evaluation_df = pd.DataFrame([evaluation_results])

evaluation_df.to_csv(
    "results/model_evaluation_metrics.csv",
    index=False
)

print("\nEvaluation Metrics Saved Successfully!")


# ------------------------------------------------------------
# STEP 10: CONFUSION MATRIX HEATMAP
# ------------------------------------------------------------

print("\nGenerating Confusion Matrix...")

plt.figure(figsize=(7, 5))

plt.imshow(cm, interpolation="nearest", cmap="Blues")

plt.title("Confusion Matrix - " + model_name)
plt.colorbar()

plt.xticks([0, 1], ["Non-Default", "Default"])
plt.yticks([0, 1], ["Non-Default", "Default"])

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

# Display values inside matrix
for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="black",
            fontsize=13
        )

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.close()

print("Confusion Matrix Saved!")


# ------------------------------------------------------------
# STEP 11: ROC CURVE
# ------------------------------------------------------------

print("\nGenerating ROC Curve...")

fpr, tpr, roc_thresholds = roc_curve(
    y_valid,
    y_probability
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"{model_name} (AUC = {roc_auc:.4f})"
)

# Random classifier reference line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "results/roc_curve.png",
    dpi=300
)

plt.close()

print("ROC Curve Saved!")


# ------------------------------------------------------------
# STEP 12: PRECISION-RECALL CURVE
# ------------------------------------------------------------

print("\nGenerating Precision-Recall Curve...")

pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
    y_valid,
    y_probability
)

plt.figure(figsize=(8, 6))

plt.plot(
    pr_recall,
    pr_precision,
    label=f"{model_name} (PR-AUC = {pr_auc:.4f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision-Recall Curve")

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "results/precision_recall_curve.png",
    dpi=300
)

plt.close()

print("Precision-Recall Curve Saved!")


# ------------------------------------------------------------
# STEP 13: FEATURE IMPORTANCE
# ------------------------------------------------------------

print("\nGenerating Feature Importance...")

# XGBoost provides feature_importances_
if hasattr(model, "feature_importances_"):

    importance_values = model.feature_importances_

# Logistic Regression provides coefficients
elif hasattr(model, "coef_"):

    importance_values = np.abs(model.coef_[0])

else:

    importance_values = None


if importance_values is not None:

    # Match feature names with importance values
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values
    })

    # Sort from highest to lowest
    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    # Save all feature importance
    importance_df.to_csv(
        "results/feature_importance.csv",
        index=False
    )

    # Select top 20 features
    top_features = importance_df.head(20)

    plt.figure(figsize=(10, 8))

    plt.barh(
        top_features["Feature"][::-1],
        top_features["Importance"][::-1]
    )

    plt.xlabel("Importance")
    plt.ylabel("Features")

    plt.title("Top 20 Important Features")

    plt.tight_layout()

    plt.savefig(
        "results/feature_importance.png",
        dpi=300
    )

    plt.close()

    print("\nTop 20 Features:")
    print(top_features.to_string(index=False))

    print("\nFeature Importance Saved!")

else:

    print("Feature Importance is not available for this model.")


# ------------------------------------------------------------
# STEP 14: PREDICTION PROBABILITY DISTRIBUTION
# ------------------------------------------------------------

print("\nGenerating Probability Distribution...")

plt.figure(figsize=(8, 6))

plt.hist(
    y_probability[y_valid == 0],
    bins=40,
    alpha=0.6,
    label="Non-Default"
)

plt.hist(
    y_probability[y_valid == 1],
    bins=40,
    alpha=0.6,
    label="Default"
)

plt.axvline(
    threshold,
    linestyle="--",
    label=f"Threshold = {threshold:.4f}"
)

plt.xlabel("Predicted Default Probability")
plt.ylabel("Number of Customers")

plt.title("Prediction Probability Distribution")

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "results/probability_distribution.png",
    dpi=300
)

plt.close()

print("Probability Distribution Saved!")


# ------------------------------------------------------------
# STEP 15: FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION SUMMARY")
print("=" * 60)

print("\nBest Model:", model_name)
print("Selected Threshold:", round(threshold, 4))

print("\nAccuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))
print("PR-AUC:", round(pr_auc, 4))

print("\nAll Evaluation Files Saved Inside Results Folder!")

print("\n" + "=" * 60)
print("05 MODEL EVALUATION COMPLETED SUCCESSFULLY!")
print("=" * 60)
