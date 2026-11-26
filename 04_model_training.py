import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)

print("\nLoading Processed Dataset...")

data = joblib.load(
    "model/feature_engineering_output.pkl"
)

X_train = data["X_train"]
X_valid = data["X_valid"]

y_train = data["y_train"]
y_valid = data["y_valid"]

feature_names = data["feature_names"]

print("\nDataset Loaded Successfully!")

print("Training Shape:", X_train.shape)
print("Validation Shape:", X_valid.shape)


print("\nTarget Distribution:")

print(y_train.value_counts())

print("\nTarget Percentage:")

print(
    y_train.value_counts(normalize=True) * 100
)

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\nClass Imbalance Information:")

print("Non-Default Customers:", negative_count)
print("Default Customers:", positive_count)

print("Scale Pos Weight:", scale_pos_weight)

models = {}

models["Logistic Regression"] = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

models["Random Forest"] = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)


try:

    from xgboost import XGBClassifier

    models["XGBoost"] = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        reg_lambda=1,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    print("\nXGBoost Available!")

except ImportError:

    print("\nXGBoost Not Installed!")
    print("Other models will continue training.")


results = []
trained_models = {}

os.makedirs("model", exist_ok=True)

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    y_pred = model.predict(X_valid)

    y_probability = model.predict_proba(X_valid)[:, 1]

    accuracy = accuracy_score(y_valid, y_pred)

    precision = precision_score(
        y_valid,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_valid,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_valid,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_valid,
        y_probability
    )

    pr_auc = average_precision_score(
        y_valid,
        y_probability
    )

    cm = confusion_matrix(
        y_valid,
        y_pred
    )

    print("\nDefault Threshold: 0.50")

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))
    print("PR-AUC:", round(pr_auc, 4))

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y_valid,
            y_pred,
            zero_division=0
        )
    )


    precision_values, recall_values, thresholds = (
        precision_recall_curve(
            y_valid,
            y_probability
        )
    )

    f1_values = (
        2 * precision_values[:-1] * recall_values[:-1]
        /
        (
            precision_values[:-1]
            + recall_values[:-1]
            + 1e-10
        )
    )

    best_index = np.argmax(f1_values)

    best_threshold = thresholds[best_index]

    tuned_pred = (
        y_probability >= best_threshold
    ).astype(int)

    tuned_precision = precision_score(
        y_valid,
        tuned_pred,
        zero_division=0
    )

    tuned_recall = recall_score(
        y_valid,
        tuned_pred,
        zero_division=0
    )

    tuned_f1 = f1_score(
        y_valid,
        tuned_pred,
        zero_division=0
    )

    print("\nThreshold Tuning Results:")

    print(
        "Best Threshold:",
        round(best_threshold, 4)
    )

    print(
        "Tuned Precision:",
        round(tuned_precision, 4)
    )

    print(
        "Tuned Recall:",
        round(tuned_recall, 4)
    )

    print(
        "Tuned F1 Score:",
        round(tuned_f1, 4)
    )

    # Tuned Confusion Matrix
    print("\nTuned Confusion Matrix:")

    print(
        confusion_matrix(
            y_valid,
            tuned_pred
        )
    )

    model_file_name = (
        model_name.lower().replace(" ", "_")
        + ".pkl"
    )

    joblib.dump(
        model,
        os.path.join("model", model_file_name)
    )

    trained_models[model_name] = model

    print("\nModel Saved:", model_file_name)

    print(
        "Training Time:",
        round(training_time, 2),
        "seconds"
    )


    results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1_Score": f1,

        "ROC_AUC": roc_auc,

        "PR_AUC": pr_auc,

        "Best_Threshold": best_threshold,

        "Tuned_Precision": tuned_precision,

        "Tuned_Recall": tuned_recall,

        "Tuned_F1": tuned_f1,

        "Training_Time": training_time

    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="PR_AUC",
    ascending=False
)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(index=False)
)


best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

best_threshold = results_df.iloc[0]["Best_Threshold"]

print("\nBest Model:", best_model_name)

print(
    "Best Model PR-AUC:",
    round(results_df.iloc[0]["PR_AUC"], 4)
)

print(
    "Selected Threshold:",
    round(best_threshold, 4)
)


results_df.to_csv(
    "model/model_comparison.csv",
    index=False
)

print("\nModel Comparison Saved!")


best_model_output = {

    "model": best_model,

    "model_name": best_model_name,

    "threshold": best_threshold,

    "feature_names": feature_names,

    "preprocessor": data["preprocessor"]

}

joblib.dump(
    best_model_output,
    "model/best_model.pkl"
)

print("\nBest Model Saved Successfully!")

print("File: model/best_model.pkl")

print("\n" + "=" * 60)

print("04 MODEL TRAINING COMPLETED SUCCESSFULLY!")

print("=" * 60)
