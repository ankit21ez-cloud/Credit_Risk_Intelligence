
import os
import joblib
import numpy as np
import pandas as pd

os.makedirs("results", exist_ok=True)

print("\nLoading Prediction System...")

best_model_data = joblib.load("model/best_model.pkl")

model = best_model_data["model"]
model_name = best_model_data["model_name"]
threshold = best_model_data["threshold"]
feature_names = best_model_data["feature_names"]

print("\nModel Loaded Successfully!")
print("Model Name:", model_name)
print("Selected Threshold:", threshold)

data = joblib.load("model/feature_engineering_output.pkl")

X_test = np.asarray(data["X_test"], dtype=np.float64)

test_df = pd.read_csv("dataset/application_test.csv")

test_ids = test_df["SK_ID_CURR"].reset_index(drop=True)

print("\nTest Dataset Loaded Successfully!")
print("Test Features Shape:", X_test.shape)
print("Total Applicants:", len(X_test))

missing_values = np.isnan(X_test).sum()
infinite_values = np.isinf(X_test).sum()

print("\nData Validation Results:")
print("Missing Values:", missing_values)
print("Infinite Values:", infinite_values)

if X_test.ndim != 2:
    raise ValueError("Test dataset must be a two-dimensional array!")

if X_test.shape[1] != len(feature_names):
    raise ValueError("Feature count does not match trained model!")

if len(test_ids) != len(X_test):
    raise ValueError("Applicant IDs and test dataset rows do not match!")

if test_ids.duplicated().any():
    raise ValueError("Duplicate applicant IDs found in test dataset!")

if missing_values > 0 or infinite_values > 0:
    raise ValueError("Test dataset contains missing or infinite values!")

print("\nData Validation Completed!")

if hasattr(model, "n_features_in_"):
    if model.n_features_in_ != X_test.shape[1]:
        raise ValueError("Model and test dataset feature count mismatch!")

if 1 not in model.classes_:
    raise ValueError("Default class (1) not found in trained model!")

if not 0 < threshold < 1:
    raise ValueError("Invalid prediction threshold!")

print("\nModel Feature Validation Successful!")

print("\nGenerating Default Risk Probabilities...")

class_index = list(model.classes_).index(1)

y_probability = model.predict_proba(X_test)[:, class_index]

if not np.isfinite(y_probability).all():
    raise ValueError("Model generated invalid probability values!")

y_prediction = (y_probability >= threshold).astype(int)

print("Probability Prediction Completed!")
print("Final Predictions Generated!")

low_risk_threshold = threshold * 0.50

risk_category = np.where(
    y_probability < low_risk_threshold,
    "Low Risk",
    np.where(
        y_probability < threshold,
        "Medium Risk",
        "High Risk"
    )
)

prediction_labels = np.where(
    y_prediction == 1,
    "Default",
    "Non-Default"
)

prediction_results = pd.DataFrame({

    "SK_ID_CURR": test_ids,

    "Default_Probability": y_probability,

    "Default_Risk_Percentage": y_probability * 100,

    "Predicted_Class": y_prediction,

    "Prediction_Label": prediction_labels,

    "Risk_Category": risk_category

})

prediction_results["Default_Probability"] = (
    prediction_results["Default_Probability"].round(4)
)

prediction_results["Default_Risk_Percentage"] = (
    prediction_results["Default_Risk_Percentage"].round(2)
)

prediction_results = prediction_results.sort_values(
    by="Default_Risk_Percentage",
    ascending=False
).reset_index(drop=True)

prediction_results.to_csv(
    "results/test_predictions.csv",
    index=False
)

print("\nAll Predictions Saved Successfully!")

high_risk_applicants = prediction_results[
    prediction_results["Risk_Category"] == "High Risk"
].copy()

high_risk_applicants.to_csv(
    "results/high_risk_applicants.csv",
    index=False
)

print("High Risk Applicants Saved Successfully!")

risk_summary = (
    prediction_results["Risk_Category"]
    .value_counts()
    .reindex(
        ["Low Risk", "Medium Risk", "High Risk"],
        fill_value=0
    )
)

risk_summary_df = pd.DataFrame({

    "Risk_Category": risk_summary.index,

    "Total_Applicants": risk_summary.values

})

risk_summary_df["Percentage"] = (
    risk_summary_df["Total_Applicants"] / len(prediction_results)
) * 100

risk_summary_df["Percentage"] = (
    risk_summary_df["Percentage"].round(2)
)

total_applicants = len(prediction_results)

total_default = int((y_prediction == 1).sum())

total_non_default = int((y_prediction == 0).sum())

default_percentage = (
    total_default / total_applicants
) * 100

average_risk = np.mean(y_probability) * 100

summary_data = {

    "Metric": [

        "Model Name",
        "Total Applicants",
        "Predicted Default",
        "Predicted Non-Default",
        "Predicted Default Percentage",
        "Average Default Probability",
        "Selected Threshold",
        "High Risk Applicants",
        "Medium Risk Applicants",
        "Low Risk Applicants"

    ],

    "Value": [

        model_name,
        total_applicants,
        total_default,
        total_non_default,
        round(default_percentage, 2),
        round(average_risk, 2),
        round(threshold, 4),
        int(risk_summary["High Risk"]),
        int(risk_summary["Medium Risk"]),
        int(risk_summary["Low Risk"])

    ]

}

summary_df = pd.DataFrame(summary_data)

risk_summary_df.to_csv(
    "results/risk_category_summary.csv",
    index=False
)

summary_df.to_csv(
    "results/prediction_summary.csv",
    index=False
)

with open(
    "results/prediction_summary.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("CREDIT RISK INTELLIGENCE\n")
    file.write("=======================\n\n")
    file.write(summary_df.to_string(index=False))
    file.write("\n\nRISK CATEGORY SUMMARY\n")
    file.write("=====================\n\n")
    file.write(risk_summary_df.to_string(index=False))

print("\nTOP 10 HIGH RISK APPLICANTS")
print("===========================")

print(
    prediction_results[
        [
            "SK_ID_CURR",
            "Default_Risk_Percentage",
            "Prediction_Label",
            "Risk_Category"
        ]
    ].head(10).to_string(index=False)
)

print("\nRISK CATEGORY SUMMARY")
print("=====================")
print(risk_summary_df.to_string(index=False))

print("\nPREDICTION SUMMARY")
print("==================")
print(summary_df.to_string(index=False))

print("\n======================================")
print("CREDIT RISK PREDICTION COMPLETED!")
print("======================================")

print("\nGenerated Files:")
print("1. results/test_predictions.csv")
print("2. results/high_risk_applicants.csv")
print("3. results/risk_category_summary.csv")
print("4. results/prediction_summary.csv")
print("5. results/prediction_summary.txt")

print("\nTotal Applicants:", total_applicants)
print("Predicted Default:", total_default)
print("Predicted Non-Default:", total_non_default)

print("\n06 MODEL PREDICTION COMPLETED SUCCESSFULLY!")
