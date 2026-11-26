
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from scipy import sparse


st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="📊",
    layout="wide"
)


@st.cache_resource
def load_model():
    if not os.path.exists("model/best_model.pkl"):
        raise FileNotFoundError(
            "model/best_model.pkl not found."
        )

    model_data = joblib.load("model/best_model.pkl")

    model = model_data["model"]
    model_name = model_data["model_name"]
    threshold = float(model_data["threshold"])
    preprocessor = model_data["preprocessor"]

    return model, model_name, threshold, preprocessor


def create_features(data):

    data = data.copy()

    if "DAYS_EMPLOYED" in data.columns:
        data["DAYS_EMPLOYED"] = data["DAYS_EMPLOYED"].replace(
            365243, np.nan
        )

    if "DAYS_BIRTH" in data.columns:
        data["AGE_YEARS"] = data["DAYS_BIRTH"].abs() / 365

    if "DAYS_EMPLOYED" in data.columns:
        data["EMPLOYED_YEARS"] = (
            data["DAYS_EMPLOYED"].abs() / 365
        )

    if "DAYS_REGISTRATION" in data.columns:
        data["REGISTRATION_YEARS"] = (
            data["DAYS_REGISTRATION"].abs() / 365
        )

    if "DAYS_ID_PUBLISH" in data.columns:
        data["ID_PUBLISH_YEARS"] = (
            data["DAYS_ID_PUBLISH"].abs() / 365
        )

    def safe_divide(numerator, denominator):

        result = numerator / denominator

        return result.replace(
            [np.inf, -np.inf],
            np.nan
        )

    if all(col in data.columns for col in [
        "AMT_CREDIT", "AMT_INCOME_TOTAL"
    ]):
        data["CREDIT_INCOME_RATIO"] = safe_divide(
            data["AMT_CREDIT"],
            data["AMT_INCOME_TOTAL"]
        )

    if all(col in data.columns for col in [
        "AMT_ANNUITY", "AMT_INCOME_TOTAL"
    ]):
        data["ANNUITY_INCOME_RATIO"] = safe_divide(
            data["AMT_ANNUITY"],
            data["AMT_INCOME_TOTAL"]
        )

    if all(col in data.columns for col in [
        "AMT_CREDIT", "AMT_ANNUITY"
    ]):
        data["CREDIT_TERM"] = safe_divide(
            data["AMT_CREDIT"],
            data["AMT_ANNUITY"]
        )

    if all(col in data.columns for col in [
        "AMT_ANNUITY", "AMT_CREDIT"
    ]):
        data["PAYMENT_RATE"] = safe_divide(
            data["AMT_ANNUITY"],
            data["AMT_CREDIT"]
        )

    if all(col in data.columns for col in [
        "AMT_GOODS_PRICE", "AMT_CREDIT"
    ]):
        data["GOODS_CREDIT_RATIO"] = safe_divide(
            data["AMT_GOODS_PRICE"],
            data["AMT_CREDIT"]
        )

    if all(col in data.columns for col in [
        "AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"
    ]):
        data["INCOME_PER_PERSON"] = safe_divide(
            data["AMT_INCOME_TOTAL"],
            data["CNT_FAM_MEMBERS"]
        )

    if all(col in data.columns for col in [
        "AMT_INCOME_TOTAL", "CNT_CHILDREN"
    ]):
        data["INCOME_PER_CHILD"] = safe_divide(
            data["AMT_INCOME_TOTAL"],
            data["CNT_CHILDREN"] + 1
        )

    if all(col in data.columns for col in [
        "EMPLOYED_YEARS", "AGE_YEARS"
    ]):
        data["EMPLOYMENT_AGE_RATIO"] = safe_divide(
            data["EMPLOYED_YEARS"],
            data["AGE_YEARS"]
        )

    external_columns = [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3"
    ]

    available_external = [
        col for col in external_columns
        if col in data.columns
    ]

    if len(available_external) > 0:

        data["EXT_SOURCE_MEAN"] = (
            data[available_external].mean(axis=1)
        )

        data["EXT_SOURCE_MIN"] = (
            data[available_external].min(axis=1)
        )

        data["EXT_SOURCE_MAX"] = (
            data[available_external].max(axis=1)
        )

        data["EXT_SOURCE_STD"] = (
            data[available_external].std(axis=1)
        )

    document_columns = [
        col for col in data.columns
        if col.startswith("FLAG_DOCUMENT_")
    ]

    if len(document_columns) > 0:
        data["DOCUMENT_COUNT"] = (
            data[document_columns].sum(axis=1)
        )

    contact_columns = [
        "FLAG_MOBIL",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_CONT_MOBILE",
        "FLAG_PHONE",
        "FLAG_EMAIL"
    ]

    available_contact = [
        col for col in contact_columns
        if col in data.columns
    ]

    if len(available_contact) > 0:
        data["CONTACT_INFORMATION_COUNT"] = (
            data[available_contact].sum(axis=1)
        )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return data


def make_predictions(uploaded_data, model, threshold, preprocessor):

    data = uploaded_data.copy()

    applicant_ids = (
        data["SK_ID_CURR"].reset_index(drop=True)
        if "SK_ID_CURR" in data.columns
        else pd.Series(
            np.arange(1, len(data) + 1),
            name="Applicant_ID"
        )
    )

    data = data.drop(
        columns=["TARGET", "SK_ID_CURR"],
        errors="ignore"
    )

    data = create_features(data)

    expected_features = list(preprocessor.feature_names_in_)

    missing_columns = [
        col for col in expected_features
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Uploaded CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    data = data[expected_features]

    processed_data = preprocessor.transform(data)

    if sparse.issparse(processed_data):
        values = processed_data.data
    else:
        values = np.asarray(processed_data)

    if np.isnan(values).any():
        raise ValueError("Processed data contains missing values.")

    if np.isinf(values).any():
        raise ValueError("Processed data contains infinite values.")

    probabilities = model.predict_proba(processed_data)

    default_class_index = list(model.classes_).index(1)

    default_probability = probabilities[:, default_class_index]

    predictions = (
        default_probability >= threshold
    ).astype(int)

    low_threshold = threshold * 0.50

    risk_categories = np.where(
        default_probability < low_threshold,
        "Low Risk",
        np.where(
            default_probability < threshold,
            "Medium Risk",
            "High Risk"
        )
    )

    results = pd.DataFrame({
        "SK_ID_CURR": applicant_ids,
        "Default Probability (%)": np.round(
            default_probability * 100, 2
        ),
        "Prediction": np.where(
            predictions == 1,
            "Default",
            "Non-Default"
        ),
        "Risk Category": risk_categories
    })

    return results


st.title("Credit Risk Intelligence")
st.caption(
    "Machine Learning Based Credit Default Risk Prediction System"
)

st.divider()

try:
    model, model_name, threshold, preprocessor = load_model()

    st.success(f"Model Loaded Successfully: {model_name}")

except Exception as error:
    st.error(f"Model Loading Error: {error}")
    st.stop()


st.subheader("Upload Applicant Dataset")

st.write(
    "Upload a CSV file containing applicant information "
    "to generate credit risk predictions."
)

uploaded_file = st.file_uploader(
    "Choose CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        uploaded_data = pd.read_csv(uploaded_file)

        if uploaded_data.empty:
            st.error("Uploaded CSV file is empty.")
            st.stop()

        st.subheader("Dataset Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Applicants",
                f"{len(uploaded_data):,}"
            )

        with col2:
            st.metric(
                "Total Columns",
                f"{uploaded_data.shape[1]:,}"
            )

        with col3:
            st.metric(
                "Missing Values",
                f"{uploaded_data.isnull().sum().sum():,}"
            )

        with st.expander("Preview Uploaded Dataset"):
            st.dataframe(
                uploaded_data.head(10),
                use_container_width=True
            )

        if st.button(
            "Generate Predictions",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Generating predictions..."):

                results = make_predictions(
                    uploaded_data,
                    model,
                    threshold,
                    preprocessor
                )

                st.session_state["prediction_results"] = results

        if "prediction_results" in st.session_state:

            results = st.session_state["prediction_results"]

            if len(results) != len(uploaded_data):
                st.session_state.pop("prediction_results")
                st.info("Please generate predictions again.")
                st.stop()

            st.divider()

            st.subheader("Prediction Dashboard")

            total_applicants = len(results)

            high_risk = (
                results["Risk Category"] == "High Risk"
            ).sum()

            medium_risk = (
                results["Risk Category"] == "Medium Risk"
            ).sum()

            low_risk = (
                results["Risk Category"] == "Low Risk"
            ).sum()

            predicted_default = (
                results["Prediction"] == "Default"
            ).sum()

            predicted_non_default = (
                results["Prediction"] == "Non-Default"
            ).sum()

            avg_probability = results[
                "Default Probability (%)"
            ].mean()

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total Applicants",
                f"{total_applicants:,}"
            )

            col2.metric(
                "High Risk",
                f"{high_risk:,}"
            )

            col3.metric(
                "Predicted Default",
                f"{predicted_default:,}"
            )

            col4.metric(
                "Average Default Probability",
                f"{avg_probability:.2f}%"
            )

            st.subheader("Risk Category Distribution")

            risk_summary = pd.DataFrame({
                "Risk Category": [
                    "Low Risk",
                    "Medium Risk",
                    "High Risk"
                ],
                "Applicants": [
                    low_risk,
                    medium_risk,
                    high_risk
                ]
            })

            st.bar_chart(
                risk_summary.set_index("Risk Category")
            )

            st.subheader("Prediction Distribution")

            prediction_summary = pd.DataFrame({
                "Prediction": [
                    "Non-Default",
                    "Default"
                ],
                "Applicants": [
                    predicted_non_default,
                    predicted_default
                ]
            })

            st.bar_chart(
                prediction_summary.set_index("Prediction")
            )

            st.subheader("Applicant Prediction Results")

            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True
            )

            st.subheader("High Risk Applicants")

            high_risk_results = results[
                results["Risk Category"] == "High Risk"
            ].sort_values(
                by="Default Probability (%)",
                ascending=False
            )

            st.dataframe(
                high_risk_results,
                use_container_width=True,
                hide_index=True
            )

            csv_data = results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Prediction Report",
                data=csv_data,
                file_name="credit_risk_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.caption(
                f"Classification threshold: {threshold:.4f} | "
                f"Model: {model_name}"
            )

    except Exception as error:
        st.error(f"Prediction Error: {error}")

else:
    st.info(
        "Upload a CSV file to get started."
    )

st.divider()

st.caption(
    "Credit Risk Intelligence | Machine Learning Project"
)
