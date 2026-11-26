
from pathlib import Path
from io import BytesIO

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best_model.pkl"

best_model_data = joblib.load(MODEL_PATH)

model = best_model_data["model"]
preprocessor = best_model_data["preprocessor"]
threshold = best_model_data["threshold"]

required_features = list(preprocessor.feature_names_in_)

if 1 not in model.classes_:
    raise ValueError("Default class (1) not found in trained model!")

if not 0 < threshold < 1:
    raise ValueError("Invalid prediction threshold!")


app = FastAPI(
    title="Credit Risk Intelligence API",
    description="Credit Risk Prediction using XGBoost Machine Learning Model",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def safe_divide(numerator, denominator):
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def create_features(data):

    data = data.copy()

    data["DAYS_EMPLOYED"] = data["DAYS_EMPLOYED"].replace(
        365243, np.nan
    )

    data["AGE_YEARS"] = abs(data["DAYS_BIRTH"]) / 365

    data["EMPLOYED_YEARS"] = (
        abs(data["DAYS_EMPLOYED"]) / 365
    )

    data["REGISTRATION_YEARS"] = (
        abs(data["DAYS_REGISTRATION"]) / 365
    )

    data["ID_PUBLISH_YEARS"] = (
        abs(data["DAYS_ID_PUBLISH"]) / 365
    )

    data["CREDIT_INCOME_RATIO"] = safe_divide(
        data["AMT_CREDIT"],
        data["AMT_INCOME_TOTAL"]
    )

    data["ANNUITY_INCOME_RATIO"] = safe_divide(
        data["AMT_ANNUITY"],
        data["AMT_INCOME_TOTAL"]
    )

    data["CREDIT_TERM"] = safe_divide(
        data["AMT_CREDIT"],
        data["AMT_ANNUITY"]
    )

    data["PAYMENT_RATE"] = safe_divide(
        data["AMT_ANNUITY"],
        data["AMT_CREDIT"]
    )

    data["GOODS_CREDIT_RATIO"] = safe_divide(
        data["AMT_GOODS_PRICE"],
        data["AMT_CREDIT"]
    )

    data["INCOME_PER_PERSON"] = safe_divide(
        data["AMT_INCOME_TOTAL"],
        data["CNT_FAM_MEMBERS"]
    )

    data["INCOME_PER_CHILD"] = safe_divide(
        data["AMT_INCOME_TOTAL"],
        data["CNT_CHILDREN"] + 1
    )

    data["EMPLOYMENT_AGE_RATIO"] = safe_divide(
        data["EMPLOYED_YEARS"],
        data["AGE_YEARS"]
    )

    ext_source_columns = [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3"
    ]

    data["EXT_SOURCE_MEAN"] = data[
        ext_source_columns
    ].mean(axis=1)

    data["EXT_SOURCE_MIN"] = data[
        ext_source_columns
    ].min(axis=1)

    data["EXT_SOURCE_MAX"] = data[
        ext_source_columns
    ].max(axis=1)

    data["EXT_SOURCE_STD"] = data[
        ext_source_columns
    ].std(axis=1)

    document_columns = [
        col for col in data.columns
        if col.startswith("FLAG_DOCUMENT_")
    ]

    data["DOCUMENT_COUNT"] = data[
        document_columns
    ].sum(axis=1)

    contact_columns = [
        "FLAG_MOBIL",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_CONT_MOBILE",
        "FLAG_PHONE",
        "FLAG_EMAIL"
    ]

    data["CONTACT_INFORMATION_COUNT"] = data[
        contact_columns
    ].sum(axis=1)

    return data


@app.get("/")
def home():

    return {
        "message": "Credit Risk Intelligence API is running",
        "status": "success",
        "model": best_model_data["model_name"]
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid CSV file."
        )

    try:

        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV file is empty."
            )

        data = pd.read_csv(BytesIO(contents))

        if data.empty:
            raise HTTPException(
                status_code=400,
                detail="CSV file contains no applicant records."
            )

        if "TARGET" in data.columns:
            data = data.drop(columns=["TARGET"])

        if "SK_ID_CURR" in data.columns:
            applicant_ids = data["SK_ID_CURR"].tolist()
            X = data.drop(columns=["SK_ID_CURR"])
        else:
            applicant_ids = list(range(1, len(data) + 1))
            X = data.copy()

        X = create_features(X)

        missing_features = [
            col for col in required_features
            if col not in X.columns
        ]

        if missing_features:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "CSV is missing required features.",
                    "missing_features": missing_features
                }
            )

        X = X[required_features]

        X_processed = preprocessor.transform(X)

        class_index = list(model.classes_).index(1)

        probabilities = model.predict_proba(
            X_processed
        )[:, class_index]

        if not np.isfinite(probabilities).all():
            raise ValueError(
                "Model generated invalid probability values!"
            )

        predictions = (
            probabilities >= threshold
        ).astype(int)

        low_risk_threshold = threshold * 0.50

        risk_categories = []

        for probability in probabilities:

            if probability < low_risk_threshold:
                risk_categories.append("Low Risk")

            elif probability < threshold:
                risk_categories.append("Medium Risk")

            else:
                risk_categories.append("High Risk")

        results = pd.DataFrame({

            "Applicant_ID": applicant_ids,

            "Default_Probability": probabilities,

            "Prediction": predictions,

            "Risk_Category": risk_categories

        })

        results["Default_Probability"] = (
            results["Default_Probability"].round(4)
        )

        risk_counts = results[
            "Risk_Category"
        ].value_counts()

        return {

            "status": "success",

            "total_applicants": len(results),

            "average_default_probability": round(
                float(np.mean(probabilities)), 4
            ),

            "risk_distribution": {

                "Low Risk": int(
                    risk_counts.get("Low Risk", 0)
                ),

                "Medium Risk": int(
                    risk_counts.get("Medium Risk", 0)
                ),

                "High Risk": int(
                    risk_counts.get("High Risk", 0)
                )

            },

            "predictions": results.to_dict(
                orient="records"
            )

        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
