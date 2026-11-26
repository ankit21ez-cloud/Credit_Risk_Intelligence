
import os
import pandas as pd
import numpy as np
import joblib

from scipy import sparse

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

train = pd.read_csv("dataset/application_train.csv")
test = pd.read_csv("dataset/application_test.csv")

print("Training Dataset Shape:", train.shape)
print("Testing Dataset Shape:", test.shape)

X = train.drop(columns=["TARGET", "SK_ID_CURR"])
y = train["TARGET"]

test_ids = test["SK_ID_CURR"]
X_test = test.drop(columns=["SK_ID_CURR"])

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)


X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Validation Shape:", X_valid.shape)


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

        result = result.replace(
            [np.inf, -np.inf],
            np.nan
        )

        return result

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


X_train_fe = create_features(X_train)
X_valid_fe = create_features(X_valid)
X_test_fe = create_features(X_test)

print("\nFeature Engineering Completed")

print("Training Shape:", X_train_fe.shape)
print("Validation Shape:", X_valid_fe.shape)
print("Testing Shape:", X_test_fe.shape)


new_features = [
    "AGE_YEARS",
    "EMPLOYED_YEARS",
    "REGISTRATION_YEARS",
    "ID_PUBLISH_YEARS",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_TERM",
    "PAYMENT_RATE",
    "GOODS_CREDIT_RATIO",
    "INCOME_PER_PERSON",
    "INCOME_PER_CHILD",
    "EMPLOYMENT_AGE_RATIO",
    "EXT_SOURCE_MEAN",
    "EXT_SOURCE_MIN",
    "EXT_SOURCE_MAX",
    "EXT_SOURCE_STD",
    "DOCUMENT_COUNT",
    "CONTACT_INFORMATION_COUNT"
]

available_new_features = [
    col for col in new_features
    if col in X_train_fe.columns
]

print("\nNewly Created Features:")
print(available_new_features)

print("\nNew Feature Count:", len(available_new_features))

print("\nNew Feature Statistical Analysis:")

print(
    X_train_fe[available_new_features].describe().T
)


X_valid_fe = X_valid_fe.reindex(
    columns=X_train_fe.columns
)

X_test_fe = X_test_fe.reindex(
    columns=X_train_fe.columns
)

print("\nFeature Alignment Check:")
print("Training Columns:", len(X_train_fe.columns))
print("Validation Columns:", len(X_valid_fe.columns))
print("Testing Columns:", len(X_test_fe.columns))


numerical_features = X_train_fe.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train_fe.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical Features:", len(numerical_features))
print("Categorical Features:", len(categorical_features))

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


mode_features = [
    col for col in categorical_features
    if col == "NAME_TYPE_SUITE"
]

unknown_features = [
    col for col in categorical_features
    if col != "NAME_TYPE_SUITE"
]


mode_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

unknown_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Unknown"
            )
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


transformers = [
    (
        "numerical",
        numerical_pipeline,
        numerical_features
    )
]

if len(mode_features) > 0:

    transformers.append(
        (
            "categorical_mode",
            mode_pipeline,
            mode_features
        )
    )

if len(unknown_features) > 0:

    transformers.append(
        (
            "categorical_unknown",
            unknown_pipeline,
            unknown_features
        )
    )

preprocessor = ColumnTransformer(
    transformers=transformers
)


X_train_processed = preprocessor.fit_transform(
    X_train_fe
)

X_valid_processed = preprocessor.transform(
    X_valid_fe
)

X_test_processed = preprocessor.transform(
    X_test_fe
)

print("\nFinal Processed Data Shapes:")

print("Training:", X_train_processed.shape)
print("Validation:", X_valid_processed.shape)
print("Testing:", X_test_processed.shape)


def check_data_quality(data, name):

    # Sparse matrix ko handle karna
    if sparse.issparse(data):
        values = data.data
    else:
        values = np.asarray(data)

    print("\n", name)

    print(
        "Missing Values:",
        np.isnan(values).sum()
    )

    print(
        "Infinite Values:",
        np.isinf(values).sum()
    )


check_data_quality(
    X_train_processed,
    "Training Data"
)

check_data_quality(
    X_valid_processed,
    "Validation Data"
)

check_data_quality(
    X_test_processed,
    "Testing Data"
)

print(
    "\nTraining Target Missing:",
    y_train.isnull().sum()
)

print(
    "Validation Target Missing:",
    y_valid.isnull().sum()
)

feature_names = preprocessor.get_feature_names_out()

print("\nTotal Final Features:", len(feature_names))

print("\nFirst 30 Features:")
print(feature_names[:30])

os.makedirs("model", exist_ok=True)

output = {
    "X_train": X_train_processed,
    "X_valid": X_valid_processed,
    "X_test": X_test_processed,
    "y_train": y_train,
    "y_valid": y_valid,
    "test_ids": test_ids,
    "preprocessor": preprocessor,
    "feature_names": feature_names
}

joblib.dump(
    output,
    "model/feature_engineering_output.pkl"
)

print("\nFeature Engineering Output Saved Successfully!")

print(
    "File: model/feature_engineering_output.pkl"
)

print("\n03 FEATURE ENGINEERING COMPLETED")
