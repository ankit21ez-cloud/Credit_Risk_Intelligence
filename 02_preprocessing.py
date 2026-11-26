import pandas as pd
import numpy as np
train = pd.read_csv(r"C:\Users\ANKIT KASHYAP\Desktop\Credit_Risk_Intelligence\dataset\application_train.csv")
print("Dataset Shape:")
print(train.shape)

missing_values = train.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0].sort_values(ascending=False))

missing_percentage = (missing_values/len(train)*100)
missing_data = pd.DataFrame({
    "Missing Count":missing_values,
    "Missing Percentage":missing_percentage
})
print("\nMissing Values Anlysis:")
print(missing_data)

# Remove columns having zero missing values
missing_data = missing_data[missing_data["Missing Count"] > 0]
print("Missing Values Anlysis:")
print(missing_data)

# Missing Values Classification
low_missing = missing_data[missing_data["Missing Percentage"] <= 10]
medium_missing = missing_data[
    (missing_data["Missing Percentage"] > 10)&
    (missing_data["Missing Percentage"] <= 40)
]
high_missing = missing_data[missing_data["Missing Percentage"] > 40]

print("Low Missing Columns:",len(low_missing))
print("Medium Missing Columns:",len(medium_missing))
print("High Missing Columns:",len(high_missing))

# Separate Numerical and Categorical Columns
numerical_columns = train.select_dtypes(include=["int64","float64"]).columns
categorical_columns = train.select_dtypes(include=["object"]).columns

print("\nSeparate Numerical and Categorical Columns")
print("Numerical Columns:",len(numerical_columns))
print("Categorical Columns:",len(categorical_columns))

# Numerical Columns Missing Values
numerical_missing = train[numerical_columns].isnull().sum()
numerical_missing = numerical_missing[numerical_missing > 0]
print("\nNumerical Missing Value:")
print(numerical_missing.sort_values(ascending=False))

# Categorical Columns Missing Values
categorical_missing = train[categorical_columns].isnull().sum()
categorical_missing = categorical_missing[categorical_missing > 0]
print("\nCategorical Missing Values:")
print(categorical_missing.sort_values(ascending=False))

# Numerical Missing Value Analysis
for column in numerical_missing.index:
    print("\nColumn:",column)
    print("Data type:",train[column].dtype)
    print("Median:",train[column].median())
    print("Mean:",train[column].mean())
    print("Skenwess:",train[column].skew())

# Categorical Missing Value Analysis
for column in categorical_missing.index:
    print("\nColumn:", column)
    print("Missing Values:", categorical_missing[column])
    print("Most Frequent Value:")
    print(train[column].mode()[0])
    print("Unique Values:")
    print(train[column].nunique())

# categorical missing percentage
categorical_missing_percentage = (categorical_missing/len(train))*100
print("\nCategorical Missing Percentage:")
print(categorical_missing_percentage.sort_values(ascending=False))

# numerical missing percentage
numerical_missing_percentage = (numerical_missing/len(train))*100
print("\nNumerical Missing Percentage:")
print(numerical_missing_percentage.sort_values(ascending=False))


numerical_missing = train[numerical_columns].isnull().sum()
numerical_missing = numerical_missing[numerical_missing > 0]

categorical_missing = train[categorical_columns].isnull().sum()
categorical_missing = categorical_missing[categorical_missing > 0]

# Numerical Missing Value Groups
low_missing_num = numerical_missing_percentage[
    numerical_missing_percentage <= 10
]
medium_missing_num = numerical_missing_percentage[
    (numerical_missing_percentage > 10) &
    (numerical_missing_percentage <= 40)
]
high_missing_num = numerical_missing_percentage[
    numerical_missing_percentage > 40
]
print("Low Missing Numerical:", len(low_missing_num))
print("Medium Missing Numerical:", len(medium_missing_num))
print("High Missing Numerical:", len(high_missing_num))


# Numerical Missing Value Strategy
for column in numerical_missing_percentage.index:
    missing_percent = numerical_missing_percentage[column]
    if missing_percent <= 10:
        print(column, "-> Median Imputition")

    elif missing_percent <= 40:
        print(column, "-> Median Imputition")

    else:
        print(column, "-> Feature Evalution Required")



# Categorical Missing Value Strategy

for column in categorical_missing_percentage.index:
    missing_percent = categorical_missing_percentage[column]
    if missing_percent <= 10:
        print(column, "-> Mode Imputation")

    else:
        print(column, "-> Unknown Category")


# Check Target Column
print("Target Column:",train["TARGET"].name)
print("\nTarget Distribution:")
print(train["TARGET"].value_counts())

# Separate Features and Target
x = train.drop(columns=["TARGET","SK_ID_CURR"])
y = train["TARGET"]

print("Feature shape:",x.shape)
print("Target shape:",y.shape)


from sklearn.model_selection import train_test_split
x_train,x_valid,y_train,y_valid = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

print("Traning Feature:",x_train.shape)
print("Validation Feature:",x_valid.shape)

print("Traning Target Distribution:")
print(y_train.value_counts(normalize=True)*100)
print("\nValidation Target Distribution:")
print(y_valid.value_counts(normalize=True)*100)

# Separate Numerical and Categorical Features
numerical_features = x_train.select_dtypes(include=["int64","float64"]).columns
categorical_features = x_train.select_dtypes(include=["object"]).columns

print("Numerical Features:",numerical_features)
print("Categorical Features:",categorical_features)


# Missing Values in Training Features

numerical_missing = x_train[numerical_features].isnull().sum()
numerical_missing = numerical_missing[numerical_missing > 0]

categorical_missing = x_train[categorical_features].isnull().sum()
categorical_missing = categorical_missing[categorical_missing > 0]

print("Numerical Missing Columns:", len(numerical_missing))
print(numerical_missing.sort_values(ascending=False))

print("\nCategorical Missing Columns:", len(categorical_missing))
print(categorical_missing.sort_values(ascending=False))

# Numerical Missing Value Imputation

from sklearn.impute import SimpleImputer

numerical_imputer = SimpleImputer(strategy="median")

x_train[numerical_features] = numerical_imputer.fit_transform(
    x_train[numerical_features]
)
x_valid[numerical_features] = numerical_imputer.transform(
    x_valid[numerical_features]
)
print("Remaining Numerical Missing Values in Training:",
      x_train[numerical_features].isnull().sum().sum())

print("Remaining Numerical Missing Values in Validation:",
      x_valid[numerical_features].isnull().sum().sum())

# CATEGORICAL MISSING VALUE IMPUTATION
# Separate Mode and Unknown Features

mode_features = ["NAME_TYPE_SUITE"]

unknown_features = [
    column for column in categorical_features
    if column != "NAME_TYPE_SUITE"
]


# 1. Mode Imputation

mode_imputer = SimpleImputer(strategy="most_frequent")
x_train[mode_features] = mode_imputer.fit_transform(
    x_train[mode_features]
)
x_valid[mode_features] = mode_imputer.transform(
    x_valid[mode_features]
)


# 2. Unknown Category Imputation

unknown_imputer = SimpleImputer(
    strategy="constant",
    fill_value="Unknown"
)
x_train[unknown_features] = unknown_imputer.fit_transform(
    x_train[unknown_features]
)
x_valid[unknown_features] = unknown_imputer.transform(
    x_valid[unknown_features]
)


# 3. Check Missing Values

print("\nCategorical Imputation Results")
print("Training Missing Values:",
      x_train[categorical_features].isnull().sum().sum())
print("Validation Missing Values:",
      x_valid[categorical_features].isnull().sum().sum())


# ENCODING AND FEATURE SCALING

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from scipy import sparse

# Create Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("numerical_scaling", StandardScaler(), numerical_features),
        ("categorical_encoding", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

x_train_processed = preprocessor.fit_transform(x_train)
x_valid_processed = preprocessor.transform(x_valid)

print("\nEncoding and Scaling Results")
print("Training Shape:", x_train_processed.shape)
print("Validation Shape:", x_valid_processed.shape)
print("Training Target Shape:", y_train.shape)
print("Validation Target Shape:", y_valid.shape)

# Check Missing Values
if sparse.issparse(x_train_processed):
    train_missing = np.isnan(x_train_processed.data).sum()
else:
    train_missing = np.isnan(x_train_processed).sum()

if sparse.issparse(x_valid_processed):
    valid_missing = np.isnan(x_valid_processed.data).sum()
else:
    valid_missing = np.isnan(x_valid_processed).sum()

print("Training Missing Values:", train_missing)
print("Validation Missing Values:", valid_missing)


# PREPROCESSING FINAL CHECKS

print("\nPreprocessing Final Report")
feature_names = preprocessor.get_feature_names_out()
print("Total Processed Features:", len(feature_names))

if sparse.issparse(x_train_processed):
    train_infinite = np.isinf(x_train_processed.data).sum()
    valid_infinite = np.isinf(x_valid_processed.data).sum()
else:
    train_infinite = np.isinf(x_train_processed).sum()
    valid_infinite = np.isinf(x_valid_processed).sum()

print("Training Infinite Values:", train_infinite)
print("Validation Infinite Values:", valid_infinite)
print("\nTraining Target Distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nValidation Target Distribution:")
print(y_valid.value_counts(normalize=True) * 100)
print("\nFirst 20 Processed Features:")
print(feature_names[:20])
