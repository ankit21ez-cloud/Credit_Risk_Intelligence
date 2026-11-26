import pandas as pd
train = pd.read_csv(r"C:\Users\ANKIT KASHYAP\Desktop\Credit_Risk_Intelligence\dataset\application_train.csv")
test = pd.read_csv(r"C:\Users\ANKIT KASHYAP\Desktop\Credit_Risk_Intelligence\dataset\application_test.csv")

print("Traning shape",train.shape)
print("Testing shape",test.shape)

print("\nTraning Data")
print(train.head())
print("\nTesting Data")
print(test.head())

print("\nDataset Information:")
print(train.info())
print("\nDataset Columns:")
print(test.columns)

print("\nMissing Value")
print(train.isnull().sum().sort_values(ascending=False))

print("\nTarget Distribution")
print(train["TARGET"].value_counts())

print("\nTarget percent")
print(train["TARGET"].value_counts(normalize=True)*100)

import matplotlib.pyplot as plt
import seaborn as sns
target_count = train["TARGET"].value_counts()
plt.figure(figsize=(10,6))
plt.bar(target_count.index,target_count.values)
plt.title("Credit Risk - Target Distribution")
plt.xlabel("Target")
plt.ylabel("Number of Customer")
plt.xticks([0,1],["No Payment Difficulties","Payment Difficulties"])
plt.show()

print("\nData Type Count:")
print(train.dtypes.value_counts())

categorical_column= train.select_dtypes(include=["object"]).columns
print("\nCategorical Column:")
print(categorical_column)


numerical_column= train.select_dtypes(include=["int64","float64"]).columns
print("Numerical Column:")
print(numerical_column)

print("\nCategorical columns:")
print(train.select_dtypes(include=["object"]).columns.tolist())

print("\nUnique Values")
for col in train.select_dtypes(include=["object"]).columns:
    print(col, ":", train[col].nunique())


print("\nCODE_GENDER value:")
print(train["CODE_GENDER"].value_counts(dropna=False))
print("\nNAME_CONTRACT_TYPE value:")
print(train["NAME_CONTRACT_TYPE"].value_counts(dropna=True))
print("\nORGANIZATION_TYPE top 10:")
print(train["ORGANIZATION_TYPE"].value_counts(dropna=False).head(10))

print("\nCategoriacl missing value:")
print(train.select_dtypes(include="object").isnull().sum())

cat_missing = train.select_dtypes(include="object").isnull().sum()
cat_missing_percent = (cat_missing/len(train))*100
print("\nCategorical Missing Prescentage")
print(cat_missing_percent.sort_values(ascending=False))

num_missing = train.select_dtypes(include=["int64","float64"]).isnull().sum()
print("\nNumerical missing values:")
print(num_missing.sort_values(ascending=False).head(20))

num_missing = train.select_dtypes(include=["int64","float64"]).isnull().sum()
num_missing_percent = (num_missing / len(train))*100

print("\nNumerical Missing Percantage:")
print(num_missing_percent.sort_values(ascending=False).head(20))

print("\nNumerical Statistical values:")
print(train.describe())

financial_column = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE"
]
print("\nFinancial Feature Anlysis:")
print(train[financial_column].describe().T)

Q1 = train["AMT_INCOME_TOTAL"].quantile(0.25)
Q3 = train["AMT_INCOME_TOTAL"].quantile(0.75)
IQR = Q3-Q1

lower_limt = Q1 - 1.5*IQR
upper_limt = Q3 + 1.5*IQR

outliers = train[
    (train["AMT_INCOME_TOTAL"] < lower_limt)|
    (train["AMT_INCOME_TOTAL"] > upper_limt)

]
print("AMT_INCOME_TOTAL Outliers Analysis:")
print("Q1",Q1)
print("Q3",Q3)
print("IQR",IQR)
print("Lower_Limt:",lower_limt)
print("Upper_Limit:",upper_limt)
print("Total Outliers:",len(outliers))

plt.figure(figsize=(10,6))
plt.boxplot(train["AMT_INCOME_TOTAL"].dropna())
plt.title("Income Outliers Dection")
plt.ylabel("Income")
plt.show()

Q1 = train["AMT_CREDIT"].quantile(0.25)
Q3 = train["AMT_CREDIT"].quantile(0.75)
IQR = Q3 - Q1

min_limit = Q1 - 1.5*IQR
max_limit = Q3 + 1.5*IQR

outliers = train[
    (train["AMT_CREDIT"] < min_limit)|
    (train["AMT_CREDIT"] > max_limit)
]
print("\nAMT_CREDIT Outliers Anlysis:")
print("Q1:",Q1)
print("Q3:",Q3)
print("IQR:",IQR)
print("Min Limit:",min_limit)
print("Max Limit:",max_limit)
print("Total outliers:",len(outliers))


Q1 = train["AMT_ANNUITY"].quantile(0.25)
Q3 = train["AMT_ANNUITY"].quantile(0.75)
IQR = Q3 - Q1

low_limit =  Q1 - 1.5*IQR
high_limit = Q3 + 1.5*IQR

outliers = train[
    (train["AMT_ANNUITY"] < low_limit)|
    (train["AMT_ANNUITY"] > high_limit)
]
print("AMT_ANNUITY Outliers Anlysis:")
print("Q1",Q1)
print("Q3",Q3)
print("IQR",IQR)
print("Low Limit:",low_limit)
print("High Limit:",high_limit)
print("Total Outliers:",len(outliers))

Q1 = train["AMT_GOODS_PRICE"].quantile(0.25)
Q3 = train["AMT_GOODS_PRICE"].quantile(0.75)
IQR = Q3 - Q1
buttom_limit = Q1 - 1.5*IQR
top_limit = Q3 + 1.5*IQR
outliers = train[
    (train["AMT_GOODS_PRICE"] < buttom_limit)|
    (train["AMT_GOODS_PRICE"] > top_limit)
]
print("AMT_GOODS_PRICE Outliers Anlysisi:")
print("Q1:",Q1)
print("Q3:",Q3)
print("IQR:",IQR)
print("Buttom Limit:",buttom_limit)
print("Top Limit:",top_limit)
print("Total Oultires:",len(outliers))


financial_column = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "TARGET"
]
correlation = train[financial_column].corr()
print("\nFinancial Correlation:")
print(correlation)


plt.figure(figsize=(10,6))
sns.heatmap(correlation,annot=True,cmap="coolwarm",fmt=".2f")
plt.title("Financial Correlation Heatmap")
plt.show()

financial_column = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE"
]
skewness = train[financial_column].skew()
print("\nFeature Skewness Anlysis:")
print(skewness)

plt.figure(figsize=(10,6))
for i,column in enumerate(financial_column):
    plt.subplot(2,2,i+1)
    sns.histplot(train[column], bins=50, kde=True)
    plt.title(column)
plt.tight_layout()
plt.show()


print("Age Statistical:")
print(train["DAYS_BIRTH"].describe())

train["AGE"] = (-train["DAYS_BIRTH"]/365).round(1)
print("Age Statistics:")
print(train["AGE"].describe())

plt.figure(figsize=(10, 6))
sns.histplot(train["AGE"], bins=50, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age in Years")
plt.ylabel("Number of Applicants")
plt.show()

print(train.groupby("TARGET")[financial_column].mean())