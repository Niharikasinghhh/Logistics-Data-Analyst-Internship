"""
Week 2 - Data Collection, Cleaning and Preprocessing for Logistics Analysis
Reference dataset: Olist Brazilian E-Commerce Public Dataset
Input:  olist_orders_dataset.csv
Output: cleaned_logistics_data.csv
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. LOAD DATA
df = pd.read_csv("olist_orders_dataset.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# 2. REMOVE DUPLICATES
print("\nDuplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

# 3. CONVERT DATE COLUMNS
date_columns = [
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for column in date_columns:
    df[column] = pd.to_datetime(df[column], errors="coerce")

# 4. HANDLE MISSING VALUES
clean_df = df.dropna(
    subset=[
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
).copy()

# 5. FEATURE ENGINEERING
clean_df["delivery_time_days"] = (
    clean_df["order_delivered_customer_date"]
    - clean_df["order_purchase_timestamp"]
).dt.total_seconds() / 86400

clean_df["delayed"] = (
    clean_df["order_delivered_customer_date"]
    > clean_df["order_estimated_delivery_date"]
).astype(int)

# 6. OUTLIER DETECTION USING IQR
Q1 = clean_df["delivery_time_days"].quantile(0.25)
Q3 = clean_df["delivery_time_days"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

clean_df["potential_outlier"] = (
    (clean_df["delivery_time_days"] < lower_bound)
    | (clean_df["delivery_time_days"] > upper_bound)
)

print("\nPotential outliers:", clean_df["potential_outlier"].sum())

# 7. NORMALIZATION
scaler = MinMaxScaler()

clean_df["delivery_time_normalized"] = scaler.fit_transform(
    clean_df[["delivery_time_days"]]
)

# 8. FINAL VALIDATION
print("\nFinal dataset shape:", clean_df.shape)
print("\nFinal missing values:")
print(clean_df.isnull().sum())

# 9. SAVE CLEAN DATA
clean_df.to_csv("cleaned_logistics_data.csv", index=False)

print("\nPreprocessing completed successfully.")