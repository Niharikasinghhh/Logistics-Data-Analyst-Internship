# Week 1 - Strategic Planning and Data Exploration in Logistics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. LOAD DATA
# -------------------------------

# Replace this file name with your actual dataset later
df = pd.read_csv("logistics_data.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())


# -------------------------------
# 2. DATA CLEANING
# -------------------------------

# Remove duplicate records
df = df.drop_duplicates()

# Convert date columns to datetime format
df["order_date"] = pd.to_datetime(df["order_date"])
df["delivery_date"] = pd.to_datetime(df["delivery_date"])
df["expected_delivery_date"] = pd.to_datetime(
    df["expected_delivery_date"]
)

# Create delivery time feature
df["delivery_time_days"] = (
    df["delivery_date"] - df["order_date"]
).dt.days


# -------------------------------
# 3. CREATE DELAY STATUS
# -------------------------------

# 1 = Delayed
# 0 = Delivered on time

df["delayed"] = (
    df["delivery_date"] > df["expected_delivery_date"]
).astype(int)


# -------------------------------
# 4. KPI CALCULATIONS
# -------------------------------

# On-time delivery rate
on_time_delivery_rate = (
    1 - df["delayed"].mean()
) * 100

# Delivery delay rate
delivery_delay_rate = (
    df["delayed"].mean()
) * 100

# Average delivery time
average_delivery_time = (
    df["delivery_time_days"].mean()
)

print("\n--- LOGISTICS KPIs ---")

print(
    f"On-Time Delivery Rate: "
    f"{on_time_delivery_rate:.2f}%"
)

print(
    f"Delivery Delay Rate: "
    f"{delivery_delay_rate:.2f}%"
)

print(
    f"Average Delivery Time: "
    f"{average_delivery_time:.2f} days"
)


# -------------------------------
# 5. EXPLORATORY DATA ANALYSIS
# -------------------------------

# Delivery time distribution

plt.figure(figsize=(8, 5))

plt.hist(
    df["delivery_time_days"].dropna(),
    bins=20
)

plt.title("Distribution of Delivery Time")

plt.xlabel("Delivery Time (Days)")

plt.ylabel("Number of Deliveries")

plt.show()


# -------------------------------
# 6. BASIC INSIGHTS
# -------------------------------

print("\nAverage Delivery Time Summary:")

print(
    df["delivery_time_days"].describe()
)