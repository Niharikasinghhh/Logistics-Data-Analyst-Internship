"""
Week 3: Advanced Data Analysis and Visualization in Logistics
This script creates a hypothetical logistics dataset and performs EDA.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 1. DATA SIMULATION
n = 500

df = pd.DataFrame({
    "shipment_id": [f"SHP{i:04d}" for i in range(1, n + 1)],
    "region": np.random.choice(["North", "South", "East", "West"], n),
    "transport_mode": np.random.choice(["Road", "Rail", "Air"], n, p=[0.65, 0.22, 0.13]),
    "shipment_volume": np.random.gamma(3.5, 28, n).clip(10, 300),
    "distance_km": np.random.normal(650, 280, n).clip(50, 1800)
})

mode_factor = df["transport_mode"].map({"Road": 1.0, "Rail": 1.15, "Air": 0.55})
region_factor = df["region"].map({"North": 1.00, "South": 1.10, "East": 0.95, "West": 1.05})

df["delivery_time_days"] = (
    1.5
    + df["distance_km"] / 180
    + np.random.normal(0, 1.3, n)
) * mode_factor * region_factor

df["delivery_time_days"] = df["delivery_time_days"].clip(1, 25)

df["transport_cost"] = (
    220
    + df["distance_km"] * 1.9
    + df["shipment_volume"] * 7.5
    + np.random.normal(0, 220, n)
).clip(200)

delay_probability = np.clip(
    0.10
    + (df["delivery_time_days"] > df["delivery_time_days"].median()) * 0.10
    + (df["distance_km"] > 900) * 0.08
    + (df["transport_mode"] == "Road") * 0.05,
    0.05,
    0.60
)

df["delivery_status"] = np.where(
    np.random.binomial(1, delay_probability) == 1,
    "Delayed",
    "On Time"
)

df.to_csv("week3_hypothetical_logistics_data.csv", index=False)

# 2. EXPLORATORY DATA ANALYSIS
print("\nDESCRIPTIVE STATISTICS")
print(df.describe())

print("\nCORRELATION MATRIX")
print(df.corr(numeric_only=True))

print("\nAVERAGE DELIVERY TIME BY REGION")
print(df.groupby("region")["delivery_time_days"].mean())

print("\nAVERAGE TRANSPORT COST BY MODE")
print(df.groupby("transport_mode")["transport_cost"].mean())

print("\nDELIVERY STATUS COUNTS")
print(df["delivery_status"].value_counts())

# 3. VISUALIZATION: DELIVERY TIME DISTRIBUTION
plt.figure(figsize=(8, 5))
plt.hist(df["delivery_time_days"], bins=25, edgecolor="black")
plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (Days)")
plt.ylabel("Number of Shipments")
plt.tight_layout()
plt.show()

# 4. VISUALIZATION: COST BY TRANSPORT MODE
mode_means = df.groupby("transport_mode")["transport_cost"].mean()
plt.figure(figsize=(8, 5))
plt.bar(mode_means.index, mode_means.values)
plt.title("Average Transportation Cost by Transport Mode")
plt.xlabel("Transport Mode")
plt.ylabel("Average Transportation Cost")
plt.tight_layout()
plt.show()

# 5. VISUALIZATION: DISTANCE VS COST
plt.figure(figsize=(8, 5))
plt.scatter(df["distance_km"], df["transport_cost"], alpha=0.55)
plt.title("Relationship Between Distance and Transportation Cost")
plt.xlabel("Distance (KM)")
plt.ylabel("Transportation Cost")
plt.tight_layout()
plt.show()

# 6. VISUALIZATION: DELIVERY TIME BY REGION
region_delivery = df.groupby("region")["delivery_time_days"].mean()
plt.figure(figsize=(8, 5))
plt.bar(region_delivery.index, region_delivery.values)
plt.title("Average Delivery Time by Region")
plt.xlabel("Region")
plt.ylabel("Average Delivery Time (Days)")
plt.tight_layout()
plt.show()

# 7. VISUALIZATION: DELIVERY STATUS
status_counts = df["delivery_status"].value_counts()
plt.figure(figsize=(8, 5))
plt.bar(status_counts.index, status_counts.values)
plt.title("Delivery Performance: On-Time vs Delayed Shipments")
plt.xlabel("Delivery Status")
plt.ylabel("Number of Shipments")
plt.tight_layout()
plt.show()
