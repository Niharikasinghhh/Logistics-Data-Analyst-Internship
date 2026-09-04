"""
Week 4: Predictive Modeling and Optimization in Logistics Systems
Predict delivery time using Linear Regression and Random Forest.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# 1. DATA SIMULATION
# --------------------------------------------------
np.random.seed(42)
n = 600

regions = np.random.choice(["North", "South", "East", "West"], n)
transport_modes = np.random.choice(
    ["Road", "Rail", "Air"], n, p=[0.64, 0.23, 0.13]
)

shipment_volume = np.random.gamma(3.2, 30, n).clip(8, 350)
distance_km = np.random.normal(650, 280, n).clip(50, 1800)

mode_factor = pd.Series(transport_modes).map(
    {"Road": 1.00, "Rail": 1.18, "Air": 0.58}
).to_numpy()

region_factor = pd.Series(regions).map(
    {"North": 1.00, "South": 1.12, "East": 0.96, "West": 1.06}
).to_numpy()

delivery_time_days = (
    1.4
    + distance_km / 190
    + shipment_volume / 220
    + np.random.normal(0, 0.9, n)
) * mode_factor * region_factor

delivery_time_days = np.clip(delivery_time_days, 1, 25)

transport_cost = (
    250
    + distance_km * 1.85
    + shipment_volume * 7.2
    + pd.Series(transport_modes).map(
        {"Road": 100, "Rail": 180, "Air": 500}
    ).to_numpy()
    + np.random.normal(0, 180, n)
)

transport_cost = np.clip(transport_cost, 200, None)

df = pd.DataFrame({
    "region": regions,
    "transport_mode": transport_modes,
    "shipment_volume": shipment_volume,
    "distance_km": distance_km,
    "transport_cost": transport_cost,
    "delivery_time_days": delivery_time_days
})

df.to_csv("week4_logistics_prediction_data.csv", index=False)

# --------------------------------------------------
# 2. FEATURES AND TARGET
# --------------------------------------------------
features = [
    "region",
    "transport_mode",
    "shipment_volume",
    "distance_km",
    "transport_cost"
]

target = "delivery_time_days"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

categorical_features = ["region", "transport_mode"]
numerical_features = [
    "shipment_volume",
    "distance_km",
    "transport_cost"
]

preprocessor = ColumnTransformer([
    (
        "numerical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]),
        numerical_features
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_features
    )
])

# --------------------------------------------------
# 3. MODELS
# --------------------------------------------------
linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

random_forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        random_state=42
    ))
])

models = {
    "Linear Regression": linear_model,
    "Random Forest": random_forest_model
}

results = {}

# --------------------------------------------------
# 4. TRAINING AND EVALUATION
# --------------------------------------------------
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="r2"
    )

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "CV_R2": cv_scores.mean()
    }

print("\nMODEL PERFORMANCE")
for model_name, metrics in results.items():
    print(f"\n{model_name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# Select model with lowest RMSE
best_model_name = min(
    results,
    key=lambda model: results[model]["RMSE"]
)

best_model = models[best_model_name]
best_model.fit(X_train, y_train)

best_predictions = best_model.predict(X_test)

print(f"\nSelected Model: {best_model_name}")

# --------------------------------------------------
# 5. ACTUAL VS PREDICTED VISUALIZATION
# --------------------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(y_test, best_predictions, alpha=0.65)

minimum = min(y_test.min(), best_predictions.min())
maximum = max(y_test.max(), best_predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.title(f"Actual vs Predicted Delivery Time - {best_model_name}")
plt.xlabel("Actual Delivery Time (Days)")
plt.ylabel("Predicted Delivery Time (Days)")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 6. OPTIMIZATION: HIGH-RISK SHIPMENTS
# --------------------------------------------------
df["predicted_delivery_time"] = best_model.predict(X)

threshold = df["predicted_delivery_time"].quantile(0.75)

df["delivery_risk"] = np.where(
    df["predicted_delivery_time"] > threshold,
    "High Risk",
    "Normal"
)

print("\nHIGH-RISK SHIPMENT SUMMARY")
print(df["delivery_risk"].value_counts())

# Compare predicted delivery time by transport mode
mode_delivery = df.groupby(
    "transport_mode"
)["predicted_delivery_time"].mean()

plt.figure(figsize=(8, 5))
plt.bar(mode_delivery.index, mode_delivery.values)
plt.title("Average Predicted Delivery Time by Transport Mode")
plt.xlabel("Transport Mode")
plt.ylabel("Predicted Delivery Time (Days)")
plt.tight_layout()
plt.show()

# Save optimization results
df.to_csv(
    "week4_logistics_optimization_results.csv",
    index=False
)

print("\nWeek 4 analysis completed successfully.")
