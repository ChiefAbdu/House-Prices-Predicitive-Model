import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Set visual style
sns.set(style="whitegrid", context="notebook")

# Create output directory
output_dir = "california_housing_figures"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

# Rename columns for clarity
df.rename(columns={
    "MedInc": "MedianIncome",
    "HouseAge": "HouseAge",
    "AveRooms": "AverageRooms",
    "AveBedrms": "AverageBedrooms",
    "Population": "Population",
    "AveOccup": "AverageOccupancy",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
    "MedHouseVal": "MedianHouseValue"
}, inplace=True)

# Convert house values into USD
df["MedianHouseValueUSD"] = df["MedianHouseValue"] * 100000

# -------------------------------
# 5 HIGHLY INFORMATIVE FIGURES
# -------------------------------

# 1. Geographical Scatter Plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    df["Longitude"], df["Latitude"],
    c=df["MedianHouseValue"],
    cmap="viridis", alpha=0.6
)
plt.colorbar(scatter, label="Median House Value (Hundreds of Thousands USD)")
plt.title("Geographical Distribution of House Prices")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig(f"{output_dir}/01_geographical_price_scatter.png", dpi=300)
plt.close()

# 2. Median Income vs. Median House Value
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="MedianIncome", y="MedianHouseValue",
    data=df, alpha=0.5
)
plt.title("Median Income vs. Median House Value")
plt.xlabel("Median Income (Tens of Thousands USD)")
plt.ylabel("Median House Value (Hundreds of Thousands USD)")
plt.savefig(f"{output_dir}/02_income_vs_house_value.png", dpi=300)
plt.close()

# 3. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation Heatmap Matrix")
plt.savefig(f"{output_dir}/03_correlation_heatmap.png", dpi=300)
plt.close()

# 4. Average Rooms vs. Median House Value with Trendline
plt.figure(figsize=(8, 6))
sns.regplot(
    x="AverageRooms",
    y="MedianHouseValue",
    data=df,
    scatter_kws={"alpha": 0.3},
    order=2,
    line_kws={"color": "red"}
)
plt.title("Average Rooms vs. Median House Value (Polynomial Trendline)")
plt.xlabel("Average Rooms")
plt.ylabel("Median House Value")
plt.savefig(f"{output_dir}/04_rooms_vs_value_trendline.png", dpi=300)
plt.close()

# 5. Boxplot of House Value by Discretized House Age
age_bins = pd.cut(
    df["HouseAge"],
    bins=[0, 10, 20, 30, 40, 50, 60],
    labels=["0–10", "10–20", "20–30", "30–40", "40–50", "50–60"]
)

plt.figure(figsize=(10, 6))
sns.boxplot(x=age_bins, y=df["MedianHouseValue"])
plt.title("Median House Value by House Age Groups")
plt.xlabel("House Age (Years)")
plt.ylabel("Median House Value")
plt.savefig(f"{output_dir}/05_age_vs_value_boxplot.png", dpi=300)
plt.close()

# -------------------------------
# 5 NON-INFORMATIVE FIGURES
# -------------------------------

# 6. Population vs. House Age Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x="Population", y="HouseAge", data=df, alpha=0.5)
plt.title("Population vs. House Age")
plt.savefig(f"{output_dir}/06_population_vs_house_age.png", dpi=300)
plt.close()

# 7. Average Occupancy vs. Longitude
plt.figure(figsize=(8, 6))
sns.scatterplot(x="Longitude", y="AverageOccupancy", data=df, alpha=0.5)
plt.title("Average Occupancy vs. Longitude")
plt.savefig(f"{output_dir}/07_occupancy_vs_longitude.png", dpi=300)
plt.close()

# 8. Raw Boxplot of Average Rooms
plt.figure(figsize=(8, 6))
sns.boxplot(y="AverageRooms", data=df)
plt.title("Raw Boxplot of Average Rooms (With Outliers)")
plt.savefig(f"{output_dir}/08_raw_boxplot_average_rooms.png", dpi=300)
plt.close()

# 9. Latitude vs. Longitude Without Color
plt.figure(figsize=(8, 6))
plt.scatter(df["Longitude"], df["Latitude"], alpha=0.5)
plt.title("Latitude vs. Longitude (No Price Information)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig(f"{output_dir}/09_latitude_vs_longitude_plain.png", dpi=300)
plt.close()

# 10. Histogram of Population (Linear Scale)
plt.figure(figsize=(8, 6))
plt.hist(df["Population"], bins=50)
plt.title("Histogram of Population (Linear Scale)")
plt.xlabel("Population")
plt.ylabel("Frequency")
plt.savefig(f"{output_dir}/10_population_histogram_linear.png", dpi=300)
plt.close()

print("✅ All figures generated successfully!")
print(f"📁 Saved in: {os.path.abspath(output_dir)}")
