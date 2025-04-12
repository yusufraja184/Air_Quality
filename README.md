import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Load dataset
df = pd.read_csv(r"C:\Users\yusuf raja\Downloads\Air_Quality.csv")

# Optional: Clean column names
df.columns = df.columns.str.strip()

# ===============================
# 1.Data Cleaning
# ===============================
# Drop rows with missing location info
df_clean = df.dropna(subset=["Geo Place Name", "Geo Join ID"]).copy()

# Check for missing values
print("Missing values:\n", df_clean.isnull().sum())

# ===============================
# 2.Compare Air Quality in Different Locations
# ===============================
plt.figure(figsize=(14, 6))
top_locations = df_clean.groupby("Geo Place Name")["Data Value"].mean().sort_values(ascending=False).head(10)

sns.barplot(x=top_locations.values, y=top_locations.index, hue=top_locations.index, palette="coolwarm", dodge=False, legend=False)
plt.title("Top 10 Locations with Highest Average Air Pollution")
plt.xlabel("Average Pollution Level")
plt.ylabel("Location")
plt.tight_layout()
plt.show()

# ===============================
# 3.Spot Areas with the Worst Air Pollution
# ===============================
worst = df_clean.sort_values("Data Value", ascending=False).head(10)
print("Top 10 Most Polluted Records:\n", worst[["Geo Place Name", "Name", "Data Value", "Time Period"]])

# ===============================
# 4.Compare Pollution Between Area Types
# ===============================
plt.figure(figsize=(12, 6))
sns.boxplot(x="Geo Type Name", y="Data Value", data=df_clean)
plt.title("Air Pollution Comparison by Area Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ===============================
# 5.Seasonal Pollution Trends
# ===============================
df_clean["Season"] = df_clean["Time Period"].apply(lambda x: x.split()[0] if pd.notnull(x) and " " in x else x)

plt.figure(figsize=(10, 5))
sns.boxplot(x="Season", y="Data Value", data=df_clean)
plt.title("Pollution Levels by Season")
plt.tight_layout()
plt.show()

# ===============================
# Additional Analysis 1: T-Test between two area types
# ===============================
urban = df_clean[df_clean["Geo Type Name"] == "Urban"]["Data Value"].dropna()
suburban = df_clean[df_clean["Geo Type Name"] == "Suburban"]["Data Value"].dropna()

print(f"Urban sample size: {len(urban)}")
print(f"Suburban sample size: {len(suburban)}")

if len(urban) >= 2 and len(suburban) >= 2:
    t_stat, p_value = ttest_ind(urban, suburban, equal_var=False)
    print("\nT-Test: Urban vs Suburban Air Pollution Levels")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("→ Significant difference between Urban and Suburban air pollution levels.")
    else:
        print("→ No significant difference between Urban and Suburban air pollution levels.")
else:
    print("❌ Not enough data to perform T-test.")

# ===============================
# Additional Analysis 2: Heatmap of Season vs Geo Type Name
# ===============================
pivot_table = df_clean.pivot_table(values="Data Value", index="Geo Type Name", columns="Season", aggfunc="mean")

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5)
plt.title("Average Pollution by Season and Area Type")
plt.tight_layout()
plt.show()

# ===============================
# Additional Analysis 3: Met-style plot – Pollution trend by year
# ===============================
df_clean["Year"] = df_clean["Time Period"].apply(lambda x: x.split()[-1] if pd.notnull(x) and " " in x else None)
df_clean["Year"] = pd.to_numeric(df_clean["Year"], errors='coerce')

plt.figure(figsize=(12, 5))
sns.scatterplot(x="Year", y="Data Value", data=df_clean, alpha=0.4)
sns.lineplot(x="Year", y="Data Value", data=df_clean, estimator='mean', errorbar=None, color='red', label="Mean Trend")
plt.title("Pollution Levels Over Time")
plt.tight_layout()
plt.show()

# ===============================
# Additional Analysis 4: Scatter Plot – Pollution vs Year by Area Type
# ===============================
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_clean.dropna(subset=["Year", "Data Value"]), x="Year", y="Data Value", hue="Geo Type Name", alpha=0.6, palette="Set2")
plt.title("Scatter Plot: Pollution Level vs Year by Area Type")
plt.xlabel("Year")
plt.ylabel("Pollution Level (Data Value)")
plt.legend(title="Area Type")
plt.tight_layout()
plt.show()


