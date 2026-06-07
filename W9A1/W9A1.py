"""
Week 9 - Activity 1: KMeans Clustering on Fitness App User Data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. LOAD DATA
df = pd.read_excel("Fitness_App_User_Data.xlsx")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# 2. DATA CLEANING

# 2a. Check missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 2b. Check duplicates
print(f"\nDuplicate rows: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)

# 2c. Check data types
print("\nData types:")
print(df.dtypes)

# 2d. Encode categorical columns as numbers
df["Gender_Enc"] = df["Gender"].map({"Male": 0, "Female": 1})
df["Sub_Enc"] = df["Subscription_Type"].map({"Free": 0, "Basic": 1, "Premium": 2})

print("\nCleaning complete. No missing values or duplicates found.")

# 3. FEATURE SELECTION & SCALING
features = ["Age", "Workouts_per_Week", "Avg_Session_Duration_Min",
            "Steps_per_Day", "Gender_Enc", "Sub_Enc"]

X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. FIND OPTIMAL K (Elbow + Silhouette)
inertias = []
silhouettes = []
K_range = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(K_range, inertias, "bo-")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia")
axes[0].axvline(x=3, color="red", linestyle="--", label="k=3")
axes[0].legend()

axes[1].plot(K_range, silhouettes, "go-")
axes[1].set_title("Silhouette Scores")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].axvline(x=3, color="red", linestyle="--", label="k=3")
axes[1].legend()

plt.tight_layout()
plt.savefig("elbow_silhouette.png", dpi=150)
plt.close()
print("Saved: elbow_silhouette.png")

# 5. APPLY KMEANS WITH k=3
km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = km_final.fit_predict(X_scaled)

print("\nCluster distribution:")
print(df["Cluster"].value_counts().sort_index())

# 6. INTERPRET CLUSTERS
cluster_summary = df.groupby("Cluster")[features + ["Churned"]].mean().round(2)
print("\nCluster Profiles:")
print(cluster_summary)

# 7. VISUALISATIONS

# 7a. Scatter plot: Steps vs Workouts coloured by cluster
plt.figure(figsize=(8, 5))
colors = ["#e74c3c", "#3498db", "#2ecc71"]
for c in range(3):
    subset = df[df["Cluster"] == c]
    plt.scatter(subset["Steps_per_Day"], subset["Workouts_per_Week"],
                c=colors[c], label=f"Cluster {c}", alpha=0.7, s=60)
plt.title("KMeans Clusters: Steps per Day vs Workouts per Week")
plt.xlabel("Steps per Day")
plt.ylabel("Workouts per Week")
plt.legend()
plt.tight_layout()
plt.savefig("scatter_clusters.png", dpi=150)
plt.close()
print("Saved: scatter_clusters.png")

# 7b. Heatmap of cluster profiles
plt.figure(figsize=(8, 4))
sns.heatmap(cluster_summary[features], annot=True, fmt=".1f",
            cmap="YlOrRd", linewidths=0.5)
plt.title("Cluster Profile Heatmap (Mean Feature Values)")
plt.tight_layout()
plt.savefig("cluster_heatmap.png", dpi=150)
plt.close()
print("Saved: cluster_heatmap.png")

# 7c. Bar chart - Churn rate per cluster
churn_rate = df.groupby("Cluster")["Churned"].mean() * 100
plt.figure(figsize=(6, 4))
churn_rate.plot(kind="bar", color=colors, edgecolor="black")
plt.title("Churn Rate by Cluster (%)")
plt.xlabel("Cluster")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("churn_by_cluster.png", dpi=150)
plt.close()
print("Saved: churn_by_cluster.png")

# 8. SAVE LABELLED DATA
df.to_excel("Fitness_Clustered.xlsx", index=False)
print("\nSaved labelled dataset: Fitness_Clustered.xlsx")
print("\nDone!")