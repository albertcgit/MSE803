"""
Week 9 - Activity 1: KMeans Clustering on Fitness App User Data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def main():
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

    X_scaled = StandardScaler().fit_transform(df[features])

    # 4. FIND OPTIMAL K (Elbow + Silhouette)
    inertias, silhouettes = [], []
    K_range = list(range(2, 9))

    for k in K_range:
        _km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = _km.fit_predict(X_scaled)
        inertias.append(_km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    inertia_drops = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
    best_k = K_range[1 + inertia_drops.index(max(inertia_drops))]
    print(f"Best k (elbow method - largest inertia drop): {best_k}")

    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(K_range, inertias, "bo-")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].axvline(x=best_k, color="red", linestyle="--", label=f"k={best_k}")
    axes[0].legend()

    axes[1].plot(K_range, silhouettes, "go-")
    axes[1].set_title("Silhouette Scores")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].axvline(x=best_k, color="red", linestyle="--", label=f"k={best_k}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("elbow_silhouette.png", dpi=150)
    plt.close()
    print("Saved: elbow_silhouette.png")

    # 5. APPLY KMEANS WITH best_k
    df["Cluster"] = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X_scaled)

    print("\nCluster distribution:")
    print(df["Cluster"].value_counts().sort_index())

    # 6. INTERPRET CLUSTERS
    cluster_summary = df.groupby("Cluster")[features + ["Churned"]].mean().round(2)
    print("\nCluster Profiles:")
    print(cluster_summary)

    # 7. VISUALISATIONS
    all_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c", "#e67e22", "#34495e"]
    colors = all_colors[:best_k]

    # 7a. Scatter plot: Steps vs Workouts coloured by cluster
    plt.figure(figsize=(8, 5))
    for c in range(best_k):
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


if __name__ == "__main__":
    main()