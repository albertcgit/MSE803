import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "dataset for assignment 2.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

summary_lines = []

def log(msg):
    print(msg)
    summary_lines.append(str(msg))


# Load dataset
df = pd.read_csv(DATA_PATH)
log(f"Initial shape: {df.shape}")
log(f"Columns: {list(df.columns)}")
df.columns = [c.strip() for c in df.columns]

# Check and handle missing values
missing = df.isnull().sum()
log("Missing values per column:")
log(missing.to_string())
if missing.sum() > 0:
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.int64, np.float64]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    log("Missing values imputed.")
else:
    log("No missing values found.")

# Check and remove duplicates
dupes = df.duplicated().sum()
log(f"Duplicate rows found: {dupes}")
if dupes > 0:
    df = df.drop_duplicates()
    log(f"Duplicates removed. New shape: {df.shape}")

# Outlier check using IQR, flagged only, not removed
numeric_cols = ["Age", "App Sessions", "Distance Travelled (km)", "Calories Burned"]
log("Outlier check (IQR method):")
for col in numeric_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    log(f"{col}: {len(outliers)} potential outliers (bounds: {lower:.1f} to {upper:.1f})")

# Encode categorical variables
activity_order = {"Sedentary": 0, "Moderate": 1, "Active": 2}
df["Activity_Level_Ordinal"] = df["Activity Level"].map(activity_order)

le_gender = LabelEncoder()
df["Gender_Encoded"] = le_gender.fit_transform(df["Gender"])

df_location_dummies = pd.get_dummies(df["Location"], prefix="Location", drop_first=True)
df = pd.concat([df, df_location_dummies], axis=1)

log("Encoding applied: Activity Level ordinal, Gender label encoded, Location one-hot encoded")

# Descriptive statistics
log("Descriptive statistics:")
log(df[numeric_cols].describe().to_string())

log("Categorical distributions:")
for col in ["Gender", "Activity Level", "Location"]:
    log(f"{col}:")
    log(df[col].value_counts().to_string())

# Correlation heatmap
plt.figure(figsize=(7, 5))
corr_cols = ["Age", "App Sessions", "Distance Travelled (km)", "Calories Burned", "Activity_Level_Ordinal"]
corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_correlation_heatmap.png", dpi=150)
plt.close()
log("Saved 01_correlation_heatmap.png")
log(corr.to_string())

# Boxplots by activity level
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col in zip(axes, ["App Sessions", "Distance Travelled (km)", "Calories Burned"]):
    sns.boxplot(data=df, x="Activity Level", y=col, order=["Sedentary", "Moderate", "Active"], ax=ax)
    ax.set_title(f"{col} by Activity Level")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_boxplots_by_activity.png", dpi=150)
plt.close()
log("Saved 02_boxplots_by_activity.png")

# Boxplots by location and gender
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df, x="Location", y="Calories Burned", ax=axes[0])
axes[0].set_title("Calories Burned by Location")
sns.boxplot(data=df, x="Gender", y="Calories Burned", ax=axes[1])
axes[1].set_title("Calories Burned by Gender")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_boxplots_location_gender.png", dpi=150)
plt.close()
log("Saved 03_boxplots_location_gender.png")

# Scatter plot with regression line
plt.figure(figsize=(7, 5))
sns.regplot(data=df, x="Distance Travelled (km)", y="Calories Burned",
            scatter_kws={"alpha": 0.3, "s": 10}, line_kws={"color": "red"})
plt.title("Distance Travelled vs Calories Burned")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_scatter_distance_calories.png", dpi=150)
plt.close()
log("Saved 04_scatter_distance_calories.png")

# ANOVA across location groups
groups = [df[df["Location"] == loc]["Calories Burned"] for loc in df["Location"].unique()]
f_stat, p_val = stats.f_oneway(*groups)
log(f"ANOVA Calories Burned across Location: F={f_stat:.4f}, p={p_val:.4f}")
log("Significant difference" if p_val < 0.05 else "No significant difference")

# T-test by gender
male = df[df["Gender"] == "Male"]["Calories Burned"]
female = df[df["Gender"] == "Female"]["Calories Burned"]
t_stat, p_val_t = stats.ttest_ind(male, female)
log(f"T-test Calories Burned by Gender: t={t_stat:.4f}, p={p_val_t:.4f}")
log("Significant difference" if p_val_t < 0.05 else "No significant difference")

# Regression model predicting Calories Burned
feature_cols_reg = ["Age", "App Sessions", "Distance Travelled (km)",
                     "Activity_Level_Ordinal", "Gender_Encoded"] + list(df_location_dummies.columns)
X_reg = df[feature_cols_reg]
y_reg = df["Calories Burned"]

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=RANDOM_STATE)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
log(f"Linear Regression: R2={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

log("Regression coefficients:")
for feat, coef in zip(feature_cols_reg, lin_reg.coef_):
    log(f"{feat}: {coef:.3f}")
log(f"Intercept: {lin_reg.intercept_:.3f}")

# Cross-validation
cv_scores = cross_val_score(lin_reg, X_reg, y_reg, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE), scoring="r2")
log(f"5-fold CV R2 scores: {np.round(cv_scores, 4)}")
log(f"Mean CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, alpha=0.3, s=10)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Calories Burned")
plt.ylabel("Residuals")
plt.title("Residual Plot - Linear Regression")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_regression_residuals.png", dpi=150)
plt.close()
log("Saved 05_regression_residuals.png")

# Predicted vs actual plot
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.3, s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Calories Burned")
plt.ylabel("Predicted Calories Burned")
plt.title("Predicted vs Actual - Linear Regression")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_regression_pred_vs_actual.png", dpi=150)
plt.close()
log("Saved 06_regression_pred_vs_actual.png")

# Classification model predicting Activity Level
feature_cols_clf = ["Age", "App Sessions", "Distance Travelled (km)",
                     "Calories Burned", "Gender_Encoded"] + list(df_location_dummies.columns)
X_clf = df[feature_cols_clf]
y_clf = df["Activity Level"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
)

scaler = StandardScaler()
X_train_c_scaled = scaler.fit_transform(X_train_c)
X_test_c_scaled = scaler.transform(X_test_c)

log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
log_reg.fit(X_train_c_scaled, y_train_c)
y_pred_log = log_reg.predict(X_test_c_scaled)

rf_clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
rf_clf.fit(X_train_c, y_train_c)
y_pred_rf = rf_clf.predict(X_test_c)

for name, y_pred_this in [("Logistic Regression", y_pred_log), ("Random Forest", y_pred_rf)]:
    acc = accuracy_score(y_test_c, y_pred_this)
    prec = precision_score(y_test_c, y_pred_this, average="weighted")
    rec = recall_score(y_test_c, y_pred_this, average="weighted")
    f1 = f1_score(y_test_c, y_pred_this, average="weighted")
    log(f"{name}: accuracy={acc:.4f}, precision={prec:.4f}, recall={rec:.4f}, f1={f1:.4f}")

log("Classification report Random Forest:")
log(classification_report(y_test_c, y_pred_rf))

# Feature importance from random forest
importances = pd.Series(rf_clf.feature_importances_, index=feature_cols_clf).sort_values(ascending=False)
log("Random Forest feature importances:")
log(importances.to_string())

plt.figure(figsize=(7, 5))
importances.plot(kind="barh")
plt.xlabel("Importance")
plt.title("Feature Importance - Random Forest")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_feature_importance.png", dpi=150)
plt.close()
log("Saved 07_feature_importance.png")

# Confusion matrix
cm = confusion_matrix(y_test_c, y_pred_rf, labels=["Sedentary", "Moderate", "Active"])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Sedentary", "Moderate", "Active"],
            yticklabels=["Sedentary", "Moderate", "Active"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_confusion_matrix.png", dpi=150)
plt.close()
log("Saved 08_confusion_matrix.png")

# Note: classifier above uses engagement features that likely leak the label, showing near perfect accuracy
# Second classifier below uses only demographic features, no leakage, for a realistic result
feature_cols_clf2 = ["Age", "Gender_Encoded"] + list(df_location_dummies.columns)
X_clf2 = df[feature_cols_clf2]
y_clf2 = df["Activity Level"]

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_clf2, y_clf2, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf2
)

rf_clf2 = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
rf_clf2.fit(X_train_d, y_train_d)
y_pred_rf2 = rf_clf2.predict(X_test_d)

acc2 = accuracy_score(y_test_d, y_pred_rf2)
prec2 = precision_score(y_test_d, y_pred_rf2, average="weighted")
rec2 = recall_score(y_test_d, y_pred_rf2, average="weighted")
f1_2 = f1_score(y_test_d, y_pred_rf2, average="weighted")
log(f"Demographics only classifier Random Forest: accuracy={acc2:.4f}, precision={prec2:.4f}, recall={rec2:.4f}, f1={f1_2:.4f}")
log("This low accuracy is expected and realistic, since demographics alone do not strongly determine activity level")

log("Classification report demographics only classifier:")
log(classification_report(y_test_d, y_pred_rf2))

cm2 = confusion_matrix(y_test_d, y_pred_rf2, labels=["Sedentary", "Moderate", "Active"])
plt.figure(figsize=(6, 5))
sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Sedentary", "Moderate", "Active"],
            yticklabels=["Sedentary", "Moderate", "Active"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Demographics Only Classifier")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08b_confusion_matrix_demographics_only.png", dpi=150)
plt.close()
log("Saved 08b_confusion_matrix_demographics_only.png")

# K-means clustering on engagement features
cluster_features = ["App Sessions", "Distance Travelled (km)", "Calories Burned"]
X_cluster = df[cluster_features]
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

inertias = []
k_range = range(1, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km.fit(X_cluster_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(list(k_range), inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_elbow_method.png", dpi=150)
plt.close()
log("Saved 09_elbow_method.png")

# Final clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_cluster_scaled)

sil_score = silhouette_score(X_cluster_scaled, df["Cluster"])
log(f"K-Means k=3 silhouette score: {sil_score:.4f}")

log("Cluster profiles:")
cluster_profile = df.groupby("Cluster")[cluster_features + ["Age"]].mean()
log(cluster_profile.to_string())

log("Cluster size:")
log(df["Cluster"].value_counts().sort_index().to_string())

crosstab = pd.crosstab(df["Cluster"], df["Activity Level"])
log("Cluster vs Activity Level:")
log(crosstab.to_string())

# PCA projection for cluster visualization
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_cluster_scaled)
df["PCA1"], df["PCA2"] = X_pca[:, 0], X_pca[:, 1]

plt.figure(figsize=(7, 6))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", alpha=0.5, s=15)
plt.title(f"K-Means Clusters PCA Projection, explained variance {pca.explained_variance_ratio_.sum():.1%}")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_cluster_pca.png", dpi=150)
plt.close()
log("Saved 10_cluster_pca.png")

# Cluster demographics for ethical and cultural discussion
log("Cluster composition by Location percent:")
log(pd.crosstab(df["Cluster"], df["Location"], normalize="index").mul(100).round(1).to_string())
log("Cluster composition by Gender percent:")
log(pd.crosstab(df["Cluster"], df["Gender"], normalize="index").mul(100).round(1).to_string())
log("Cluster mean Age:")
log(df.groupby("Cluster")["Age"].mean().round(1).to_string())

# Demographic breakdown for ethical and cultural discussion
log("Mean engagement by Location:")
log(df.groupby("Location")[["App Sessions", "Distance Travelled (km)", "Calories Burned"]].mean().round(1).to_string())

log("Mean engagement by Gender:")
log(df.groupby("Gender")[["App Sessions", "Distance Travelled (km)", "Calories Burned"]].mean().round(1).to_string())

df["Age_Group"] = pd.cut(df["Age"], bins=[17, 29, 39, 49, 60], labels=["18-29", "30-39", "40-49", "50-59"])
log("Mean engagement by Age group:")
log(df.groupby("Age_Group")[["App Sessions", "Distance Travelled (km)", "Calories Burned"]].mean().round(1).to_string())

with open(f"{OUTPUT_DIR}/results_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))

log("All charts and results_summary.txt saved to outputs folder")
