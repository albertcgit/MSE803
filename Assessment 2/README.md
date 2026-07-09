# MSE803 Assessment 2 — Fitness Tracking App Data Analysis

This project analyses a 5,000-user fitness tracking app dataset for MSE803: Data Analytics, Assessment 2 (Case Study with Presentation). It covers:

- **Task 1**: Advanced statistical analysis (regression, classification, clustering)
- **Task 2**: Ethical and culturally relevant data analysis

## Files in this project

| File | Description |
|---|---|
| `Assessment 2.py` | Main Python script — runs all data cleaning, statistics, models, and saves charts |
| `requirements.txt` | Python packages needed to run the script |
| `dataset for assignment 2.csv` | The dataset (you provide this — see below) |
| `MSE803_Assessment2_Report_Academic.docx` | Full written report (Task 1 + Task 2), academic formatting |
| `MSE803_Assessment2_Presentation.pptx` | Slide deck with speaker notes |
| `outputs/` | Created when you run the script — contains all charts and `results_summary.txt` |

## How to run the analysis

### 1. Requirements
- Python 3.9 or newer
- Install packages:
  ```
  pip install -r requirements.txt
  ```

### 2. Dataset
Place your CSV file in the **same folder** as `Assessment 2.py`, named:
```
dataset for assignment 2.csv
```
If your file has a different name, update the `DATA_PATH` line near the top of the script.

### 3. Run
```
python3 "Assessment 2.py"
```

### 4. Output
A new `outputs/` folder appears next to the script, containing:
- 11 PNG chart files (correlation heatmap, boxplots, regression plots, both confusion matrices, elbow method, cluster plot)
- `results_summary.txt` — every number generated during the run (R², accuracy, p-values, cluster stats, etc.)

The script always looks for the dataset in its own folder, so it works no matter which directory you run the command from.

## What the script does

1. **Data cleaning** — checks for missing values, duplicates, and outliers (IQR method); encodes categorical columns
2. **Exploratory analysis** — correlation heatmap, boxplots by Activity Level / Location / Gender
3. **Hypothesis testing** — ANOVA (Location) and t-test (Gender) on Calories Burned
4. **Regression** — linear regression predicting Calories Burned, evaluated with R², RMSE, MAE, and 5-fold cross-validation
5. **Classification** — two Random Forest / Logistic Regression models predicting Activity Level:
   - One using engagement features (App Sessions, Distance, Calories) — this one shows data leakage (100% accuracy, a red flag, not a real result)
   - One using only demographic features (Age, Gender, Location) — a realistic, leakage-free result (~33% accuracy)
6. **Clustering** — K-means (k=3) user segmentation, validated with the elbow method and silhouette score

## Key findings

- App Sessions is the strongest driver of engagement (r=0.94 with Activity Level)
- Regression model: R²=0.638 for predicting Calories Burned
- The first classification model's 100% accuracy is data leakage, not a genuine result — confirmed by feature importance and by building a second, demographic-only model (33.4% accuracy, near random)
- No statistically significant difference in engagement by Location (p=0.229) or Gender (p=0.507)
- Clusters are demographically balanced, supporting the fairness discussion in Task 2

## Reports and presentation

- The Word report and PowerPoint presentation already reflect the results from this script's output — you don't need to regenerate them unless you change the analysis or dataset.
- If you re-run the script with a different dataset, let me know and I can update the report/presentation to match the new numbers.

## Before submission

- Fill in your name and Student ID on the report and presentation title pages
- In the Word report, update the Table of Contents once (select all, press F9, or right-click → Update Field)
