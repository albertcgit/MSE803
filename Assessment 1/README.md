# MSE803 Data Analytics — Assessment 1

## Avon River Environmental Analysis

This repository contains the Python analysis script and dataset for MSE803 Assessment 1 at Yoobee Colleges. This README is also included in Section 1.3 of the assessment report.

---

## Repository Structure

Assessment-1/

├── Assessment 1.py
├── Data_Set_Assignmnet_1-V0.1_20426.xlsx
├── avon_output/
│   └── avon_river_clean.csv
└── README.md

---

## Requirements

- Python 3.8 or newer
- pandas — data loading and cleaning
- scipy — statistical calculation
- numpy — numerical operations
- openpyxl — reading Excel files
- Microsoft Power BI Desktop — free download from powerbi.microsoft.com

---

## Install Dependencies

```bash
pip install pandas scipy numpy openpyxl
```

---

## How to Run

1. Place `Assessment 1.py` and `Data_Set_Assignmnet_1-V0.1_20426.xlsx` in the same folder
2. Run the script:

```bash
python "Assessment 1.py"
```

3. The script prints statistics and correlations to the console
4. Clean CSV saved to `avon_output/avon_river_clean.csv`

---

## Output

### Console
- Descriptive statistics
- Per-site averages
- Species counts
- Stress events (temp > 18°C, DO < 7.0 mg/L)
- Pearson correlations

### Files
- `avon_output/avon_river_clean.csv` — cleaned and merged dataset ready for Power BI

---

## Power BI

Load `avon_output/avon_river_clean.csv` into Power BI Desktop to recreate the visualisations.

### Charts
- Temperature over time by site — line chart with 18°C stress threshold
- Average water quality by site — clustered bar chart with thresholds

See Section 1.5 of the assessment report for full step-by-step setup instructions.

---
