import os, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "Data_Set_Assignmnet_1-V0.1_20426.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "avon_output")
TEMP_THRESH = 18.0
DO_THRESH   = 7.0


class RiverAnalysis:

    def __init__(self, data_path=DATA_PATH, output_dir=OUTPUT_DIR):
        self.data_path  = data_path
        self.output_dir = output_dir
        self.wq = self.fp = self.merged = None
        os.makedirs(output_dir, exist_ok=True)

    # Load and split raw Excel into water-quality and fish-population tables
    def load(self):
        raw = pd.read_excel(self.data_path, header=1)
        self.wq = raw.iloc[:, :5].copy()
        self.wq.columns = ["site", "date", "temp", "ph", "do"]
        self.fp = raw.iloc[:, 6:11].copy()
        self.fp.columns = ["site", "date", "species", "count", "avg_size"]
        print(f"Loaded {len(self.wq)} rows for water quality and fish population.")

    # Parse dates, impute missing pH, drop unknown species, merge datasets
    def clean(self):
        for df in (self.wq, self.fp):
            df["date"] = pd.to_datetime(df["date"])

        missing_wq = self.wq.isnull().sum()
        missing_fp = self.fp.isnull().sum()
        print("Missing - Water Quality:", missing_wq[missing_wq > 0].to_dict())
        print("Missing - Fish Population:", missing_fp[missing_fp > 0].to_dict())

        self.wq["ph"] = self.wq.groupby("site")["ph"].transform(lambda x: x.fillna(x.median()))
        self.fp.dropna(subset=["species"], inplace=True)

        self.merged = pd.merge(self.wq, self.fp, on=["site", "date"]).sort_values("date")
        print(f"Merged: {len(self.merged)} rows | "
              f"{self.merged['date'].min().date()} to {self.merged['date'].max().date()}")

    # Print descriptive stats and stress event counts to console
    def summarise(self):
        print("\n--- DESCRIPTIVE STATISTICS ---")
        print(self.merged[["temp", "ph", "do", "count", "avg_size"]].describe().round(2))
        print("\n--- PER-SITE AVERAGES ---")
        print(self.merged.groupby("site")[["temp", "ph", "do", "count", "avg_size"]].mean().round(2))
        print("\n--- SPECIES COUNTS ---")
        print(self.merged["species"].value_counts().to_string())
        high_temp = (self.merged["temp"] > TEMP_THRESH).sum()
        low_do    = (self.merged["do"]   < DO_THRESH).sum()
        print(f"\nStress events: temp > {TEMP_THRESH}C = {high_temp}, DO < {DO_THRESH} = {low_do}")

    # Pearson correlation between water quality variables and fish metrics
    def correlate(self):
        print("\n--- PEARSON CORRELATIONS ---")
        for target in ["count", "avg_size"]:
            print(f"\nvs {target}:")
            for pred in ["temp", "ph", "do"]:
                d = self.merged[[pred, target]].dropna()
                r, p = stats.pearsonr(d[pred], d[target])
                sig = "significant" if p < 0.05 else "not significant"
                print(f"  {pred:<6} r={r:+.3f}  p={p:.4f}  ({sig})")

    # Export cleaned and merged dataset as CSV for Power BI
    def export_csv(self):
        path = os.path.join(self.output_dir, "avon_river_clean.csv")
        self.merged.to_csv(path, index=False)
        print(f"\nClean dataset exported: {path}")
        print(f"Columns: {list(self.merged.columns)}")
        print(f"Rows: {len(self.merged)}")

    # Run the full pipeline in order
    def run(self):
        print("=== AVON RIVER ANALYSIS | MSE803 Assessment 1 ===\n")
        self.load()
        self.clean()
        self.summarise()
        self.correlate()
        self.export_csv()
        print(f"\nDone. Outputs saved to: {os.path.abspath(self.output_dir)}")

if __name__ == "__main__":
    RiverAnalysis().run()