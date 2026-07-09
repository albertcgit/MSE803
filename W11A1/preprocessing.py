import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(file_path, target):

    # Load dataset
    df = pd.read_csv(file_path)

    # Basic cleaning
    df = df.dropna()
    df = df.drop_duplicates()

    # ✅ Optional: handle time column (for airline dataset)
    if "Month" in df.columns:
        df["Month"] = pd.to_datetime(df["Month"])
        df = df.set_index("Month")

    # Encode categorical variables (if any)
    df = pd.get_dummies(df, drop_first=True)

    # Split features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, y