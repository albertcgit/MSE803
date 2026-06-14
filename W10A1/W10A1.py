import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline


def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.drop(columns=[col for col in df.columns if "Unnamed" in col])
    df = df.drop_duplicates().dropna()

    # Remove salary outliers via IQR
    Q1, Q3 = df["Salary"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df = df[(df["Salary"] >= Q1 - 1.5 * IQR) & (df["Salary"] <= Q3 + 1.5 * IQR)]
    return df


def build_models(X_train, y_train):
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Polynomial Regression: scale -> add x^2/x^3 features -> fit line
    poly_model = Pipeline([
        ("scaler", StandardScaler()),
        ("poly",   PolynomialFeatures(degree=3, include_bias=False)),
        ("model",  LinearRegression())
    ])
    poly_model.fit(X_train, y_train)

    return linear_model, poly_model


def evaluate(y_test, y_pred, label):
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"\n{label}")
    print(f"  MAE  : ${mae:>12,.2f}   avg dollar error")
    print(f"  MSE  : ${mse:>12,.2f}   penalises large errors (squared units)")
    print(f"  RMSE : ${rmse:>12,.2f}   like MAE but harsher on big mistakes")
    return mae, mse, rmse


def plot_results(df, X_test, y_test, x_range, y_line_lr, y_line_poly):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Salary Regression Analysis", fontsize=13, fontweight="bold")

    # Raw data relationship
    axes[0].scatter(df["YearsExperience"], df["Salary"], color="#4C72B0", alpha=0.75, edgecolors="white", s=60)
    axes[0].set_title("Experience vs Salary")
    axes[0].set_xlabel("Years of Experience")
    axes[0].set_ylabel("Salary ($)")

    # Linear regression fit
    axes[1].scatter(X_test, y_test, color="#4C72B0", label="Actual", zorder=3, edgecolors="white", s=60)
    axes[1].plot(x_range, y_line_lr, color="tomato", linewidth=2.5, label="Linear fit")
    axes[1].set_title("Model 1: Linear Regression")
    axes[1].set_xlabel("Years of Experience")
    axes[1].set_ylabel("Salary ($)")
    axes[1].legend(fontsize=8)

    # Polynomial regression fit
    axes[2].scatter(X_test, y_test, color="#4C72B0", label="Actual", zorder=3, edgecolors="white", s=60)
    axes[2].plot(x_range, y_line_poly, color="#55A868", linewidth=2.5, label="Poly fit (d=3)")
    axes[2].set_title("Model 2: Polynomial Regression (d=3)")
    axes[2].set_xlabel("Years of Experience")
    axes[2].set_ylabel("Salary ($)")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("salary_regression_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # 1. Load and clean
    df = load_and_clean("salary-dataset__1_.csv")

    # 2. Split: 80% train, 20% test
    X = df[["YearsExperience"]]
    y = df["Salary"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train both models
    linear_model, poly_model = build_models(X_train, y_train)

    # 4. Evaluate
    mae_lr,   mse_lr,   rmse_lr   = evaluate(y_test, linear_model.predict(X_test), "Linear Regression")
    mae_poly, mse_poly, rmse_poly = evaluate(y_test, poly_model.predict(X_test),   "Polynomial Regression (d=3)")

    # 5. Plot
    x_range = pd.DataFrame(
        np.linspace(df["YearsExperience"].min(), df["YearsExperience"].max(), 200),
        columns=["YearsExperience"]
    )
    plot_results(df, X_test, y_test, x_range, linear_model.predict(x_range), poly_model.predict(x_range))

    winner = "Polynomial" if rmse_poly < rmse_lr else "Linear"
    print(f"\nBetter model by RMSE: {winner} Regression")
    print("Plot saved: salary_regression_analysis.png")

    # Predict salary for 14, 14.5, and 15 years of experience
    new_experience = pd.DataFrame({"YearsExperience": [14, 14.5, 15]})
    pred_lr   = linear_model.predict(new_experience)
    pred_poly = poly_model.predict(new_experience)

    print("\nSalary Predictions")
    print(f"{'Years':<10} {'Linear':>15} {'Polynomial':>15}")
    print("-" * 42)
    for yrs, lr, poly in zip(new_experience["YearsExperience"], pred_lr, pred_poly):
        print(f"{yrs:<10} ${lr:>14,.2f} ${poly:>14,.2f}")