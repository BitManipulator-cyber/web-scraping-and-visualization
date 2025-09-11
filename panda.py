import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import argparse


df = pd.read_csv("Samples/Companies.csv")


df["Employees"] = df["Employees"].str.replace(",", "").astype(float)

df = df.dropna(subset=["Market Cap (B)", "Revenue (Billions USD)"])

le_industry = LabelEncoder()
le_country = LabelEncoder()
df["Industry"] = le_industry.fit_transform(df["Industry"])
df["Country"] = le_country.fit_transform(df["Country"])


X = df.drop(columns=["Revenue (Billions USD)", "Company Name"])
y = df["Revenue (Billions USD)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


rmse_all = np.sqrt(mean_squared_error(y_test, y_pred))
print("Initial R² Score:", r2_score(y_test, y_pred))
print("Initial RMSE:", rmse_all)


results_all = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print("\nSample Predictions (All Features):")
print(results_all.head(10))


importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature Importances:")
print(importances.sort_values(ascending=False))


importances.sort_values().plot(kind="barh", figsize=(8,6))
plt.title("Feature Importances (Decision Tree)")
plt.show()


top_features = importances.sort_values(ascending=False).head(3).index
X_train_sel, X_test_sel = X_train[top_features], X_test[top_features]

model_sel = DecisionTreeRegressor(random_state=42)
model_sel.fit(X_train_sel, y_train)
y_pred_sel = model_sel.predict(X_test_sel)


rmse_sel = np.sqrt(mean_squared_error(y_test, y_pred_sel))
print("\nRetrained with Top Features")
print("R² Score:", r2_score(y_test, y_pred_sel))
print("RMSE:", rmse_sel)


results_sel = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_sel})
print("\nSample Predictions (Top Features):")
print(results_sel.head(10))


def predict_sample(sample_dict):
    sample_df = pd.DataFrame([sample_dict])
   
    if "Industry" in sample_df:
        sample_df["Industry"] = le_industry.transform(sample_df["Industry"])
    if "Country" in sample_df:
        sample_df["Country"] = le_country.transform(sample_df["Country"])
    # Ensure columns match training set
    sample_df = sample_df.reindex(columns=X.columns, fill_value=0)
    pred = model.predict(sample_df)
    print("\nPredicted Revenue for Sample Input:", pred[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Revenue using Decision Tree")
    parser.add_argument("--rank", type=int, required=False, default=101, help="Company Rank")
    parser.add_argument("--industry", type=str, required=False, default="Technology", help="Industry")
    parser.add_argument("--employees", type=float, required=False, default=50000, help="Number of Employees")
    parser.add_argument("--country", type=str, required=False, default="United States", help="Country")
    parser.add_argument("--founded", type=int, required=False, default=2020, help="Founded Year")
    parser.add_argument("--marketcap", type=float, required=False, default=200.0, help="Market Cap (B)")

    args = parser.parse_args()

    sample_input = {
        "Rank": args.rank,
        "Industry": args.industry,
        "Employees": args.employees,
        "Country": args.country,
        "Founded": args.founded,
        "Market Cap (B)": args.marketcap
    }

    predict_sample(sample_input)
