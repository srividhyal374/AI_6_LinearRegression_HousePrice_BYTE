import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv("House Price Prediction Dataset.csv")

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns.tolist())


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

# Price is the value we want to predict.
# Id is only an identifier, so it is not used for prediction.

X = data.drop(columns=["Price", "Id"])
y = data["Price"]


# ============================================================
# 3. DEFINE NUMERIC AND CATEGORICAL FEATURES
# ============================================================

numeric_features = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "YearBuilt"
]

categorical_features = [
    "Location",
    "Condition",
    "Garage"
]


# ============================================================
# 4. DATA PREPROCESSING
# ============================================================

# Numerical data:
# Missing values are replaced with the median.

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])


# Categorical data:
# Missing values are replaced with the most common value.
# OneHotEncoder converts text categories into numerical values.

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


# Combine both preprocessing methods.

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# 5. CREATE LINEAR REGRESSION MODEL
# ============================================================

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])


# ============================================================
# 6. SPLIT DATA INTO TRAINING AND TESTING DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 7. TRAIN THE MODEL
# ============================================================

model.fit(X_train, y_train)

print("\nModel training completed successfully!")


# ============================================================
# 8. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 9. EVALUATE THE MODEL
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\n========== MODEL EVALUATION ==========")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# 10. SAVE EVALUATION METRICS
# ============================================================

with open("evaluation_metrics.txt", "w") as file:

    file.write("Linear Regression - House Price Prediction\n")
    file.write("-------------------------------------------\n")

    file.write(f"MAE  : {mae:.2f}\n")
    file.write(f"RMSE : {rmse:.2f}\n")
    file.write(f"R2   : {r2:.4f}\n")


# ============================================================
# 11. CREATE RESIDUAL PLOT
# ============================================================

residuals = y_test - y_pred

plt.figure(figsize=(8, 5))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.7
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Price")
plt.ylabel("Residuals")

plt.title(
    "Residual Plot - House Price Prediction"
)

plt.tight_layout()

plt.savefig(
    "residual_plot.png"
)

plt.show()


# ============================================================
# 12. CREATE SAMPLE PREDICTIONS
# ============================================================

sample_predictions = pd.DataFrame({
    "Actual Price": y_test.values[:10],
    "Predicted Price": y_pred[:10]
})


print("\n========== SAMPLE PREDICTIONS ==========")

print(sample_predictions.to_string(index=False))


# ============================================================
# 13. SAVE SAMPLE PREDICTIONS
# ============================================================

sample_predictions.to_csv(
    "sample_predictions.csv",
    index=False
)


# ============================================================
# 14. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    model,
    "house_price_linear_regression_model.pkl"
)


# ============================================================
# 15. COMPLETION MESSAGE
# ============================================================

print("\n========== PROJECT COMPLETED ==========")

print("Saved files:")
print("1. evaluation_metrics.txt")
print("2. residual_plot.png")
print("3. sample_predictions.csv")
print("4. house_price_linear_regression_model.pkl")

print("\nHouse Price Prediction project completed successfully!")