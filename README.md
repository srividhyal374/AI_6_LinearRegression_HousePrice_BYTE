# 🏠 House Price Prediction using Linear Regression

## 📌 Project Overview

This project predicts house prices using a Machine Learning model based on Linear Regression.

The model uses different house-related features such as area, number of bedrooms, bathrooms, floors, year built, location, condition, and garage availability to predict the house price.

The project includes data preprocessing, model training, evaluation, sample predictions, residual analysis, and a saved trained model.

---

## 📊 Dataset

### Dataset Name
House Price Prediction Dataset

### Dataset Source
Kaggle – House Price Prediction Dataset

### Dataset Size
The dataset contains approximately 2,000 house records.

### Features

- Id
- Area
- Bedrooms
- Bathrooms
- Floors
- YearBuilt
- Location
- Condition
- Garage
- Price

### Target Variable

**Price**

The `Price` column is the target variable that the model predicts.

The `Id` column is used only as an identifier and is not used for prediction.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

---

## 🔄 Data Preprocessing

The following preprocessing steps were performed:

1. The dataset was loaded using Pandas.
2. The `Id` column was removed because it is only an identifier.
3. The `Price` column was separated as the target variable.
4. Numerical features were separated from categorical features.
5. Missing numerical values were handled using the median.
6. Missing categorical values were handled using the most frequent value.
7. Categorical features were converted into numerical form using One-Hot Encoding.
8. The dataset was divided into training and testing sets.
9. 80% of the data was used for training and 20% for testing.

---

## 🤖 Machine Learning Algorithm

### Linear Regression

Linear Regression is a supervised Machine Learning algorithm used to predict a continuous numerical value.

In this project, Linear Regression is used to predict house prices based on the available house features.

The model was trained using the training dataset and evaluated using the test dataset.

---

## 📈 Model Evaluation

The model was evaluated using:

### MAE
Mean Absolute Error measures the average absolute difference between actual and predicted prices.

### RMSE
Root Mean Squared Error measures the square root of the average squared prediction error.

### R² Score
R² measures how well the model explains the variation in the target values.


### Actual Results

- MAE: 243241.98
- RMSE: 279859.73
- R²: -0.0067

The actual evaluation results are stored in:

`evaluation_metrics.txt`

---

## 📊 Residual Plot

A residual plot was created to analyze the difference between actual and predicted house prices.

The residual is calculated as:

**Residual = Actual Price − Predicted Price**

The generated residual plot is available as:

`residual_plot.png`

---

## 🔮 Sample Predictions

The project generates 10 sample predictions containing:

- Actual Price
- Predicted Price

These results are saved in:

`sample_predictions.csv`

---

## 💾 Saved Model

The trained Linear Regression model is saved as:

`house_price_linear_regression_model.pkl`

The saved model can be loaded later without training the model again.

---

## ▶️ How to Run the Project

### Step 1: Install the required libraries

Open the terminal and run:

```bash
pip install pandas numpy scikit-learn matplotlib joblib