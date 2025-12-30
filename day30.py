# ===============================
# Calories Burnt Prediction
# Day 30 Machine Learning Project
# ===============================

# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor

# ===============================
# Data Collection & Processing
# ===============================

# Load datasets
calories = pd.read_csv("calories.csv")
exercise_data = pd.read_csv("exercise.csv")

# Combine datasets
calories_data = pd.concat([exercise_data, calories["Calories"]], axis=1)

# ===============================
# Basic Data Inspection
# ===============================

print("Shape of dataset:", calories_data.shape)
print("\nDataset Info:")
print(calories_data.info())

print("\nMissing Values:")
print(calories_data.isnull().sum())

print("\nStatistical Summary:")
print(calories_data.describe())

# ===============================
# Exploratory Data Analysis (EDA)
# ===============================

sns.set(style="whitegrid")

# Gender count plot
plt.figure(figsize=(6,4))
sns.countplot(x="Gender", data=calories_data)
plt.title("Gender Distribution")
plt.show()

# Age distribution
plt.figure(figsize=(6,4))
sns.histplot(calories_data["Age"], kde=True)
plt.title("Age Distribution")
plt.show()

# Height distribution
plt.figure(figsize=(6,4))
sns.histplot(calories_data["Height"], kde=True)
plt.title("Height Distribution")
plt.show()

# Weight distribution
plt.figure(figsize=(6,4))
sns.histplot(calories_data["Weight"], kde=True)
plt.title("Weight Distribution")
plt.show()

# ===============================
# Correlation Analysis
# ===============================

correlation = calories_data.corr(numeric_only=True)

plt.figure(figsize=(10,10))
sns.heatmap(
    correlation,
    cbar=True,
    square=True,
    fmt=".1f",
    annot=True,
    annot_kws={"size":8},
    cmap="Blues"
)
plt.title("Correlation Heatmap")
plt.show()

# ===============================
# Data Preprocessing
# ===============================

# Convert categorical column to numerical
calories_data["Gender"] = calories_data["Gender"].map({
    "male": 0,
    "female": 1
})

# Feature and target separation
X = calories_data.drop(columns=["User_ID", "Calories"], axis=1)
Y = calories_data["Calories"]

# ===============================
# Train-Test Split
# ===============================

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# ===============================
# Model Training
# ===============================

model = XGBRegressor()
model.fit(X_train, Y_train)

# ===============================
# Model Evaluation
# ===============================

# Predictions
test_data_prediction = model.predict(X_test)

# Mean Absolute Error
mae = metrics.mean_absolute_error(Y_test, test_data_prediction)
print("Mean Absolute Error =", mae)

# ===============================
# END OF PROJECT
# ===============================
