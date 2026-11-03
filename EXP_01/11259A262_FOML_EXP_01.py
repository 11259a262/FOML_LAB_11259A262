# Experiment 1: Cleaning Messy Data
# Aim: Clean the messy Titanic dataset by filling missing values,
# encoding categorical columns to numbers, and verifying on a new passenger.

import pandas as pd
import seaborn as sns

# Load the Titanic passenger table
data = sns.load_dataset("titanic")
data = data[["survived", "pclass", "sex", "age", "fare", "embarked"]]

# Count empty cells in each column
print("Missing values before cleaning:")
print(data.isnull().sum())

# Fill empty ages with the median age
data["age"] = data["age"].fillna(data["age"].median())
# Fill empty ports with "S" (most common)
data["embarked"] = data["embarked"].fillna("S")

# Turn categorical words into 0/1 columns (one-hot encoding)
data = pd.get_dummies(data, columns=["sex", "embarked"], dtype=int)

print("\nFirst 5 rows of clean dataset:")
print(data.head())
print("\nAny blanks left?", data.isnull().values.any())

# Predict / Clean a brand-new passenger
print("\n--- Cleaning a Brand-New Passenger ---")
new = pd.DataFrame([{"pclass": 2, "sex": "female", "age": 27, "fare": 30, "embarked": "C"}])
new = pd.get_dummies(new, columns=["sex", "embarked"], dtype=int)
new = new.reindex(columns=data.drop("survived", axis=1).columns, fill_value=0)
print("New passenger formatted as clean numbers:")
print(new)
