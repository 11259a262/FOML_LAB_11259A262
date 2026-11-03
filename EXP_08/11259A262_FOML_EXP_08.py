# Experiment 8: Boosting (Learning from Mistakes)
# Aim: Train a sequential Gradient Boosting classifier on medical heart disease data,
# evaluate accuracy on hidden test patients,
# and screen a new patient to predict potential heart disease risk.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Load heart disease patient records
data = pd.read_csv("heart.csv")

# Input features (clinical measurements) and target (1: Disease, 0: Normal)
X = data.drop("target", axis=1)
y = data["target"]

# Train/Test split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# Fit Gradient Boosting model
model = GradientBoostingClassifier(random_state=1)
model.fit(X_train, y_train)

# Evaluate classification accuracy
acc = model.score(X_test, y_test)
print(f"Gradient Boosting accuracy on test set: {round(acc, 3)}")

# Screen a new patient from the test set
new_patient = X_test.iloc[[0]]
result = model.predict(new_patient)[0]
diagnosis = "Positive (Heart Disease Risk Detected)" if result == 1 else "Negative (Normal Heart Indicators)"
print(f"\nScreening result for new patient: {result} -> {diagnosis}")
