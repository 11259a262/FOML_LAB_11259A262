# Experiment 3: Predict a Number (Linear Regression)
# Aim: Predict car fuel efficiency (mpg) from engine and weight attributes,
# evaluate the R2 score, predict mileage for a new car,
# and visualize the learned trend line on a scatter plot.

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Load Auto MPG dataset and remove rows with missing values
data = sns.load_dataset("mpg").dropna()

# Features and target variable
features = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
X = data[features]
y = data["mpg"]

# Train/Test split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# Fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Calculate and display R2 score
r2 = model.score(X_test, y_test)
print(f"R2 score on test set: {round(r2, 3)}")

# Predict mileage for a brand-new car
new_car = pd.DataFrame([[4, 120, 90, 2500, 16, 80]], columns=features)
mpg_pred = model.predict(new_car)[0]
print(f"Predicted mileage for new car: {round(mpg_pred, 1)} mpg")

# Generate and save scatter plot with regression trend line
w = data["weight"]
m = data["mpg"]
plt.figure(figsize=(8, 5))
plt.scatter(w, m, alpha=0.5, color="#2F49D1", label="Cars data")
a, b = np.polyfit(w, m, 1)
plt.plot(w, a * w + b, color="red", linewidth=2, label="Trend line (Learned rule)")
plt.scatter([2500], [mpg_pred], color="#0B7A4D", s=180, marker="*", label=f"New car prediction ({round(mpg_pred, 1)} mpg)")
plt.xlabel("Weight (lbs)", fontsize=11)
plt.ylabel("Miles Per Gallon (mpg)", fontsize=11)
plt.title("Linear Regression: Car Weight vs Fuel Efficiency", fontsize=12, fontweight="bold")
plt.legend(frameon=True)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("Exp3_2.jpeg", dpi=200)
plt.close()
print("Graph saved to Exp3_2.jpeg successfully.")
