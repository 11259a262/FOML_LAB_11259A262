# Experiment 7: A Forest of Trees (Random Forest)
# Aim: Compare a single Decision Tree against a Random Forest classifier on the
# Breast Cancer dataset, diagnose a new tumour sample,
# and visualize the Top 5 most important features identified by the forest.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load Breast Cancer Wisconsin dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=1)

# Train single Decision Tree classifier
tree = DecisionTreeClassifier(random_state=1)
tree.fit(X_train, y_train)
tree_acc = tree.score(X_test, y_test)

# Train Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators=100, random_state=1)
forest.fit(X_train, y_train)
forest_acc = forest.score(X_test, y_test)

print(f"Single Decision Tree accuracy : {round(tree_acc, 3)}")
print(f"Random Forest ensemble accuracy: {round(forest_acc, 3)}")

# Diagnose new sample from test set
names = data.target_names
new_tumour = [X_test[0]]
result = forest.predict(new_tumour)[0]
print(f"\nDiagnosis prediction for new tumour sample: {names[result].upper()}")

# Plot Top 5 Feature Importances
importances = forest.feature_importances_
feat_names = data.feature_names
top_features = sorted(zip(importances, feat_names), reverse=True)[:5]
vals = [t[0] for t in top_features]
labs = [t[1] for t in top_features]

plt.figure(figsize=(8, 4.5))
bars = plt.barh(labs[::-1], vals[::-1], color="#2F49D1", edgecolor="black", height=0.6)
plt.xlabel("Gini Feature Importance", fontsize=11)
plt.title("Random Forest: Top 5 Predictive Features", fontsize=12, fontweight="bold")
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("Exp7_2.jpeg", dpi=200)
plt.close()
print("Feature importance chart saved to Exp7_2.jpeg successfully.")
