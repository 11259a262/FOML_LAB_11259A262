# Experiment 2: Train & Test Fairly
# Aim: Split the Iris dataset into training (80%) and test (20%) sets,
# train a K-Nearest Neighbors (KNN) model, evaluate its accuracy,
# and use it to predict the species of a brand-new flower.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the Iris flower measurements and species labels
X, y = load_iris(return_X_y=True)

# Split into 80% training data and 20% unseen test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# Initialize KNN with 5 neighbors and train on the training set
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Evaluate model accuracy on the hidden test set
accuracy = model.score(X_test, y_test)
print(f"Accuracy on unseen test set: {accuracy:.4f}")

# Predict on brand-new flower measurements
species = ["setosa", "versicolor", "virginica"]
new_flower = [[5.1, 3.5, 1.4, 0.2]]
pred_idx = model.predict(new_flower)[0]
print(f"Input measurements {new_flower[0]} -> Predicted species: {species[pred_idx]}")
