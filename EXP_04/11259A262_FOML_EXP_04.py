# Experiment 4: Predict Yes / No (KNN & SVM)
# Aim: Predict whether a customer purchases an advertised product based on
# Age and Estimated Salary using KNN and Support Vector Machine (SVM),
# compare their accuracy scores, and predict for a new potential customer.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Load dataset
data = pd.read_csv("Social_Network_Ads.csv")

# Input features (Age, Salary) and target (Purchased: 1 or 0)
features = ["Age", "EstimatedSalary"]
X = data[features]
y = data["Purchased"]

# Split into 75% training and 25% testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1)

# Feature scaling using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_acc = knn.score(X_test, y_test)
print(f"KNN accuracy: {round(knn_acc, 2)}")

# Train SVM classifier
svm = SVC()
svm.fit(X_train, y_train)
svm_acc = svm.score(X_test, y_test)
print(f"SVM accuracy: {round(svm_acc, 2)}")

# Predict for a new visitor (Age: 40, Salary: $90,000)
new_person = scaler.transform(pd.DataFrame([[40, 90000]], columns=features))
answer = knn.predict(new_person)[0]
decision = "Yes (Purchased)" if answer == 1 else "No (Not Purchased)"
print(f"Prediction for new visitor [Age 40, Salary 90k]: {answer} -> {decision}")
