# Experiment 5: Find Groups (K-Means Clustering)
# Aim: Segment mall customers into 5 natural clusters based on Annual Income
# and Spending Score using K-Means, predict the cluster for a new customer,
# and visualize the customer groups along with cluster centroids.

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load mall customer dataset
data = pd.read_csv("Mall_Customers.csv")

# Input features: Annual Income and Spending Score
features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = data[features]

# Apply K-Means with 5 clusters
model = KMeans(n_clusters=5, n_init=10, random_state=1)
model.fit(X)
data["Group"] = model.labels_

# Display customer counts per cluster
print("Customer counts in each cluster:")
print(data["Group"].value_counts().sort_index())

# Predict cluster for a brand-new customer ($75k income, 85 spending score)
new_customer = pd.DataFrame([[75, 85]], columns=features)
group = model.predict(new_customer)[0]
print(f"\nNew customer with Income $75k and Spending Score 85 -> Assigned to Cluster: {group}")

# Plot customer clusters with centroids
plt.figure(figsize=(8, 5.5))
x_val = data["Annual Income (k$)"]
y_val = data["Spending Score (1-100)"]
scatter = plt.scatter(x_val, y_val, c=model.labels_, cmap="tab10", s=45, alpha=0.85, edgecolors="none")
cen = model.cluster_centers_
plt.scatter(cen[:, 0], cen[:, 1], marker="X", s=220, c="black", linewidths=2, label="Centroids")
plt.scatter([75], [85], marker="*", s=260, c="#e06c75", edgecolors="black", linewidths=1.5, label=f"New Customer (Cluster {group})")

plt.xlabel("Annual Income (k$)", fontsize=11)
plt.ylabel("Spending Score (1-100)", fontsize=11)
plt.title("K-Means Clustering: 5 Customer Segments", fontsize=12, fontweight="bold")
plt.legend(frameon=True, loc="upper right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("Exp5_2.jpeg", dpi=200)
plt.close()
print("\nCluster visualization saved to Exp5_2.jpeg successfully.")
