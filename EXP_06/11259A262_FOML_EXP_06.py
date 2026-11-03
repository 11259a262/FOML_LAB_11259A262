# Experiment 6: Groups as a Tree (Hierarchical Clustering)
# Aim: Perform agglomerative hierarchical clustering on mall customer data,
# display the mean attributes of the resulting 5 clusters,
# place a new customer based on closest centroid distance,
# and plot a dendrogram visualization illustrating the cluster hierarchy.

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

# Input features: Annual Income and Spending Score
features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = data[features]

# Fit agglomerative clustering model with 5 clusters
model = AgglomerativeClustering(n_clusters=5)
data["Group"] = model.fit_predict(X)

# Calculate and display the mean feature values of each cluster
print("Average feature values for each cluster:")
averages = data.groupby("Group")[features].mean()
print(averages)

# Predict / assign a new customer (Income: $80k, Spending: 15)
new_customer = [80, 15]
distances = ((averages - new_customer) ** 2).sum(axis=1)
closest_cluster = distances.idxmin()
print(f"\nNew customer [Income $80k, Spending 15] -> Nearest Cluster: {closest_cluster}")

# Plot Dendrogram of customer hierarchy
plt.figure(figsize=(9, 5))
linked = linkage(X, method="ward")
dendrogram(linked, truncate_mode="lastp", p=30, leaf_rotation=45, leaf_font_size=9)
plt.title("Hierarchical Clustering Dendrogram (Ward Linkage)", fontsize=12, fontweight="bold")
plt.xlabel("Cluster Size / Sample Index", fontsize=11)
plt.ylabel("Euclidean Distance", fontsize=11)
plt.axhline(y=150, color="r", linestyle="--", linewidth=1.5, label="Cut-off Line for 5 Clusters")
plt.legend(frameon=True, loc="upper right")
plt.tight_layout()
plt.savefig("Exp6_2.jpeg", dpi=200)
plt.close()
print("\nDendrogram visualization saved to Exp6_2.jpeg successfully.")
