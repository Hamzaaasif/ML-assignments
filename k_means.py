import numpy as np
from sklearn.datasets import make_blobs
import time
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
import matplotlib.pyplot as plt

# %%
# Generating data set
# ------------------------------

centers = [[1, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)
print("Total clusters : ",n_clusters)

print("Generating blob 30,000 * 2 with std deviation 0.7")
X, labels_true = make_blobs(n_samples=30000, centers=centers, cluster_std=0.7)

# %%
# Compute clustering with KMeans
# ------------------------------
print("Computing K-means")
k_means = KMeans(init="k-means++", n_clusters=3, n_init=10)
t0 = time.time()
k_means.fit(X)
t_batch = time.time() - t0


# %%
# Establishing parity between clusters
# ------------------------------------

k_means_cluster_centers = k_means.cluster_centers_
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)


# %%
# Plotting the results
# --------------------
print("Plotting results ")

fig = plt.figure(figsize=(10, 5))
fig.subplots_adjust(left=0.02, right=2.5, bottom=0.05, top=0.9)
colors = ["#4EACC5", "#FF9C34", "#4E9A06"]

# KMeans
ax = fig.add_subplot(1, 3, 1)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker=".")
    ax.plot(
        cluster_center[0],
        cluster_center[1],
        "o",
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=6,
    )
ax.set_title("KMeans")
ax.set_xticks(())
ax.set_yticks(())
plt.text(4.8, 1.5, "Training time: %.2fs\nInertia: %f" % (t_batch, k_means.inertia_))

plt.show()
