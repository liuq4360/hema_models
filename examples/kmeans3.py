from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
k_means = KMeans(n_clusters=2, random_state=0).fit(X)
print(k_means.labels_)
print(k_means.predict([[0, 0], [12, 3]]))
print(k_means.cluster_centers_)

