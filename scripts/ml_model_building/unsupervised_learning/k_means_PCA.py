from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# load features_list_NR from previous processing steps
features_list_NR = np.load('features_list_NR.npy')

# scale data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(features_list_NR)

# PCA with 2 PCs
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# k-means
n_clusters = 5 

kmeans = KMeans(n_clusters=n_clusters, random_state=42, init = 'k-means++', max_iter=10000)
kmeans.fit(X_pca)
centroids = kmeans.cluster_centers_
labels_kmeans = kmeans.labels_