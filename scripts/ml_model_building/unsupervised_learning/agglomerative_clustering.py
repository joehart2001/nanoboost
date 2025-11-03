from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from nanoboost.scripts.utils.utils import unpickle

# load df_all_wavelets_02_DNA from previous processing steps
df_all_wavelets_02_DNA = unpickle("df_all_wavelets_02_DNA.pkl")

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(df_all_wavelets_02_DNA["bior2.2"][0])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# agglomerative clustering
agglomerative = AgglomerativeClustering(n_clusters=3)
labels_agglo = agglomerative.fit_predict(X_pca)

centroids_agglo = np.array([X_pca[labels_agglo == i].mean(axis=0) for i in range(agglomerative.n_clusters)])