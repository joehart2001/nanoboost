from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score
from nanoboost.scripts.utils.utils import unpickle

# load event_data from previous processing steps
event_data = unpickle("event_data.pkl")

# e.g. for DNA k=3
model_DTW = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=6, n_jobs=-1, init='k-means++', random_state=42).fit(event_data)
labels_DTW = model_DTW.labels_

flattened_data_bior22_unscaled = unpickle("flattened_data_bior22_unscaled.pkl")

inertia_DTW= model_DTW.inertia_
silhouette_score = silhouette_score(flattened_data_bior22_unscaled, labels_DTW)
