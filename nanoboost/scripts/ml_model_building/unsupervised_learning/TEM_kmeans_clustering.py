from skimage.measure import regionprops
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

# load labeled_image from previous processing steps
labeled_image = np.load('labeled_image.npy')

# Calculate properties of each labeled object
props = regionprops(labeled_image)

# Define minimum width and height criteria
min_width = 1
min_height = 1

# Filter objects based on width and height (using minor and major axis lengths)
filtered_props = [prop for prop in props if prop.minor_axis_length >= min_width and prop.major_axis_length >= min_height]

features_filtered = np.array([[prop.major_axis_length / prop.minor_axis_length,
                               prop.major_axis_length,
                               prop.minor_axis_length,
                               prop.eccentricity,
                               prop.solidity,
                               prop.extent] for prop in filtered_props])


scaler = MinMaxScaler()
features_filtered = scaler.fit_transform(features_filtered)

num_clusters = 5

kmeans = KMeans(n_clusters=num_clusters, random_state=42, init= "k-means++").fit(features_filtered)
cluster_labels = kmeans.labels_

# Initialize cluster_image_filtered as a 3-channel (RGB) image
cluster_image_filtered = np.zeros((*labeled_image.shape, 3), dtype=np.uint8)

# Define a colormap for clustering
colormap = plt.cm.nipy_spectral

# Calculate the number of objects in each cluster
cluster_counts = np.bincount(cluster_labels, minlength=num_clusters)

# Calculate the percentage of objects in each cluster
total_objects = len(cluster_labels)
cluster_percentages = (cluster_counts / total_objects) * 100

# Apply colors
for prop, cluster_label in zip(filtered_props, cluster_labels):
    color_index = 0.1 + (cluster_label + 1) / (num_clusters + 1)  # Avoid very dark colors
    rgb_color = (np.array(colormap(color_index)[:3]) * 255).astype(np.uint8)
    cluster_image_filtered[labeled_image == prop.label] = rgb_color

# Visualization
plt.figure(figsize=(8, 8))
plt.imshow(cluster_image_filtered)

plt.axis('off')

legend_handles = [
    mpatches.Patch(color=colormap(0.1 + (i + 1) / (num_clusters + 1)), label=f'Cluster {i} - {cluster_percentages[i]:.2f}$\%$')
    for i in range(num_clusters)
]
plt.legend(handles=legend_handles, bbox_to_anchor=(1, 1), loc='upper left', fontsize=12)

plt.tight_layout()

plt.show()