from PIL import Image
from skimage.measure import label
from skimage import io, filters, measure, morphology, color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Step 1: Convert the image to grayscale
TEM_image_path = 'nanorod_TEM.png'
TEM_image = Image.open(TEM_image_path)
TEM_image_gray = TEM_image.convert('L')  # 'L' mode is for grayscale
# Convert to numpy array for skimage processing
tem_image = np.array(TEM_image_gray)

# Step 2: Threshold the image to identify the rods (black)
# use a threshold to create a binary image (array of true/false or 0/1)
threshold = filters.threshold_otsu(tem_image)
binary_image = tem_image < threshold  # < as rods are black (0), background white (2^16 for this 16 bit image)

# Step 3: Remove small objects and separate touching objects
# Perform opening to remove small objects/noise
opened_image = morphology.binary_opening(binary_image, morphology.disk(1))

# Step 4: Label the image to identify individual objects (rods)
labeled_image, num_labels = measure.label(opened_image, return_num=True, connectivity=2)
print(f"Identified {num_labels} objects.")

# Initialize an RGB image with the same dimensions as labeled_image, filled with black (background)
rgb_image = np.zeros((*labeled_image.shape, 3), dtype=np.uint8)

# Where labeled_image is not 0 (background), set the pixels to red [255, 0, 0]
rgb_image[labeled_image != 0] = [255, 0, 0]

# Visualization
fig, ax = plt.subplots(figsize=(8, 8))

ax.imshow(rgb_image)
ax.axis('off')

# Create a legend
red_patch = mpatches.Patch(color='red', label='Identified \n Objects')
legend = ax.legend(handles=[red_patch], loc='upper left', fontsize=16)
legend.get_frame().set_alpha(1)
plt.tight_layout()
plt.show()




### K-means clustering on the identified objects

from skimage.measure import regionprops
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler

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

