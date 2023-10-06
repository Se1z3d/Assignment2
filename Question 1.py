import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from sklearn.cluster import DBSCAN

from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

# Fetch the Olivetti faces dataset
data = fetch_olivetti_faces(shuffle=True, random_state=42)

# The dataset contains images and corresponding target labels
images = data.images
target = data.target

# Print some information about the dataset
print(f"Number of images: {images.shape[0]}")
print(f"Image shape: {images.shape[1:]}")

# The target labels are integers corresponding to the identity of the person in each image
# Each person is represented by 10 different images, so there are 40 different individuals in the dataset
print(f"Number of unique individuals: {len(set(target))}")

# Define the split ratios
train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15

# Split the dataset into training and temp sets (70% training, 30% temp)
images_train, images_temp, target_train, target_temp = train_test_split(
    images, target, test_size=(1 - train_ratio), stratify=target, random_state=42
)

# Split the temp set into validation and test sets (50% validation, 50% test)
images_val, images_test, target_val, target_test = train_test_split(
    images_temp, target_temp, test_size=0.5, stratify=target_temp, random_state=42
)

# Print the number of samples in each set
print(f"Number of training samples: {len(images_train)}")
print(f"Number of validation samples: {len(images_val)}")
print(f"Number of test samples: {len(images_test)}")

# Create a Support Vector Machine (SVM) classifier
classifier = SVC(kernel='linear', random_state=42)

# Perform k-fold cross-validation on the training set
# Here, we'll use 5-fold cross-validation, but you can adjust the number of folds (cv) as needed
k = 5
cross_val_scores = cross_val_score(classifier, images_train.reshape(len(images_train), -1), target_train, cv=k)

# Print cross-validation scores
print("Cross-Validation Scores:", cross_val_scores)
print(f"Mean Cross-Validation Accuracy: {cross_val_scores.mean()}")

# Train the classifier on the entire training set
classifier.fit(images_train.reshape(len(images_train), -1), target_train)

# Evaluate the classifier on the validation set
validation_accuracy = classifier.score(images_val.reshape(len(images_val), -1), target_val)
print(f"Validation Accuracy: {validation_accuracy}")

# Flatten the images for K-Means clustering
images_train_flat = images_train.reshape(len(images_train), -1)

# Reduce dimensionality using PCA (optional but can help K-Means)
n_components = 100  # You can adjust this number based on your preference
pca = PCA(n_components=n_components, random_state=42)
images_train_pca = pca.fit_transform(images_train_flat)

# Initialize a list to store silhouette scores for different numbers of clusters
silhouette_scores = []

# Try a range of cluster numbers (e.g., from 2 to 10)
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(images_train_pca)
    silhouette_avg = silhouette_score(images_train_pca, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Find the optimal number of clusters with the highest silhouette score
optimal_n_clusters = np.argmax(silhouette_scores) + 2  # Add 2 to account for the starting cluster number (2)

print(f"Optimal number of clusters: {optimal_n_clusters}")

# Train K-Means with the optimal number of clusters
final_kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
final_cluster_labels = final_kmeans.fit_predict(images_train_pca)

# You can access the cluster centroids if needed
cluster_centers = final_kmeans.cluster_centers_

# Create a Support Vector Machine (SVM) classifier
classifier = SVC(kernel='linear', random_state=42)

# Train the classifier on the reduced-dimension dataset with cluster labels
classifier.fit(final_cluster_labels.reshape(-1, 1), target_train)

# Predict cluster labels for the validation set
images_val_flat = images_val.reshape(len(images_val), -1)
images_val_pca = pca.transform(images_val_flat)  # Apply the same PCA transformation as before
val_cluster_labels = final_kmeans.predict(images_val_pca)

# Predict identities using the trained classifier
val_predictions = classifier.predict(val_cluster_labels.reshape(-1, 1))

# Calculate the accuracy on the validation set
validation_accuracy = accuracy_score(target_val, val_predictions)
print(f"Validation Accuracy: {validation_accuracy}")

# Flatten the images and reduce dimensionality using PCA
images_train_flat = images_train.reshape(len(images_train), -1)
n_components = 100  # You can adjust this number based on your preference
pca = PCA(n_components=n_components, random_state=42)
images_train_pca = pca.fit_transform(images_train_flat)

# Normalize the PCA-transformed data to have unit length (cosine similarity)
images_train_normalized = normalize(images_train_pca)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.2, min_samples=5, metric='cosine')  # You can adjust epsilon (eps) and min_samples
cluster_labels = dbscan.fit_predict(images_train_normalized)

# Calculate the silhouette score to evaluate the clustering
silhouette_avg = silhouette_score(images_train_normalized, cluster_labels)

print(f"Silhouette Score: {silhouette_avg}")

# Identify the number of clusters (excluding noise points)
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
print(f"Number of Clusters: {n_clusters}")

# Create a scatter plot to visualize the clustering
plt.scatter(images_train_normalized[:, 0], images_train_normalized[:, 1], c=cluster_labels, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
