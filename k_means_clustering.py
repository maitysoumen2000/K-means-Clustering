import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("/content/drive/MyDrive/mnist_train.csv")
#df2 = pd.read_csv("/content/drive/MyDrive/mnist_test.csv")

df1.head(5)

df1.isnull().any()

df1.columns

df1.info()

df1.describe()

df1.isnull().sum()

[features for features in df1.columns if df1[features].isnull().sum()>0]

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
sns.heatmap(df1.isnull(),yticklabels=False, cbar=False,cmap = 'viridis')

data = df1.to_numpy()

x_train = data[:,1:]
y_train = data[:,:1]
print(x_train.shape)
print (y_train.shape)

plt.figure(figsize = (7,7))
idx = 150
grid_data = x_train[idx].reshape(28,28)
plt.imshow(grid_data,interpolation = "none",cmap = "gray")
plt.show()
print(y_train[idx])

"""#To calculate distance between centroids and data points using cosine similarity"""

def cosine_similarity(vector1, vector2):
  dot_product = np.dot(vector1, vector2)
  norm_vector1 = np.linalg.norm(vector1)
  norm_vector2 = np.linalg.norm(vector2)
  similarity = dot_product / (norm_vector1 * norm_vector2)
  return similarity

"""#assign data points to a cluster based on nearby centroid"""

def assign_clusters(data, centroids):
    clusters = {}
    for i, point in enumerate(data):
        closest_centroid_idx = np.argmax([cosine_similarity(point, centroid) for centroid in centroids])
        if closest_centroid_idx in clusters:
            clusters[closest_centroid_idx].append(i)
        else:
            clusters[closest_centroid_idx] = [i]
    return clusters

"""#updating centroids after each iteration"""

def update_centroids(data, clusters):
    centroids = []
    for cluster_points in clusters.values():
        cluster_mean = np.mean([data[i] for i in cluster_points], axis=0)
        centroids.append(cluster_mean)
    return centroids

def has_converged(old_centroids, new_centroids, tol=1e-4):
    return all(cosine_similarity(old, new) >= 1 - tol for old, new in zip(old_centroids, new_centroids))

def kmeans(data, k, max_iters):
    initial_centroids = np.random.choice(len(data), k, replace=False)
    centroids = data[initial_centroids]

    for _ in range(max_iters):
        clusters = assign_clusters(data, centroids)
        old_centroids = centroids

        centroids = update_centroids(data, clusters)

        if has_converged(old_centroids, centroids):
            break
    return clusters, centroids

"""#Visualization in Kmeans Clustering with 10 clusters"""

clusters, centroids = kmeans(data, 10, 1000)
print("\n******assigned clusters******\n",centroids)
print("\n******list of Cluster******\n", clusters)

def showImage(index):
  first_image = x_train[index, :]
  pixels = first_image.reshape((28, 28))
  plt.imshow(pixels, cmap='gray')

K = 10
for i in range(len(clusters)):
    plt.figure(figsize=(16, 16))
    print("Showing Cluster ", i+1,":")
    for index in range(K):
        first_image = x_train[clusters[i][index], :]

        axs = plt.subplot(4, K, index + 1)
        plt.imshow(first_image.reshape(28, 28))
        plt.gray()
    plt.show()

"""#Visualization in Kmeans Clustering with 7 clusters"""

clusters, centroids = kmeans(data, 7, 1000)
print("\n******assigned clusters******\n",centroids)
print("\n******list of Cluster******\n", clusters)

K = 7
for i in range(len(clusters)):
    plt.figure(figsize=(16, 16))
    print("Showing Cluster ", i+1,":")
    for index in range(K):
        first_image = x_train[clusters[i][index], :]

        axs = plt.subplot(4, K, index + 1)
        plt.imshow(first_image.reshape(28, 28))
        plt.gray()
    plt.show()

"""# Visualization in Kmeans Clustering with 4 clusters"""

clusters, centroids = kmeans(data, 4, 1000)
print("\n******assigned clusters******\n",centroids)
print("\n******list of Cluster******\n", clusters)

K = 4
for i in range(len(clusters)):
    plt.figure(figsize=(16, 16))
    print("Showing Cluster ", i+1,":")
    for index in range(K):
        first_image = x_train[clusters[i][index], :]

        axs = plt.subplot(4, K, index + 1)
        plt.imshow(first_image.reshape(28, 28))
        plt.gray()
    plt.show()