# K-Means Clustering on MNIST Dataset Using Cosine Similarity

This project performs **K-means clustering** with **cosine similarity** on the MNIST dataset of handwritten digits. The goal is to group similar digit images into clusters, showcasing how K-means can segment images based on pixel patterns.

## Overview

K-means clustering is an unsupervised learning method used to partition data points into `k` clusters. In this project, cosine similarity is used as the distance metric instead of the typical Euclidean distance. This approach often works better for high-dimensional data, such as images, where the orientation or direction of data points is more meaningful than their magnitude.

## Dataset

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is used, containing 28x28 pixel grayscale images of handwritten digits (0-9). Each image has:
- **784 pixel values** (28x28).
- **Label** (digit from 0 to 9) to indicate the actual digit (only used for evaluation, not for clustering).

## Code Structure

- **Data Loading and Preprocessing**: Loads the MNIST dataset, checks for missing values, and converts data to NumPy arrays.
- **Cosine Similarity Calculation**: Custom function to calculate cosine similarity, used for assigning data points to clusters.
- **K-means Algorithm**: Implements K-means clustering with functions for cluster assignment, centroid update, and convergence checking.
- **Visualization**: Displays sample images from each cluster after clustering with different values of `k`.

## Dependencies

The following libraries are required to run the code:
- `numpy` for numerical operations
- `pandas` for data handling
- `matplotlib` and `seaborn` for visualizations

Install dependencies using:
```bash
pip install numpy pandas matplotlib seaborn
