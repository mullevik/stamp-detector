"""
This file contains functions required for the clustering of information
blobs based on pixel locations and HUE
"""
from typing import List, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans


def cluster_image(image: np.ndarray,
                  bin_image: np.ndarray,
                  n_clusters: int = 3,
                  color_weight: float = 1.) -> Tuple[List[np.ndarray], float]:
    """
    Clusters foreground pixels based on their position and HUE.
    This functions uses sklearn.cluster.KMeans:
    (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
    :param image: bgr_image
    :param bin_image: binary image (0 - foreground, 255 - background)
    :param n_clusters: number of clusters
    :param color_weight: how important is the HUE for clustering
    :return: list of binary images (one for each cluster) (0 - pixel does not
    belong to this cluster, 255 - pixel belongs to this cluster) and a cost
    of this clustering (sum of distances of pixels from cluster centroids)
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    pixel_locations = np.argwhere(bin_image == 0)

    # extract hue from HSV image at pixel locations
    pixel_colors = hsv_image[pixel_locations[:, 0],
                   pixel_locations[:, 1], :1]

    # observations of shape (n, 3)
    observations = np.hstack((pixel_locations, pixel_colors)).astype \
        (np.float64)
    # normalize
    observations[:, 0] = observations[:, 0] / bin_image.shape[0]
    observations[:, 1] = observations[:, 1] / bin_image.shape[1]
    observations[:, 2] = (observations[:, 2] / 255) * color_weight

    kmeans = KMeans(n_clusters=n_clusters,
                    init="k-means++",
                    random_state=42)

    kmeans_output = kmeans.fit(observations)

    cluster_images = []
    total_cost = 0.

    # reconstruct cluster images
    for i in range(n_clusters):

        # create binary image of i-th cluster
        binary_cluster_image = np.full_like(bin_image, fill_value=0)
        cluster_mask = kmeans_output.labels_ == i
        cluster_observations = observations[cluster_mask]
        locations = pixel_locations[cluster_mask]
        binary_cluster_image[locations[:, 0], locations[:, 1]] = 255
        cluster_images.append(binary_cluster_image)

        # calculate within cluster cost
        cluster_center = kmeans.cluster_centers_[i]
        costs = np.linalg.norm(cluster_observations - cluster_center, axis=1)
        total_cost += np.sum(costs)

    return cluster_images, total_cost
