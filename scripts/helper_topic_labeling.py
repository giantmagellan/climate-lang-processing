import numpy as np
from typing import Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


# -------------- #
# TOPIC LABELING #
# -------------- #

def assign_categories(topic_labels: np.ndarray, num_clusters: int=5) -> dict:
    """
    Assigns each topic label to its category and prints the categorized topic labels along with category names.
    :param topic_labels: np.ndarray, array of topic labels.
    :param num_clusters: int, The desired number of categories.
    :return: dict, (categorized_topics) categories and their respective topics
    """
    # Perform clustering and find the most representative labels for each category
    categories, category_names = cluster_topic_labels(topic_labels, num_clusters)
    
    # Initialize a dictionary to store the categorized topic labels
    categorized_topics = {name: [] for name in category_names}
    
    # Assign each topic label to its category based on clustering result
    for label, category in zip(topic_labels, categories):
        # Find the category name using the category index
        category_name = category_names[category]
        # Append the topic label to the correct category in the dictionary
        categorized_topics[category_name].append(label)

    return categorized_topics


def cluster_topic_labels(topic_labels: np.ndarray, num_clusters=5) -> Tuple[np.ndarray, list]:
    """
    Cluster topic labels into categories and find the most representative label for each category.
    :param topic_labels: np.ndarray, array of topic labels.
    :param num_clusters: int, desired number of categories (constant).
    :return: tuple, (categories, category_names)
        where
        np.ndarray categories is the cluster assignment for each topic label,
        list category_names is the most representative label for each category.
    """
    # Vectorization with TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(topic_labels)

    # Clustering with K-Means
    km = KMeans(n_clusters=num_clusters, random_state=42)
    km.fit(tfidf_matrix)
    
    # Calculate the centroids of each cluster
    centroids = km.cluster_centers_

    # Calculate the cosine similarity between each item and the centroids
    similarity = cosine_similarity(tfidf_matrix, centroids)

    # For each category, find the index with the highest similarity score
    closest = similarity.argmax(axis=0)

    # Map the indexes to topic labels to get the most representative label for each category
    category_names = [topic_labels[index] for index in closest]

    return km.labels_, category_names
