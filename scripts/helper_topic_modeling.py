import numpy as np
from typing import Tuple

from string import punctuation
from nltk.corpus import stopwords
from collections import defaultdict, Counter
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, TruncatedSVD, LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.lda_model
import pyLDAvis.gensim_models

# --------------------- #
# LARGE LANGUAGE MODELS #
# --------------------- #

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


# --------------------------- #
# LATENT DIRICHLET ALLOCATION #
# --------------------------- #

def get_topics_from_lda(corpus: pd.Series, ngram_min: int=2, ngram_max: int=2):

    count_vectorizer = CountVectorizer(stop_words='english', min_df=5, max_df=0.7, 
                                       ngram_range=(ngram_min, ngram_max))
    count_vectors = count_vectorizer.fit_transform(corpus)

    lda_model = LatentDirichletAllocation(n_components=5, random_state=314)
    W_lda_matrix = lda_model.fit_transform(count_vectors)
    H_lda_matrix = lda_model.components_

    display_topics(lda_model, count_vectorizer.get_feature_names_out())

    return lda_model


def display_topics(model, features, no_top_words=5):
    for topic, words in enumerate(model.components_):
        total = words.sum()
        largest = words.argsort()[::-1] # invert sort order
        print("\nTopic %02d" % topic)
        for i in range(0, no_top_words):
            print("  %s (%2.2f)" % (features[largest[i]], abs(words[largest[i]]*100.0/total)))