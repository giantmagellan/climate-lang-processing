import os
import pandas as pd
import numpy as np
from typing import Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from rouge import Rouge
from bert_score import BERTScorer

from prompts import B_SYS, E_SYS, B_INST, E_INST 
from prompts import DEFAULT_SYSTEM_PROMPT


def build_prompt(introduction: str, instructions: str, system_prompt: str=DEFAULT_SYSTEM_PROMPT, 
                 snippet: str=None) -> str:
    """
    Creates a prompt template by combining a default system prompt and
    a task-specific set of instructions.
    :param instruction: str, instructions for the model to perform.
    :param system_prompt: str, system prompt w/ ethical standards.
    :return transcript: str, transcript to be summarized.
    """
    try:
        system_prompt = f"{B_SYS}{system_prompt}{E_SYS}"
        prompt_template = f"{system_prompt}\n{introduction}"
        instructions = f"{B_INST}{instructions}{E_INST}"

        prompt = "".join([
            prompt_template,
            snippet, 
            instructions
        ])
        
        return prompt
    
    except ValueError:
        print("Improper inputs provided.")


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


# ------------ #
# EVAL METRICS #
# ------------ #

def bert_scorer(topic: str, topic_ref: str) -> float:
    """
    :param topic: str, topic label
    :param topic_ref: str, reference topic 
    :return: float, F1 Score
    """
    # Instantiate the BERTScorer object for English language
    scorer = BERTScorer(lang="en")

    # P1, R1, F1 represent Precision, Recall, and F1 Score respectively
    P1, R1, F1 = scorer.score([topic], [topic_ref])
    return F1.tolist()[0]


def rouge_scorer(topic: str, topic_ref: str) -> list:
    """ 
    Calculates Rouge Scores.
    :param text: str, input text
    :param ref_text: str, reference text
    :return: list
    """
    # Calculate the ROUGE scores for both topic labels using reference
    rouge = Rouge()
    eval_1_rouge = rouge.get_scores(topic, topic_ref)

    return eval_1_rouge[0]['rouge-1']['f']