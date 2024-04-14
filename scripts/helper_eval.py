from bert_score import BERTScorer


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