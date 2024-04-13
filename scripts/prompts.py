# Beginning/End Indicators for instructions and system prompt
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# Default system prompt for upholding high ethical standards.
DEFAULT_SYSTEM_PROMPT = """\
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible,\
    while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, \
    dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering \
    something not correct. If you don't know the answer to a question, please don't share false information.
"""

# --------------- #
#  TOPIC MODELING #
# --------------- #
system_prompt_topic = """
    I need you to label the topic of a snippet from a news article about climate change. \
    Each snippet was taken because it contained at least one of the following 6 phrases: \
        climate change, global warming, climate crisis, greenhouse gas, greenhouse \
    gasses, and carbon tax.
    """

topic_introduction = """
    Introduction: \
    You will analyze a news snippet. \
    These snippets were pulled from news articles or transcribed from a news broadcast. \
    Your task is to provide a label for the snippet based on text found within the snippet. \
    In some cases, the snippet may contain multiple subjects. \
    Remember, the snippets were taken because they contained at least one of the following phrases: \
    climate change, global warming, climate crisis, greenhouse gas, greenhouse gasses, and carbon tax.\
    The topic cannot be either of these phrases but should attempt to disregard the subjects in the snippet
        that are not related to the above climate phrases.
    """

topic_instructions = """
    Instructions: \ 
    Example input snippet: \
    global action to reduce greenhouse gases. however, the us secretary of state, rex tillerson told the ministers, \
    
    The expected output is a single string like the following example:
    greenhouse gas reduction

    Create a label based on the initial topic, verb, and object of the snippet. \
    Limit the number of topic labels to 5. \
    Label each of the snippets with one of these 5 topic labels. \
    Each label shall contain no less than 2 words and no more than 3 words. \
    No label shall contain a stopword like 'and', 'its', 'an', 'the', 'for', and 'that', \
    However, if you cannot create a label, assign the value as 'Unknown', \
    Do not add any extra information. \
    Do not provide additional punctation. \ 
    Do not provide further explanation of the results.
    """

# ------------------- #
#  SENTIMENT ANALYSIS #
# ------------------- #
system_prompt_sentiment = """
    I need you to classify the sentiment of a snippet from a news article about climate change as positive or negative. \
    Each snippet was taken because it contained at least one of the following 6 phrases: \
        climate change, global warming, climate crisis, greenhouse gas, greenhouse \
    gasses, and carbon tax.
    """

sentiment_introduction = """
    Introduction: \
    You will analyze a news article snippet. \
    Your task is to classify the sentiment for the snippet based on text found within the snippet. \
    Remember, the snippets were taken because they contained at least one of the following phrases: \
        climate change, global warming, climate crisis, greenhouse gas, greenhouse gasses, and carbon tax.\
    """

sentiment_instructions = """
    Instructions: \ 
    Example input snippets: \
    global action to reduce greenhouse gases. however, the us secretary of state, rex tillerson told the ministers, // Neutral \
    carbon emissions have been reduced by 10% // Positive
    warming rate is increasing // Negative

    The expected output is a single string. \
    Classify the snippet as positive, negative, or neutral. \
    Do not add any extra information. \
    Do not provide additional punctation. \ 
    Do not provide further explanation of the results.
    """