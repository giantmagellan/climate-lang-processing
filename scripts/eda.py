import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer


# ------------- #
# DISTRIBUTIONS #
# ------------- #

def get_top_n_ngram(corpus: pd.Series, n_phrases: int=10, ngram_min: int=2, ngram_max: int=2, stopwords: bool=True) -> sns.barplot:
    """
    Finds most common n-gram phrases in a given corpus and plot its distribution.
    :param corpus: pd.Series, text column
    :param n_phrases: int, number of n-gram phrases to return
    :param ngram_min: int, lower n-gram range boundary
    :param ngram_max: int, upper n-gram range boundary
    :param stopwords: bool, toggle stop word removal
    :return: pd.DataFrame of n-grams and their respective counts
    """
    # Check if n-grams should include stop words
    if stopwords:
        stopwords = 'english'
    else:
        stopwords = None

    try: 
        vec = CountVectorizer(ngram_range=(ngram_min, ngram_max), stop_words=stopwords).fit(corpus)

        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

        # Store word frequencies as pd.DataFrame for plotting
        df = pd.DataFrame(words_freq[:n_phrases], columns=['snippets', 'count'])

        # Remove climate bi-gram phrases
        climate_grams = ["climate change", "global warming", "climate crisis", "greenhouse gas", "greenhouse gases", "carbon tax"]
        df = df[~df['snippets'].isin(climate_grams)]

    except ValueError:
        print("Invalid input")
    
    return plot_ngram_dist(df)


def plot_ngram_dist(df: pd.DataFrame) -> None:
    """ 
    Plot the n-gram distribution.
    :param df: pd.DataFrame, dataframe of climate news snippets.
    :return: sns.histplot
    """
    sns.set_style('darkgrid')
    sns.set_theme(rc={'figure.figsize':(10,4)})

    # Plot bar chart of n-gram frequencies
    sns.barplot(df, x='snippets', y='count', legend=False)
    plt.xticks(rotation=45)
    plt.title("Phrase Frequencies")
    plt.xlabel("N-Grams")
    plt.ylabel("Count")

    # Save and show plot
    plt.savefig('figures/n_gram_frequencies.png')
    plt.show()
