import pandas as pd
import matplotlib.pyplot as plt

from string import punctuation
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud



# ------------- #
# PREPROCESSING #
# ------------- #

punctuation = set(punctuation) # speeds up comparison
sw = stopwords.words('english')

def change_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    # Convert all column names to lowercase
    df.columns = df.columns.str.lower()
    # data preprocessing
    df['tokens'] = df['snippet'].str.lower()
    df['tokens'] = df['tokens'].apply(remove_punctuation)
    df['tokens'] = tokenize(df['tokens'].str)
    df['tokens'] = df['tokens'].apply(remove_stopwords)

    df['tokens_no_climate'] = df['snippet_no_climate'].str.lower()
    df['tokens_no_climate'] = df['tokens_no_climate'].apply(remove_punctuation)
    df['tokens_no_climate'] = tokenize(df['tokens_no_climate'].str)
    df['tokens_no_climate'] = df['tokens_no_climate'].apply(remove_stopwords)

    # reorder columns
    df = df[['matchdatetime', 'station', 'snippet', 'tokens', 'snippet_no_climate', 'tokens_no_climate']]
    
    return df


def remove_punctuation(text) :
    return "".join(ch for ch in text if ch not in punctuation)


def remove_stopwords(tokens) :
    return [token for token in tokens if token not in sw]


def tokenize(text) :
    tokens = text.split()
    return(tokens)


def remove_bigram_phrases(text, phrases_to_remove):
     for phrase in phrases_to_remove:
         text = text.replace(phrase, '')
     return text


# ---------------------- #
# DESCRIPTIVE STATISTICS #
# ---------------------- #

def descriptive_stats(tokens, verbose=True) :
    num_tokens=len(tokens)
    num_unique_tokens = len(set(tokens))
    lexical_diversity = num_unique_tokens/num_tokens
    num_characters = sum(len(token) for token in tokens)

    if verbose :
        print(f"There are {num_tokens} tokens in the data.")
        print(f"There are {num_unique_tokens} unique tokens in the data.")
        print(f"There are {num_characters} characters in the data.")
        print(f"The lexical diversity is {lexical_diversity:.3f} in the data.")

        # print the five most common tokens
        counter = Counter(tokens)
        top_10_tokens = counter.most_common(10)
        print("\nTop 10 most common tokens:")
        for token, count in top_10_tokens:
            print(f"{token}: {count} occurrences")

    return([num_tokens, num_unique_tokens,
            lexical_diversity,
            num_characters])

def count_words(df, column, preprocess=None, min_freq=2):

    # process tokens and update counter
    def update(doc):
        tokens = doc if preprocess is None else preprocess(doc)
        counter.update(tokens)

    # create counter and run through all data
    counter = Counter()
    df[column].map(update)

    # transform counter into data frame
    freq_df = pd.DataFrame.from_dict(counter, orient='index', columns=['freq'])
    freq_df = freq_df.query('freq >= @min_freq')
    freq_df.index.name = 'token'

    return freq_df.sort_values('freq', ascending=False)

def wordcloud(word_freq, title=None, max_words=200, stopwords=sw):

    wc = WordCloud(width=800, height=400,
                   background_color= "black", colormap="Paired",
                   max_font_size=150, max_words=max_words)

    # convert data frame into dict
    if type(word_freq) == pd.Series:
        counter = Counter(word_freq.fillna(0).to_dict())
    else:
        counter = word_freq

    # filter stop words in frequency counter
    if stopwords is not None:
        counter = {token:freq for (token, freq) in counter.items()
                              if token not in stopwords}
    wc.generate_from_frequencies(counter)

    plt.title(title)

    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")