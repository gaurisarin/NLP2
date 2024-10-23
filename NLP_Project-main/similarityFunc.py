import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from filterfunct import movies_df

# Gathering the relevant features into one series.
corpus = (
        movies_df['synopsis'].fillna('') + ' ' + movies_df['genre'].fillna('') + ' ' +
        movies_df['director'].fillna('') + ' ' + movies_df['cast1'].fillna('') + ' ' +
        movies_df['cast2'].fillna('') + ' ' + movies_df['cast3'].fillna('') + ' ' +
        movies_df['cast4'].fillna('')
)


def preprocess_text(text):
    """
    Preprocesses the given text by removing numbers, punctuation, stopwords,
    as well as converting everything to lowercase.
    :param text: The text to be preprocessed.
    :return: The preprocessed text.
    """
    # Remove numbers
    text = ''.join([char for char in text if not char.isdigit()])

    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]

    # Convert to lowercase
    words = [word.lower() for word in words]

    # Join the words back into a string
    processed_text = ' '.join(words)

    return processed_text


# Applying the preprocessing on the corpus series.
corpus = corpus.apply(preprocess_text)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Compute similarity matrix
similarity_matrix = cosine_similarity(X)


def get_similar_movies_index(movie_index, k=5):
    """
    Calculates and returns the k-most similar movies indices in the dataframe to the inputted movie's index.
    :param movie_index: The index of the movie in the dataframe.
    :param k: The number of most similar movie indices to return
    :return: The k-most similar movies indices in the dataframe to the inputted movie's index.
    """
    # Gets the inputted movie index's similarity scores to other movies.
    movie_scores = similarity_matrix[movie_index]

    # Gets the most similar movies by sorting in ascending order, then
    # removing the first element which would be the movie itself.
    similar_movie_indices = movie_scores.argsort()[::-1][1:k + 1]
    return similar_movie_indices


def get_movie_index_by_title(movie_title):
    """
    Helper function to get the index of a movie in the dataframe based on a given title.
    :param movie_title: The title of the movie to find the matching index.
    :return: The index of the given movie title.
    """
    try:
        # filtering the dataframe and then taking the index of matching entry.
        index = movies_df[movies_df['title_lower'] == movie_title].index[0]
        return index
    except IndexError:
        return None


def main_get_similar_movies(movie_title, k=5):
    """
    Calculates the k-most similar movies to the given movie title.
    :param movie_title: The movie to find the k-most similar movies against.
    :param k: The number of movies to return.
    :return: The title of the most similar movies.
    """
    # Gets the movie index of the given movie title argument.
    movie_index = get_movie_index_by_title(movie_title.lower())

    # Movie was not found in the Dataset.
    if movie_index is None:
        return "Movie not found in Dataset, please specify exact movie title."
    else:

        # Movie was found, run get_similar_movies_index for the given movie.
        similar_movie_indices = get_similar_movies_index(movie_index, k=k)

    # Returning just the titles from the dataframe.
    return random.choice(list(movies_df.iloc[similar_movie_indices]['title']))
