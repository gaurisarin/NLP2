import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('vader_lexicon')

# Loading the csv into the file.
movies_df = pd.read_csv('movies.csv')

movies_df['genre'] = movies_df['genre'].str.lower()
movies_df['title_lower'] = movies_df['title'].str.lower()
movies_df['user_rating'] = movies_df['metascore']


def returnGenres(genre):
    lower_cased_genre = genre.lower()
    filtered_df = movies_df[movies_df['genre'].str.contains(lower_cased_genre)]
    if len(filtered_df) == 0:
        return "Genre not found in Dataset. Please try again!"
    sorted_df = filtered_df.sort_values(by='user_rating', ascending=False)
    return list(sorted_df['title'][:10])


def returnBasedOnMovie(movie, column_to_return):
    movie_lower = movie.lower()
    filtered_df = movies_df[movies_df['title_lower'] == movie_lower]
    output = list(filtered_df[column_to_return])
    if len(output) == 0:
        return "Movie not found in the dataset. Please try again!"
    return output[0]


def returnDirector(movie):
    return returnBasedOnMovie(movie, 'director')


def returnLeadActor(movie):
    return returnBasedOnMovie(movie, 'cast1')


def returnYear(movie):
    return returnBasedOnMovie(movie, 'year')


def analyze_sentiment(sentence):
    # Tokenization
    tokens = word_tokenize(sentence)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Reconstruct the preprocessed sentence
    preprocessed_sentence = ' '.join(tokens)

    # Sentiment analysis
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(preprocessed_sentence)

    return sentiment_scores['compound']


def change_sentiment(sentiment_score, movie=None, genre=None):
    sentiment_penalty = sentiment_score * 10

    if movie is not None:
        mask = movies_df['title'] == movie
    elif genre is not None:
        genre_lower = genre.lower()
        mask = movies_df['genre'].str.contains(genre_lower)
    else:
        raise ValueError("Either movie or genre must be provided")

    sentiment_penalty = round(sentiment_penalty)

    movies_df.loc[mask, 'user_rating'] += sentiment_penalty
