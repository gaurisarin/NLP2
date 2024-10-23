import unittest
import numpy as np
from preprocess import convert_to_bio
from class_training import queries_df
from querying_testing import predict_intent_and_entities
from filterfunct import returnGenres, returnDirector, returnYear, returnLeadActor, returnBasedOnMovie, analyze_sentiment
from similarityFunc import preprocess_text, get_movie_index_by_title, get_similar_movies_index


class testConvertToBIO(unittest.TestCase):
    def testConvertToBio(self):
        self.assertEqual(convert_to_bio(queries_df.loc[0])['BIO_tags'], ['O', 'O', 'O', 'B-MOVIE_TITLE', 'O', 'O'])
        self.assertEqual(convert_to_bio(queries_df.loc[403])['BIO_tags'], ['O', 'O', 'O', 'B-GENRE', 'O', 'O', 'O'])
        self.assertEqual(convert_to_bio(queries_df.loc[470])['BIO_tags'], ['O', 'O', 'O', 'O', 'B-GENRE', 'O', 'O', 'O'])
        self.assertEqual(convert_to_bio(queries_df.loc[42])['BIO_tags'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-MOVIE_TITLE', 'O', 'O', 'O'])
        self.assertEqual(convert_to_bio(queries_df.loc[432])['BIO_tags'], ['O', 'O', 'O', 'O', 'B-GENRE', 'O', 'O', 'O'])
        self.assertEqual(convert_to_bio(queries_df.loc[476])['BIO_tags'], ['O', 'O', 'O', 'B-GENRE', 'O', 'O', 'O', 'O', 'O'])


class testPredictIntentAndEntities(unittest.TestCase):
    def testIntentAndEntities(self):
        self.assertEqual(predict_intent_and_entities("Who directed Inception?"),
                         ('Director', [('who', 'O'), ('directed', 'O'), ('inception', 'B-MOVIE_TITLE'), ('?', 'O')]))
        self.assertEqual(predict_intent_and_entities("What's a movie like The Dark Knight?"),
                         ('Synopsis', [('what', 'O'), ("'", 'O'), ("s", 'O'), ('a', 'O'), ('movie', 'O'),
                                 ('like', 'O'), ('the', 'B-MOVIE_TITLE'), ('dark', 'I-MOVIE_TITLE'),
                                 ('knight', 'I-MOVIE_TITLE'), ('?', 'O')]))
        self.assertEqual(predict_intent_and_entities("What year did Jurassic Park release?"), ('Release Date',
                                 [('what', 'O'), ('year', 'O'), ('did', 'O'), ('jurassic', 'B-MOVIE_TITLE'),
                                  ('park', 'I-MOVIE_TITLE'), ('release', 'O'), ('?', 'O')]))
        self.assertEqual(predict_intent_and_entities("What's a good action movie"), ('Genre',
                                 [('what', 'O'), ("'", 'O'), ('s', 'O'), ('a', 'O'), ('good', 'O'),
                                 ('action', 'B-GENRE'), ('movie', 'O')]))
        self.assertEqual(predict_intent_and_entities("Give me a list of comedy movies"), ('Genre',
                                  [('give', 'O'), ('me', 'O'), ('a', 'O'), ('list', 'O'), ('of', 'O'),
                                  ('comedy', 'B-GENRE'), ('movies', 'O')]))
        self.assertEqual(predict_intent_and_entities("I liked The Matrix"), ('Feedback', [('i', 'O'),
                                  ('liked', 'O'), ('the', 'B-MOVIE_TITLE'), ('matrix', 'I-MOVIE_TITLE')]))
        self.assertEqual(predict_intent_and_entities("I hated Sharknado"),
                           ('Feedback', [('i', 'O'), ('hated', 'O'), ('sharknado', 'B-MOVIE_TITLE')]))


class testFilterFuncs(unittest.TestCase):
    def testReturnGenres(self):
        self.assertEqual(returnGenres('comedy')[:3], ["Singin' in the Rain", 'City Lights', 'Pinocchio'])
        self.assertEqual(returnGenres('action')[:3], ['The Wild Bunch', 'North by Northwest', 'Shichinin no samurai'])
        self.assertEqual(returnGenres('thriller')[:3], ['Rear Window', 'Vertigo', 'The Lady Vanishes'])
        self.assertEqual(returnGenres('drama')[:3], ['Casablanca', 'Lawrence of Arabia', 'The Godfather'])
        self.assertEqual(returnGenres('romance')[:3], ['Notorious', 'Viaggio in Italia', 'Casablanca'])
        self.assertEqual(returnGenres('sci-fi')[:3], ['Metropolis', 'Gravity', 'Bride of Frankenstein'])

    def testReturnBasedOnMovie(self):
        self.assertEqual(returnBasedOnMovie('Jurassic Park', 'director'), 'Steven Spielberg')
        self.assertEqual(returnBasedOnMovie('The Matrix', 'cast1'), 'Keanu Reeves')
        self.assertEqual(returnBasedOnMovie('The Dark Knight', 'year'), '2008')
        self.assertEqual(returnBasedOnMovie('Inception', 'director'), 'Christopher Nolan')
        self.assertEqual(returnBasedOnMovie('The Godfather', 'year'), '1972')
        self.assertEqual(returnBasedOnMovie('The Shawshank Redemption', 'cast1'), 'Tim Robbins')
        self.assertEqual(returnBasedOnMovie('Pulp Fiction', 'director'), 'Quentin Tarantino')
        self.assertEqual(returnBasedOnMovie('The Avengers', 'cast1'), 'Robert Downey Jr.')
        self.assertEqual(returnBasedOnMovie('The Lion King', 'year'), '1994')

    def testReturnDirector(self):
        self.assertEqual(returnDirector('Jurassic Park'), 'Steven Spielberg')
        self.assertEqual(returnDirector('The Matrix'), 'Lana Wachowski')
        self.assertEqual(returnDirector('The Dark Knight'), 'Christopher Nolan')
        self.assertEqual(returnDirector('Inception'), 'Christopher Nolan')
        self.assertEqual(returnDirector('The Godfather'), 'Francis Ford Coppola')
        self.assertEqual(returnDirector('The Shawshank Redemption'), 'Frank Darabont')
        self.assertEqual(returnDirector('Pulp Fiction'), 'Quentin Tarantino')
        self.assertEqual(returnDirector('The Avengers'), 'Joss Whedon')
        self.assertEqual(returnDirector('The Lion King'), 'Roger Allers')

    def testReturnYear(self):
        self.assertEqual(returnYear('Jurassic Park'), '1993')
        self.assertEqual(returnYear('The Matrix'), '1999')
        self.assertEqual(returnYear('The Dark Knight'), '2008')
        self.assertEqual(returnYear('Inception'), '2010')
        self.assertEqual(returnYear('The Godfather'), '1972')
        self.assertEqual(returnYear('The Shawshank Redemption'), '1994')
        self.assertEqual(returnYear('Pulp Fiction'), '1994')
        self.assertEqual(returnYear('The Avengers'), '2012')
        self.assertEqual(returnYear('The Lion King'), '1994')

    def testReturnLeadActor(self):
        self.assertEqual(returnLeadActor('Jurassic Park'), 'Sam Neill')
        self.assertEqual(returnLeadActor('The Matrix'), 'Keanu Reeves')
        self.assertEqual(returnLeadActor('The Dark Knight'), 'Christian Bale')
        self.assertEqual(returnLeadActor('Inception'), 'Leonardo DiCaprio')
        self.assertEqual(returnLeadActor('The Godfather'), 'Marlon Brando')
        self.assertEqual(returnLeadActor('The Shawshank Redemption'), 'Tim Robbins')
        self.assertEqual(returnLeadActor('Pulp Fiction'), 'John Travolta')
        self.assertEqual(returnLeadActor('The Avengers'), 'Robert Downey Jr.')
        self.assertEqual(returnLeadActor('The Lion King'), 'Matthew Broderick')

    def testSentimentAnalyzer(self):
        self.assertGreater(analyze_sentiment("I loved The Dark Knight"), 0)
        self.assertLess(analyze_sentiment("I hated The Dark Knight"), 0)
        self.assertGreater(analyze_sentiment('I thought Jurassic Park was a good movie'), 0)
        self.assertLess(analyze_sentiment('I thought The Matrix was a bad movie'), 0)
        self.assertGreater(analyze_sentiment('I liked The Shawshank Redemption'), 0)
        self.assertLess(analyze_sentiment("I wasn't a fan of  Pulp Fiction"), 0)
        self.assertGreater(analyze_sentiment('I absolutely loved The Avengers'), 0)
        self.assertLess(analyze_sentiment('I absolutely hated The Lion King'), 0)


class testSimilarityFunc(unittest.TestCase):
    def testPreprocessText(self):
        self.assertEqual(preprocess_text('Jurassic Park'), 'jurassic park')
        self.assertEqual(preprocess_text('The Dark Knight'), 'dark knight')
        self.assertEqual(preprocess_text('I loved Inception'), 'loved inception')
        self.assertEqual(preprocess_text('When did The Matrix release'), 'matrix release')
        self.assertEqual(preprocess_text('Who directed The Godfather'), 'directed godfather')
        self.assertEqual(preprocess_text('I liked The Shawshank Redemption'), 'liked shawshank redemption')
        self.assertEqual(preprocess_text('What a good action movie to watch'), 'good action movie watch')
        self.assertEqual(preprocess_text('Give me a list of comedy movies'), 'give list comedy movies')
        self.assertEqual(preprocess_text("Who crafted the direction for 'Big Hero 6'?"), 'crafted direction big hero')

    def testGetMovieIndexByTitle(self):
        self.assertEqual(get_movie_index_by_title('Jurassic Park'.lower()), 36)
        self.assertEqual(get_movie_index_by_title('The Dark Knight'.lower()), 0)
        self.assertEqual(get_movie_index_by_title('Inception'.lower()), 2)
        self.assertEqual(get_movie_index_by_title('The Matrix'.lower()), 5)
        self.assertEqual(get_movie_index_by_title('The Godfather'.lower()), 4058)
        self.assertEqual(get_movie_index_by_title('The Shawshank Redemption'.lower()), 4517)
        self.assertEqual(get_movie_index_by_title('Pulp Fiction'.lower()), 4062)
        self.assertEqual(get_movie_index_by_title('The Avengers'.lower()), 83)
        self.assertEqual(get_movie_index_by_title('The Lion King'.lower()), 1717)

    def testGetSimilarMoviesIndex(self):
        similar_jurassic_park = get_similar_movies_index(36)
        self.assertTrue(np.array_equal(similar_jurassic_park, np.array([827, 1432, 8289, 550, 1590])))

        similar_dark_knight = get_similar_movies_index(0)
        self.assertTrue(np.array_equal(similar_dark_knight, np.array([39, 18, 4526, 967, 446])))

        similar_inception = get_similar_movies_index(2)
        self.assertTrue(np.array_equal(similar_inception, np.array([39, 4200, 909, 1714, 4526])))

        similar_matrix = get_similar_movies_index(5)
        self.assertTrue(np.array_equal(similar_matrix, np.array([694, 387, 1381, 1274, 1556])))

        similar_godfather = get_similar_movies_index(4058)
        self.assertTrue(np.array_equal(similar_godfather, np.array([4165, 4060, 4527, 4297, 5169])))

        similar_shawshank_redemption = get_similar_movies_index(4517)
        self.assertTrue(np.array_equal(similar_shawshank_redemption, np.array([5190, 2810, 9499, 7375, 1067])))

        similar_pulp_fiction = get_similar_movies_index(4062)
        self.assertTrue(np.array_equal(similar_pulp_fiction, np.array([40, 86, 5956, 4133, 4977])))

        similar_avengers = get_similar_movies_index(1684)
        self.assertTrue(np.array_equal(similar_avengers, np.array([8781, 3094, 9521, 5447, 2866])))

        similar_lion_king = get_similar_movies_index(1717)
        self.assertTrue(np.array_equal(similar_lion_king, np.array([7762, 2000, 4768, 7633, 9738])))


if __name__ == "__main__":
    unittest.main()
