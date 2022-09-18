import numpy as np
import pandas as pd
import requests

from sklearn.impute import SimpleImputer

from settings import DATA_DIR


class MovieDataAggregator:

    def __init__(self):
        self._build_master_df()
        self.movie_map = self._build_movie_map()
        self.reverse_movie_map = self._build_reverse_movie_map()

    # --- Private methods
    def _build_df_from_csv(self, filename):
        """"
        Builds dataframe of one of the movie CSVs.
        Returns dataframe
        """
        df = pd.read_csv(f'{DATA_DIR}ml-latest/{filename}.csv',
                         index_col=0)
        return df

    def _build_master_df(self):
        """"
        Builds one large dataframe from several smaller ones;
        essentially a full database.
        individual user ratings are aggregated so we get an averte rating and a rating count.
        """
        files = ["movies", "ratings_agg", "links", "imdb_img"]
        for f in files:
            df = self._build_df_from_csv(f)
            setattr(self, f"df_{f}", df)
        df1 = pd.merge(self.df_movies, self.df_links, on="movieId")
        df2 = pd.merge(df1, self.df_ratings_agg, on="movieId")

        # shape (53889, 7)
        self.df = df2
        return df2

    def _build_ratings_df(self):
        """
        So far I haven't found a solution for putting such a large file on github 
        so I will have to only run this locally.
        """
        ratings = self._build_df_from_csv("ratings")
        ratings.drop(columns=['timestamp'])
        imputer = SimpleImputer(strategy="constant", fill_value=0)
        self.df_ratings = pd.DataFrame(imputer.fit_transform(
            ratings), columns=ratings.columns, index=list(ratings.index))

    def _build_imdb_image_df(self):
        # df2 = pd.merge(df1, self.df_imdb_img, on="movieId")
        df1 = self.df_imdb_img.drop(columns=['imdbId'])
        df2 = pd.merge(df1, self.df, on="movieId")[['title', 'img']]
        return df2

    def _build_movie_map(self):
        """Returns a dictionary of {movieId: 'Movie Title'}"""
        return self.df_movies.to_dict()['title']

    def _build_reverse_movie_map(self):
        """Returns a dictionary of {'Movie Title':movieId}"""
        movie_map = self._build_movie_map()
        return {v:k for k,v in movie_map.items()}

    # --- Public methods


    def get_most_popular_movies(self, n_results=20, min_ratings=20):
        # 4.4 gives me 3 results, 4.3 gives me 9, 4.2 gives me 33
        df_min_threshold = self.df[(self.df['rating_mean'] >= 4.2) & (
            self.df['rating_count'] >= min_ratings)][['title', 'imdbId', 'rating_count']]
        favorites = df_min_threshold.sort_values(
            'rating_count', ascending=False).dropna().head(n_results)

        return favorites

    def get_poster_img(self, api_key, imdb_id):
        """Gets IMDB poster url from https://www.omdbapi.com/"""
        movie_id = ""
        imdb_id = str(imdb_id)
        base_url = f"https://www.omdbapi.com/?apikey={api_key}&i=tt"
        if len(imdb_id) < 7:
            diff = 7 - len(imdb_id)
            for i in range(diff):
                movie_id += "0"
        movie_id += imdb_id
        url = base_url + movie_id
        resp = requests.get(url)
        j = resp.json()
        return j["Poster"]

    def cosim(x, y):
        """
        Calculate Cosine, for Cosine-Similarities between users
        """
        num = np.dot(x, y)
        den = np.sqrt(np.sum(x**2)) * np.sqrt(np.sum(y**2))
        return num/den
