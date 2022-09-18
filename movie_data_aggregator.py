import numpy as np
import pandas as pd
import requests

from settings import DATA_DIR


class MovieDataAggregator:

    def __init__(self):
        self._build_master_df()

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
        files = ["movies", "ratings", "links", "imdb_img"]
        #files = ["movies", "ratings", "links"]
        for f in files:
            df = self._build_df_from_csv(f)
            setattr(self, f"df_{f}", df)

        df1 = pd.merge(self.df_movies, self.df_links, on="movieId")

        temp_df = self.df_ratings.groupby('movieId').agg(rating_count=(
            'rating', 'size'), rating_mean=('rating', 'mean')).reset_index()
        temp_df.set_index('movieId', inplace=True)

        df2 = pd.merge(df1, temp_df, on="movieId")
        # shape (53889, 7)
        self.df = df2
        return df2

    def _build_imdb_image_df(self):
        # df2 = pd.merge(df1, self.df_imdb_img, on="movieId")
        df1 = self.df_imdb_img.drop(columns=['imdbId'])
        df2 = pd.merge(df1, self.df, on="movieId")[['title', 'img']]
        return df2

    # --- Public methods
    def build_movie_map(self):
        """Returns a dictionary of {movieId: 'Movie Title'}"""
        return self.df_movies.to_dict()['title']

    def get_most_popular_movies(self, n_results=20, min_ratings=20):
        # 4.4 gives me 3 results, 4.3 gives me 9, 4.2 gives me 33
        df_min_threshold = self.df[(self.df['rating_mean'] >= 4.2) & (
            self.df['rating_count'] >= min_ratings)][['title', 'imdbId', 'rating_count']]
        favorites = df_min_threshold.sort_values(
            'rating_count', ascending=False).dropna().head(n_results)

        # images
        # test = pd.merge(df2, self.df_imdb_img, on="movieId")

        # return favorites.to_dict()['count']
        # return favorites.to_dict()
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
