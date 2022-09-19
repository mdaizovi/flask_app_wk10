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

    @property
    def __data_dir__(self):
        return f'{DATA_DIR}ml-latest-small/'

    @staticmethod
    def format_imdb_id(imdb_id):
        movie_id = "tt"
        imdb_id = str(imdb_id)
        if len(imdb_id) < 7:
            diff = 7 - len(imdb_id)
            for i in range(diff):
                movie_id += "0"
        movie_id += imdb_id
        return movie_id

    # --- Private methods
    def _build_df_from_csv(self, filename):
        """"
        Builds dataframe of one of the movie CSVs.
        Returns dataframe
        """
        df = pd.read_csv(f'{self.__data_dir__}{filename}.csv',
                         index_col=0)
        return df

    def _build_master_df(self):
        """"
        Builds one large dataframe from several smaller ones;
        essentially a full database.
        individual user ratings are aggregated so we get an averte rating and a rating count.
        """
        files = ["movies", "ratings", "ratings_agg", "links", "imdb_img"]
        for f in files:
            df = self._build_df_from_csv(f)
            setattr(self, f"df_{f}", df)

        # drop timestamp, haven't found a use for it
        self.df_ratings.reset_index().rename(columns={'index': 'userId'})
        self.df_ratings.reset_index(inplace=True)
        self.df_ratings.drop(columns=["timestamp"], inplace=True)

        tmp = pd.merge(self.df_movies, self.df_links, on="movieId")
        self.df = pd.merge(tmp, self.df_ratings_agg, on="movieId")

    def _aggregate_ratings_to_csv(self):
        """
        Aggregates movie ratings and creates an output csv;
        only needs to be run once (unless source data changes)
        """
        ratings = self._build_df_from_csv("ratings")
        ratings.drop(columns=['timestamp'])

        temp_df = ratings.groupby('movieId').agg(rating_count=(
            'rating', 'size'), rating_mean=('rating', 'mean')).reset_index()
        temp_df.set_index('movieId', inplace=True)
        temp_df.to_csv(f'{self.__data_dir__}ratings_agg.csv')

    def _collect_movie_posters_to_csv(self, api_key):
        """
        Aggregates movie posters from API and creates an output csv;
        only needs to be run once (unless source data changes)
        """
        favorites_df = self.get_most_popular_movies(
            n_results=50, min_ratings=20)

        df = pd.merge(favorites_df, self.df_links, on="movieId")
        movieId = df.to_dict()['imdbId']  # makes dict of {movieId:imdbId}
        columns = ["movieId", "img"]
        data = []
        for k, v in movieId.items():
            img = self.get_poster_img(api_key, v)
            data.append([k, img])
        img_df = pd.DataFrame(data, columns=columns)
        img_df.set_index(columns[0], inplace=True)
        img_df.to_csv(f'{self.__data_dir__}imdb_img.csv')

# move this to somewhere else
        # imputer = SimpleImputer(strategy="constant", fill_value=0)
        # self.df_ratings = pd.DataFrame(imputer.fit_transform(
        #     ratings), columns=ratings.columns, index=list(ratings.index))

    def _build_imdb_image_df(self):
        df2 = pd.merge(self.df_imdb_img, self.df, on="movieId")[
            ['title', 'img']]
        return df2

    def _build_movie_map(self):
        """Returns a dictionary of {movieId: 'Movie Title'}"""
        return self.df_movies.to_dict()['title']

    def _build_reverse_movie_map(self):
        """Returns a dictionary of {'Movie Title':movieId}"""
        movie_map = self._build_movie_map()
        return {v: k for k, v in movie_map.items()}

    # --- Public methods

    def get_most_popular_movies(self, n_results=20, min_ratings=20):

        # 4.4 gives me 3 results, 4.3 gives me 9, 4.2 gives me 33
        # IMO these resukts were a little weird. Mabe I'll do movies with most reviews total and then sort by rating valiue
        # df_min_threshold = self.df[(self.df['rating_mean'] >= 4.2) & (
        #     self.df['rating_count'] >= min_ratings)][['title', 'imdbId', 'rating_count']]
        # favorites = df_min_threshold.sort_values(
        #     'rating_count', ascending=False).dropna().head(n_results)

        df_min_threshold = self.df_ratings_agg[(
            self.df_ratings_agg['rating_count'] >= min_ratings)]
        # df_min_threshold.shape  is (1297,2)
        # maybe I'll just take top 10% of most frewuently rated movies
        top10_n = int(df_min_threshold.shape[0]/10)
        most_rated = self.df_ratings_agg.sort_values(
            'rating_count', ascending=False).head(top10_n)
        favorites = most_rated.sort_values(
            'rating_mean', ascending=False).head(n_results)
        # Get movie titles
        favorites_with_titles = pd.merge(
            favorites, self.df_movies, on="movieId")
        return favorites_with_titles

    def get_poster_img(self, api_key, imdb_id):
        """Gets IMDB poster url from https://www.omdbapi.com/"""
        base_url = f"https://www.omdbapi.com/?apikey={api_key}&i="
        movie_id = MovieDataAggregator.format_imdb_id(imdb_id)
        url = base_url + movie_id
        resp = requests.get(url)
        j = resp.json()
        json_key = "Poster"
        if json_key in j:
            return j[json_key]
        else:
            print(f"\n\n{j}\n\n")
