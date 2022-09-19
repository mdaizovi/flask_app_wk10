import numpy as np
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity

from settings import DATA_DIR


class MovieRecommender:
    generic_popular_movies = [
        "The Shawshank Redemption",
        "Star Wars: Episode IV - A New Hope",
        "Pulp Fiction",
        "The Dark Knight",
        "Forrest Gump",
        "Inception",
        "The Matrix",
        "Saving Private Ryan",
        "Casablanca",
        "The Lion King"
    ]

    @staticmethod
    def cosim(x, y):
        num = np.dot(x, y)
        den = np.sqrt(np.sum(x**2)) * np.sqrt(np.sum(y**2))
        return num/den

    def similarity_table(self, neighboring_users_df, new_user_df):

        # We want to create user pairs of cosine similarities
        # Initialize an empty list that shall contain all user pair lists
        all_user_pairs = []
        # We shall go through every user and capture the indices between this user and the rest of the users
        users = list(neighboring_users_df.index)
        for user in users:
            # Initialize another empty list to capture user-users indices
            single_user_list = []
            for single_user in users:
                single_user_list.append(
                    MovieRecommender.cosim(new_user_df.loc[user], new_user_df.loc[single_user]))
            all_user_pairs.append(single_user_list)
        return pd.DataFrame(all_user_pairs, columns=neighboring_users_df.index, index=list(neighboring_users_df.index))

    def get_user_nbc_recommendations(self, mda, user_ratings_dict):

        movie_title_list = list(user_ratings_dict.keys())
        movie_id_list = [int(mda.reverse_movie_map.get(
            x)) for x in movie_title_list]

        neighboring_users_df = mda.df_ratings[mda.df_ratings['movieId'].isin(
            movie_id_list)]
        # TODO drop timestamp if you ahven't already
        #neighboring_users_df.drop(columns=["timestamp"], inplace=True)
        tmp = neighboring_users_df.reset_index().rename(
            columns={'index': 'userId'})
        neighboring_users_ids = tmp["userId"].unique()

        tmp_df_ratings = mda.df_ratings.reset_index().rename(
            columns={'index': 'userId'})
        neighboring_users_ratings_df = tmp_df_ratings[tmp_df_ratings['userId'].isin(
            neighboring_users_ids)]
        neighboring_users_ratings_df.set_index('userId', inplace=True)

        dummy_id = 99999999
        columns = ["userId", "movieId", "rating"]
        data = []
        for k, v in user_ratings_dict.items():
            row = [dummy_id, mda.reverse_movie_map.get(k), v]
            data.append(row)
        user_ratings_df = pd.DataFrame(data, columns=columns)
        user_ratings_df.set_index('userId', inplace=True)

        # cosine_table = self.similarity_table(
        #     neighboring_users_df, user_ratings_df)

        # or add user to the ratings_df and
        frames = [neighboring_users_ratings_df, user_ratings_df]
        combined_df = pd.concat(frames)
        df = combined_df.reset_index().rename(columns={'index': 'userId'})

        neighbor_rated_movies = df["movieId"].unique()
        unseen_movies_ids = [
            int(x) for x in neighbor_rated_movies if not x in movie_id_list]

       # pd.DataFrame(cosine_similarity(df))

       # TODO rewrite Make predictions for unseen movies so it works with this

        predicted_ratings = []
        for movie in unseen_movies_ids:
            # Capture the users who have watched the movie

            other_users = df[df['movieId'] == movie].index
            # Go through each of these users who have watched the movie and calculate av. rating
            # Initialize numerator and denominator
            num = 0
            den = 0
            for user in other_users:
                # capture rating for this `user'
                user_rating = df[user][movie]
                # Calculate the similarity between this user and the target user
                similarity = MovieRecommender.cosim(
                    ratings_df.loc[user], ratings_df.loc[target_user])
                num += user_rating * similarity
                den += similarity
            if den != 0:
                predicted_rating = num/den
            else:
                predicted_rating = 0
            predicted_ratings.append((movie, predicted_rating))

    def reshape_ratings_df(self, partial_df):
        # takes a df like after importing ratings.csv
        # rearanges so is like in class examplel with movie IDs as column names and ratings as values.
        movie_ids = list(partial_df["movieId"].unique())
        columns = ["userId"] + movie_ids
        # Each row leeds to be a list of their userId, then rating for the movies.

        data = []
        for k, v in user_ratings_dict.items():
            row = [dummy_id, mda.reverse_movie_map.get(k), v]
            data.append(row)
        user_ratings_df = pd.DataFrame(data, columns=columns)
        user_ratings_df.set_index('userId', inplace=True)

    def get_generic_recommendations(self):
        movies = random.shuffle(self.generic_popular_movies)
        return movies[:3]
