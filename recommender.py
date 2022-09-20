from textwrap import fill
import numpy as np
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer

from settings import DATA_DIR


class MovieRecommender:

    def __init__(self, mda):
        self.mda = mda  # MovieDataAggregator so I don't have to instantiate all the time

    @staticmethod
    def cosim(x, y):
        num = np.dot(x, y)
        den = np.sqrt(np.sum(x**2)) * np.sqrt(np.sum(y**2))
        return num/den

    def get_ids_of_users_who_saw_same_movies(self, user_ratings_dict):
        """
        Receives raw dict of {title:rating}
        gets movieId,
        finds users who reviewed at least 1 of the same movies
        Returns list of UserId 
        """
        movie_title_list = list(user_ratings_dict.keys())
        movie_id_list = [int(self.mda.reverse_movie_map.get(
            x)) for x in movie_title_list]

        neighboring_users_df = self.mda.df_ratings[self.mda.df_ratings['movieId'].isin(
            movie_id_list)]

        return list(neighboring_users_df["userId"].unique())

    def get_reviews_df_by_user_id_list(self, user_id_list):
        """
        Receives list of UserIds
        returns df of reviews for said users
        """
        return self.mda.df_ratings[self.mda.df_ratings['userId'].isin(
            user_id_list)]

    def reshape_ratings_df(self, user_review_df):
        """takes a df like after importing ratings.csv
        expect columns [index, userId, movieId, rating]
        rearanges so is like in class example with movie IDs as column names and ratings as values.
        """
        return pd.pivot_table(user_review_df,
                              index='userId',
                              columns='movieId',
                              values='rating')

    def get_predicted_ratings_from_neighbors_and_unseen_movie_ids(self, unseen_movie_ids, closest_neighbors, ratings_df):
        # for each unseen movie, get user who also saw it and calculate av. rating
        closest_neighbor_ids = list(closest_neighbors.index.values)

        # This is super time consuming!
        predicted_ratings = []
        for movie in unseen_movie_ids:
            num = 0
            den = 0
            for user in closest_neighbor_ids:
                # capture rating for this `user'
                similarity = closest_neighbors.loc[user, "cosim"]
                try:
                    user_rating = ratings_df.loc[user, movie]
                    num += user_rating * similarity
                    den += similarity
                except KeyError:
                    # if user didn't see this movie
                    pass
            if den != 0:
                predicted_rating = num/den
            else:
                predicted_rating = 0
            predicted_ratings.append((movie, predicted_rating))
        # Make df of movie scores
        predicted_rating_df = pd.DataFrame(
            predicted_ratings, columns=["movieId", "rating"]).sort_values('rating', ascending=False)
        predicted_rating_df.set_index('movieId', inplace=True)
        return predicted_rating_df

    def make_cosine_simularity_table(self, neighboring_users_list, ratings_df, target_user_id):

        # Make cosine simularity table between our user and other users
        cosim_columns = ["userId", "cosim"]
        cosim_data = []
        for u in neighboring_users_list:
            similarity = MovieRecommender.cosim(
                ratings_df.loc[u], ratings_df.loc[target_user_id])
            cosim_data.append([u, similarity])
        cosim_df = pd.DataFrame(
            data=cosim_data, columns=cosim_columns)
        cosim_df.set_index('userId', inplace=True)
        cosim_df = cosim_df.sort_values('cosim', ascending=False)

        return cosim_df

    def get_user_nbc_recommendations(self, user_ratings_dict, max_neighbors=20):

        user_movie_title_list = list(user_ratings_dict.keys())
        user_movie_id_list = [int(self.mda.reverse_movie_map.get(
            x)) for x in user_movie_title_list]

        neighboring_users_list = self.get_ids_of_users_who_saw_same_movies(
            user_ratings_dict)
        # make a dataframe of only reviews by users who have seen at least one of same movie as our user.
        neighbors_reviews_df = self.get_reviews_df_by_user_id_list(
            neighboring_users_list)

        # make a dataframe that includes our user as well as the other user reviews
        dummy_id = 99999999  # arbitrary high userId that I know doesn't exist
        columns = ["userId", "movieId", "rating"]
        data = []
        for k, v in user_ratings_dict.items():
            row = [dummy_id, self.mda.reverse_movie_map.get(k), v]
            data.append(row)
        user_ratings_df = pd.DataFrame(data, columns=columns)

        neighbors_reviews_df = pd.concat(
            [neighbors_reviews_df, user_ratings_df], ignore_index=True, sort=False)
        reshaped_df = self.reshape_ratings_df(neighbors_reviews_df)

        # fill_value = 0  # or should i do average score?
        fill_value = self.mda.df_ratings["rating"].mean()
        imputer = SimpleImputer(strategy="constant", fill_value=fill_value)
        ratings_df = pd.DataFrame(imputer.fit_transform(
            reshaped_df), columns=reshaped_df.columns, index=list(reshaped_df.index))

        cosim_df = self.make_cosine_simularity_table(
            neighboring_users_list=neighboring_users_list, ratings_df=ratings_df, target_user_id=dummy_id)

        # This takes too damn long. user a shorter list of closest users (max_neighbors)
        if cosim_df.shape[0] > max_neighbors:
            closest_neighbors = cosim_df.head(max_neighbors)
        else:
            closest_neighbors = cosim_df

        # # get list of unseen movies
        unseen_movie_ids = list(neighbors_reviews_df[~neighbors_reviews_df['movieId'].isin(
            user_movie_id_list)]["movieId"].unique())

        predicted_rating_df = self.get_predicted_ratings_from_neighbors_and_unseen_movie_ids(
            unseen_movie_ids, closest_neighbors, ratings_df)

        print(predicted_rating_df.head())
        return predicted_rating_df.head(3)
