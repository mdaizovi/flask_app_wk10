import numpy as np
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer

from settings import DATA_DIR

user_ratings_dict = {"City of Lost Children, The (Cité des enfants perdus, La) (1995)": 5, "Usual Suspects, The (1995)": 4,
                     "Species (1995)": 3, "To Wong Foo, Thanks for Everything! Julie Newmar (1995)": 4, "Before Sunrise (1995)": 4}


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

    def __init__(self, mda):
        self.mda = mda  # MovieDataAggregator so I don't have to instantiate all the time

    @staticmethod
    def cosim(x, y):
        num = np.dot(x, y)
        den = np.sqrt(np.sum(x**2)) * np.sqrt(np.sum(y**2))
        return num/den

    # def similarity_table(self, neighboring_users_df, new_user_df):

    #     # We want to create user pairs of cosine similarities
    #     # Initialize an empty list that shall contain all user pair lists
    #     all_user_pairs = []
    #     # We shall go through every user and capture the indices between this user and the rest of the users
    #     users = list(neighboring_users_df.index)
    #     for user in users:
    #         # Initialize another empty list to capture user-users indices
    #         single_user_list = []
    #         for single_user in users:
    #             single_user_list.append(
    #                 MovieRecommender.cosim(new_user_df.loc[user], new_user_df.loc[single_user]))
    #         all_user_pairs.append(single_user_list)
    #     return pd.DataFrame(all_user_pairs, columns=neighboring_users_df.index, index=list(neighboring_users_df.index))

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


# WIP WIP WIP WIP

    def get_user_nbc_recommendations(self, user_ratings_dict):

        user_movie_title_list = list(user_ratings_dict.keys())
        user_movie_id_list = [int(self.mda.reverse_movie_map.get(
            x)) for x in user_movie_title_list]

        neighboring_users_list = self.get_ids_of_users_who_saw_same_movies(
            user_ratings_dict)
        # make a dataframe of only reviews by users who have seen at least one of same movie as our user.
        neighbors_reviews_df = self.get_reviews_df_by_user_id_list(
            neighboring_users_list)
        print("\nneighbors_reviews_df.head")
        print(neighbors_reviews_df.head())
        print(neighbors_reviews_df.shape)

        # make a dataframe that includes our user as well as the other user reviews
        dummy_id = 99999999  # arbitrary high userId that I know doesn't exist
        columns = ["userId", "movieId", "rating"]
        data = []
        for k, v in user_ratings_dict.items():
            row = [dummy_id, self.mda.reverse_movie_map.get(k), v]
            data.append(row)
        user_ratings_df = pd.DataFrame(data, columns=columns)
        print("\nuser_ratings_df")
        print(user_ratings_df.head())
        print(user_ratings_df.shape)

        neighbors_reviews_df = pd.concat(
            [neighbors_reviews_df, user_ratings_df], ignore_index=True, sort=False)
        print("\nneighbors_reviews_df.head again")
        print(neighbors_reviews_df.head())
        print(neighbors_reviews_df.shape)
        print("\nand now the tai;")
        print(neighbors_reviews_df.tail())

        reshaped_df = self.reshape_ratings_df(neighbors_reviews_df)
        print("reshaped_df.head()")
        print(reshaped_df.head())
        print(reshaped_df.shape)

        print("sanity check: how many unique users, how many unique movies?")
        print("users")
        print(len(neighbors_reviews_df["userId"].unique()))
        print("movies")
        print(len(neighbors_reviews_df["movieId"].unique()))

        # imputer = SimpleImputer(strategy="constant", fill_value=0)
        # ratings_df = pd.DataFrame(imputer.fit_transform(reshaped_df), columns=reshaped_df.columns, index=list(reshaped_df.index))

        # Find users with high cosine similarity, and use each other's ratings to recommend movies the other hasn't seen
        # Extend this to a couple of neighbours:
        # Use average ratings of those highly similar users:
        # sum(ratings)/N
        # Use weighted average of their ratings:
        # sum(similarity * rating)/sum(similarity)
        # Extend this to all users and use weighted average

    def get_generic_recommendations(self):
        movies = random.shuffle(self.generic_popular_movies)
        return movies[:3]
