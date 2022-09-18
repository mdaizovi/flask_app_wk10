import pandas as pd
import random

from settings import DATA_DIR
"""
- [X] Calculate the average rating for each movie in the dataset
- [X] Filter out movies that have been watched by less than 20 users
- [X] Recommend the top ten movies that a user has not seen yet
- [ ] Write a function recommend_popular(query, ratings, k=10) that gets a user query of 
    rated movie-ids and the ratings table as input. It returns a list of k movie-ids.
    The user query is a python dictionary that looks like this: {12: 5, 234: 1, 234: 4.5
"""
df = pd.read_csv(f'{DATA_DIR}ml-latest/ratings.csv', index_col=0)

# Calculate the average rating for each movie in the dataset
df.groupby(['movieId']).mean()  # This gets average rating and timestamp

# Filter out movies that have been watched by less than 20 users
# way 1:
#df.groupby("movieId").filter(lambda x: len(x) >= 20)
# way 2:
#df2 = df[df['movieId'].map(df['movieId'].value_counts()) >= 20]
# way 3:
df2 = df[df.groupby("movieId")['movieId'].transform('size') >= 20]

# Recommend the top ten movies that a user has not seen yet
df2.sort_values('rating', ascending=False).head(10)


def recommend_popular(query, ratings, k=10):
    """
    Gets a user query of rated movie-ids and the ratings table as input. 
    Returns a list of k movie-ids.
    The user query is a python dictionary that looks like this: {12: 5, 234: 1, 234: 4.5}
    """
    already_seen = list(query.keys())
    df = ratings.sort_values('rating', ascending=False)
    df[~df['movieId'].isin(already_seen)]
    return df


# NOTE: will need to display 10 recommendations on a result page
# Think about adding movie images: https://spiced.space/ginger-pipeline/ds-course/chapters/project_movie_recommender/web/building_a_web_app.html
movies = [
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


def get_recommendations():
    random.shuffle(movies)
    return movies[:3]
