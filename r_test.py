import numpy as np
import pandas as pd

from movie_data_aggregator import *
from recommender import *

urd1 = {"City of Lost Children, The (Cité des enfants perdus, La) (1995)": 5, "Usual Suspects, The (1995)": 4,
        "Species (1995)": 3, "To Wong Foo, Thanks for Everything! Julie Newmar (1995)": 4, "Before Sunrise (1995)": 4}

urd2 = {'Matrix, The (1999)': 5, 'Matrix Reloaded, The (2003)': 4,
        'Matrix Revolutions, The (2003)': 3, 'Animatrix, The (2003)': 3, 'Aeon Flux (2005)': 4}


mda = MovieDataAggregator()
mr = MovieRecommender(mda=mda)


def get_test_recs(user_ratings_dict, n=25):
    recs_df = mr.get_user_nbc_recommendations(
        user_ratings_dict, max_neighbors=n)
    print(recs_df.head())
    recs = mda.convert_movie_is_list_to_title_imdb_dict(
        list(recs_df.index.values))
    print(recs)
