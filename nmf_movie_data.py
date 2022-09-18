#import numpy as np
import pandas as pd
#from sklearn.decomposition import NMF

from settings import DATA_DIR

# Derive a user item matrix of the rating table and adapt the following code to the MovieLens dataset:
# movie, ratings by users
data = [
    [5, 4, 1, 1, 3],
    [3, 2, 1, 3, 1],
    [3, 3, 3, 3, 5],
    [1, 1, 5, 4, 4],
]
users = ['Ada', 'Bob', 'Steve', 'Margaret']
movieId = ['Titanic', 'Tiffany', 'Terminator', 'Star Trek', 'Star Wars']
# need a dataframe for this
df = pd.DataFrame(data, index=users, columns=movieId)


query = [[1, 2, 5, 4, 5]]


# def predict(df, query):
#     R = df.values
#     # create a model and set the hyperparameters
#     # model assumes R ~ PQ'
#     model = NMF(n_components=2, init='random', random_state=10)

#     model.fit(R)

#     Q = model.components_  # movie-genre matrix
#     P = model.transform(R)  # user-genre matrix

#     # reconstruction error
#     print(f'\nReconstruction error is {model.reconstruction_err_}')

#     nR = np.dot(P, Q)
#     print(f'\nReconstructed Matrix is \n{nR}\n')  # The reconstructed matrix!

#     print("Hidden features or new data point (new user providing ratings for 5 movies)")
#     # predict the hidden features for a new data point
#     # in this case, a new user providing ratings for the 5 movies.
#     t = model.transform(query)
#     print(t)
#     # NOTE I'm confused why do I get a list of only 2 items and not 5?
#     return t


#predict(df, query)

"""
Use the NMF model to produce recommendations for one user.
To map movie names to movie ID's, construct a {name: id} dictionary
To deal with small differences in the names, the fuzzywuzzy package is quite useful
Create an vector of three movies the user likes. Set these to 5 and all others to zero.
"""
