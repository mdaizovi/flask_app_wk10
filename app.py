from flask import Flask, Response, render_template, redirect, url_for, request,  jsonify
from flask_bootstrap import Bootstrap, Bootstrap4
from flask_wtf import FlaskForm
import json
import numpy as np
import random
from wtforms import StringField, SubmitField, SelectField, StringField, Form
from wtforms.validators import DataRequired

from recommender import MovieRecommender
from movie_data_aggregator import MovieDataAggregator

mda = MovieDataAggregator()
# dict of most popular movies {title:img}
img_df = mda._build_imdb_image_df()
img_df.set_index('title', inplace=True)
MOVIE_IMG_DICT = img_df.to_dict()["img"]
MOVIES = list(mda.df["title"].unique())
MOST_POPULAR_DICT = mda.get_most_popular_movies().to_dict()
MOST_POPULAR = list(MOST_POPULAR_DICT['title'].keys())

MOVIE_IMDB_DICT = mda.df_links.to_dict()["imdbId"]


app = Flask(__name__)
app.config['SECRET_KEY'] = 'my-top-secret-string'
bootstrap = Bootstrap4(app)


class MovieForm(Form):
    text1 = 'Please rate movie from 1-5'
    text2 = 'Insert Movie Title'
    choices = [(n, n) for n in range(1, 6)]

    autocomp_1 = StringField(text2, id='movie_autocomplete_1')
    rating_1 = SelectField(text1, choices=choices,
                           validators=[DataRequired()])
    autocomp_2 = StringField(text2, id='movie_autocomplete_2')
    rating_2 = SelectField(text1, choices=choices,
                           validators=[DataRequired()])
    autocomp_3 = StringField(text2, id='movie_autocomplete_3')
    rating_3 = SelectField(text1, choices=choices,
                           validators=[DataRequired()])
    autocomp_4 = StringField(text2, id='movie_autocomplete_4')
    rating_4 = SelectField(text1, choices=choices,
                           validators=[DataRequired()])
    autocomp_5 = StringField(text2, id='movie_autocomplete_5')
    rating_5 = SelectField(text1, choices=choices,
                           validators=[DataRequired()])


@app.route('/_autocomplete', methods=['GET'])
def autocomplete():
    return Response(json.dumps(MOVIES), mimetype='application/json')


@app.route('/')
def index():
    random_keys = np.random.choice(
        list(MOVIE_IMG_DICT.keys()), size=3, replace=True)
    top_titles = [
        {"title": x, "img": MOVIE_IMG_DICT.get(x)} for x in random_keys]
    return render_template('index.html', top_titles=top_titles)


@app.route('/recommender', methods=['GET', 'POST'])
def recommender():

    if request.method == 'POST':
        # Enter your function POST behavior here
        movie_ratings = {}
        form_results = request.form.to_dict(flat=False)
        for i in range(5):
            movie_title = form_results.get(f"autocomp_{i+1}")[0]
            movie_rating = form_results.get(f"rating_{i+1}")[0]
            movie_ratings[movie_title] = movie_rating
        mr = MovieRecommender(mda=mda)
        recs_df = mr.get_user_nbc_recommendations(
            movie_ratings, max_neighbors=40)
        rec_id_list = list(recs_df.index.values)
        recs = []
        for m in rec_id_list:
            title = mda.movie_map.get(m)
            imdb_id = MOVIE_IMDB_DICT.get(m)
            imdb_clean = mda.format_imdb_id(imdb_id)
            recs.append({title: imdb_clean})

        form = None
    else:
        recs = None
        form = MovieForm()

    return render_template('recommendations.html',
                           movies=MOVIES, recommended=recs, form=form)


# run with 'python app.py'
if __name__ == "__main__":
    app.run(debug=True, port=5000)


# Alternately:
# every time we start new terminal
# `export FLASK_APP=app.py`
#  `export FLASK_DEBUG=1`
# `flask run`
