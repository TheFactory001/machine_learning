from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load movies data
movies_data = pd.read_csv('movies.csv')
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + \
    movies_data['tagline'] + ' ' + movies_data['cast'] + \
    ' ' + movies_data['director']
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)


@app.route('/', methods=['GET', 'POST'])
def index():
    recommended_movies = []

    if request.method == 'POST':
        movie_name = request.form['movie_name']
        list_of_all_titles = movies_data['title'].tolist()
        close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

        if close_match:
            close_match = close_match[0]
            index_of_movie = movies_data[movies_data.title ==
                                         close_match]['index'].values[0]
            similarity_score = list(enumerate(similarity[index_of_movie]))
            sorted_similar_movies = sorted(
                similarity_score, key=lambda x: x[1], reverse=True)

            for i, movie in enumerate(sorted_similar_movies):
                index = movie[0]
                title_from_index = movies_data[movies_data.index ==
                                               index]['title'].values[0]
                if i < 10:
                    recommended_movies.append(title_from_index)

    return render_template('index.html', recommended_movies=recommended_movies)


if __name__ == '__main__':
    app.run(debug=True)
