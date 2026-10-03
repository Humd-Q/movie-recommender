"""Flask interface for the content-based TMDB movie recommender."""
from pathlib import Path
import pickle

import pandas as pd
from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)


def load_artifacts():
    with (ROOT / "model.pkl").open("rb") as f:
        movie_data = pickle.load(f)
    with (ROOT / "similarity.pkl").open("rb") as f:
        similarity_matrix = pickle.load(f)
    if len(movie_data) != similarity_matrix.shape[0]:
        raise ValueError("Model and similarity artifacts have different movie counts")
    return movie_data, similarity_matrix


movies, similarity = load_artifacts()
titles = movies["title"].astype(str).tolist()


def get_recommendations(movie_title, limit=12):
    matches = movies.index[movies["title"] == movie_title]
    if len(matches) == 0:
        return []
    movie_index = int(matches[0])
    scores = similarity.getrow(movie_index).toarray().ravel() if hasattr(similarity, "getrow") else similarity[movie_index]
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    result = []
    for idx, score in ranked:
        if idx == movie_index:
            continue
        result.append({"title": movies.iloc[idx]["title"], "score": float(score)})
        if len(result) == limit:
            break
    return result


@app.get("/")
def home():
    return render_template("index.html", movie_list=titles, recommendations=None, selected_movie=None)


@app.post("/recommend")
def recommend():
    movie_title = request.form.get("selected_movie", "")
    recommendations = get_recommendations(movie_title)
    if not recommendations:
        return render_template("index.html", movie_list=titles, recommendations=[], selected_movie=None,
                               error="Please select a movie from the list."), 400
    return render_template("index.html", movie_list=titles, recommendations=recommendations,
                           selected_movie=movie_title, error=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
