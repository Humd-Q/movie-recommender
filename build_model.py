"""Build the content model and sparse cosine-similarity artifact from TMDB CSVs."""
import ast
from pathlib import Path
import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent


def names(value):
    try:
        return [item["name"].replace(" ", "") for item in ast.literal_eval(value)]
    except (ValueError, TypeError, SyntaxError):
        return []


def directors(value):
    try:
        crew = ast.literal_eval(value)
        return [next((person["name"].replace(" ", "") for person in crew if person.get("job") == "Director"), "")]
    except (ValueError, TypeError, SyntaxError):
        return [""]


def build():
    movies = pd.read_csv(ROOT / "data" / "tmdb_5000_movies.csv")
    # The supplied credits file has thousands of empty trailing columns in its header.
    credits = pd.read_csv(ROOT / "data" / "tmdb_5000_credits.csv", usecols=["title", "cast", "crew"], low_memory=False)
    credits = credits.drop_duplicates("title").set_index("title")
    movies = movies.join(credits[["cast", "crew"]], on="title", rsuffix="_credits")
    movies = movies.dropna(subset=["title", "overview", "genres", "keywords", "cast", "crew"])
    features = []
    for row in movies.itertuples(index=False):
        genres = names(row.genres)
        keywords = names(row.keywords)
        cast = names(row.cast)[:3]
        director = directors(row.crew)
        features.append(" ".join(row.overview.split() + genres * 2 + keywords + cast + director))
    model = pd.DataFrame({"id": movies["id"].astype(int).values,
                          "title": movies["title"].astype(str).values,
                          "tags": features})
    vectorizer = CountVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(model["tags"])
    similarity = cosine_similarity(vectors, dense_output=False).tocsr()
    with (ROOT / "model.pkl").open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    with (ROOT / "similarity.pkl").open("wb") as f:
        pickle.dump(similarity, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Built {len(model)} movies; sparse similarity matrix {similarity.shape}, {similarity.nnz:,} nonzero entries.")


if __name__ == "__main__":
    build()
