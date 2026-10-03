# Movie Recommendation System:

A  **Movie Recommendation System** using **Flask**, **Python**, and **Machine Learning**, containerized via **Docker**!


The goal of this project is to recommend similar movies based on a user's selection via a content-based recommendation approach. It suggests movies with similar content (genres, keywords, etc.) rather than simply relying on user ratings.

By Humd Qazi!

---
## Run Locally:

For local use, run from this folder:

```bash
python -m pip install -r requirements.txt
python app.py
```

`model.pkl` and `similarity.pkl` are included. Rebuild them after changing the CSV files with `python build_model.py` or run `analysis.ipynb` (The similarity matrix is stored sparsely). The app displays text recommendations without poster lookups or a TMDB API key.

The bonus collaborative and hybrid models need user-movie ratings, which are not included in this dataset.

### Docker:
```bash
docker build -t movie-recommender .
```

### Running the Container:
```bash
docker run -p 5000:5000 movie-recommender
```

Then go to `http://localhost:5000` :)

---

## Planned Improvements:

- Add user-based collaborative filtering
- Deploy on a cloud platform like Heroku or AWS

