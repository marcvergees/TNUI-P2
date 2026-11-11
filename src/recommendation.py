"""Personalized recommendations and user rating history."""

import pandas as pd


def recommend_movies(model, user_id, movies, ratings_train_set, n=10):
    """Return the top ``n`` known, unrated movies for a user."""
    columns = ["movieId", "title", "genres", "predicted_rating"]
    if user_id not in model.user2idx:
        return pd.DataFrame(columns=columns)

    seen_movies = ratings_train_set.loc[
        ratings_train_set["userId"] == user_id, "movieId"
    ].tolist()
    candidates = []

    for movie_id in model.movie_ids:
        if movie_id not in seen_movies:
            prediction = model.predict_rating(user_id, movie_id)
            candidates.append((movie_id, prediction))

    candidates.sort(key=lambda item: item[1], reverse=True)
    top_candidates = candidates[:n]
    recommendations = pd.DataFrame(
        top_candidates, columns=["movieId", "predicted_rating"]
    )
    recommendations = recommendations.merge(
        movies[["movieId", "title", "genres"]], on="movieId", how="left"
    )
    return recommendations[columns]


def get_user_history(user_id, ratings_train_set, movies):
    """Return a user's train ratings with titles and genres, highest rating first."""
    user_history = ratings_train_set.loc[ratings_train_set["userId"] == user_id]
    user_history = user_history.merge(movies, on="movieId", how="left")
    user_history = user_history.sort_values(by="rating", ascending=False)
    return user_history[["movieId", "title", "genres", "rating"]]
