"""Data loading, statistics, and chronological train/test splitting."""

from pathlib import Path

import pandas as pd


def load_data(data_dir):
    """Load MovieLens ratings and movies CSV files from ``data_dir``."""
    data_dir = Path(data_dir)
    ratings = pd.read_csv(data_dir / "ratings.csv")
    movies = pd.read_csv(data_dir / "movies.csv")
    return ratings, movies


def dataset_statistics(ratings, movies):
    """Return user, movie, rating, and observed matrix-density statistics."""
    n_users = ratings["userId"].nunique()
    n_movies = movies["movieId"].nunique()
    n_ratings = len(ratings)
    density = n_ratings / (n_users * n_movies)
    return {
        "n_users": n_users,
        "n_movies": n_movies,
        "n_ratings": n_ratings,
        "density": density,
    }


def temporal_train_test_split(ratings):
    """Split each user's ratings chronologically, retaining at least one train row."""
    users = ratings["userId"].unique()
    train_list = []
    test_list = []

    for i in range(users.shape[0]):
        ratings_subset = ratings.loc[ratings["userId"] == users[i]].sort_values(
            "timestamp"
        )
        limit = max(1, int(ratings_subset.shape[0] * 0.8))
        train_list.append(ratings_subset.iloc[:limit])
        test_list.append(ratings_subset.iloc[limit:])

    ratings_train_set = pd.concat(train_list, ignore_index=True)
    ratings_test_set = pd.concat(test_list, ignore_index=True)
    return ratings_train_set, ratings_test_set
