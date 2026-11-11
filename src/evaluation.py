"""Prediction and ranking metrics for the recommender."""

import numpy as np


def evaluate_regression(model, ratings_test_set):
    """Calculate RMSE and MAE over test rows the model can predict."""
    ratings_test_set_predictions = []
    rmse = 0
    mae = 0

    for _, row in ratings_test_set.iterrows():
        predicted_rating = model.predict_rating(row["userId"], row["movieId"])
        actual_rating = row["rating"]

        if predicted_rating is None:
            continue

        rmse += (predicted_rating - actual_rating) ** 2
        mae += abs(predicted_rating - actual_rating)
        ratings_test_set_predictions.append(predicted_rating)

    n_valid_predictions = len(ratings_test_set_predictions)
    if not n_valid_predictions:
        return {"rmse": np.nan, "mae": np.nan, "n_valid": 0}

    rmse = (rmse / n_valid_predictions) ** 0.5
    mae = mae / n_valid_predictions
    return {"rmse": rmse, "mae": mae, "n_valid": n_valid_predictions}


def ndcg_at_k(actual_ratings, predicted_ratings, k=10):
    """Calculate NDCG@k using 2**rating - 1 as graded relevance."""
    actual_ratings = np.asarray(actual_ratings, dtype=float)
    predicted_ratings = np.asarray(predicted_ratings, dtype=float)

    if len(actual_ratings) == 0:
        return np.nan

    gains = (2**actual_ratings) - 1
    discounts = np.log2(np.arange(2, len(actual_ratings) + 2))
    ranked_indices = np.argsort(predicted_ratings)[::-1][:k]
    ideal_indices = np.argsort(gains)[::-1][:k]
    dcg = np.sum(gains[ranked_indices] / discounts[: len(ranked_indices)])
    idcg = np.sum(gains[ideal_indices] / discounts[: len(ideal_indices)])
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(model, ratings_test_set, k=10):
    """Calculate regression metrics and mean per-user NDCG@k."""
    regression_metrics = evaluate_regression(model, ratings_test_set)
    user_ndcg = []

    for _, group in ratings_test_set.groupby("userId"):
        valid_actual = []
        valid_predictions = []

        for row in group.itertuples(index=False):
            prediction = model.predict_rating(row.userId, row.movieId)
            if prediction is not None:
                valid_actual.append(row.rating)
                valid_predictions.append(prediction)

        if valid_actual:
            user_ndcg.append(ndcg_at_k(valid_actual, valid_predictions, k))

    mean_ndcg = np.mean(user_ndcg) if user_ndcg else np.nan
    return {
        **regression_metrics,
        "ndcg_at_k": mean_ndcg,
        "k": k,
        "n_users_with_ndcg": len(user_ndcg),
    }
