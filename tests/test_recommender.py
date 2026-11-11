import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from src.data import dataset_statistics, temporal_train_test_split
from src.evaluation import evaluate_regression, ndcg_at_k
from src.model import BiasedMatrixFactorization
from src.recommendation import get_user_history, recommend_movies


class RecommenderTests(unittest.TestCase):
    def setUp(self):
        self.ratings = pd.DataFrame(
            [
                (1, 10, 4.0, 1),
                (1, 20, 3.0, 2),
                (1, 30, 5.0, 3),
                (2, 10, 2.0, 1),
                (2, 30, 4.0, 2),
            ],
            columns=["userId", "movieId", "rating", "timestamp"],
        )
        self.movies = pd.DataFrame(
            [
                (10, "A", "Drama"),
                (20, "B", "Comedy"),
                (30, "C", "Drama"),
            ],
            columns=["movieId", "title", "genres"],
        )

    def test_temporal_split_keeps_each_users_earliest_ratings_in_train(self):
        train, test = temporal_train_test_split(self.ratings)
        self.assertEqual(len(train), 3)
        self.assertEqual(len(test), 2)
        for user_id in self.ratings["userId"].unique():
            user_train = train.loc[train.userId == user_id]
            user_test = test.loc[test.userId == user_id]
            self.assertGreaterEqual(len(user_train), 1)
            if not user_test.empty:
                self.assertLessEqual(
                    user_train.timestamp.max(), user_test.timestamp.min()
                )

    def test_model_training_matches_original_sgd_updates(self):
        train = self.ratings.copy()
        seed = 41

        np.random.seed(seed)
        unique_users = train["userId"].unique()
        unique_movies = train["movieId"].unique()
        user2idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        movie2idx = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
        mu = train["rating"].mean()
        b_u = np.zeros(len(unique_users))
        b_i = np.zeros(len(unique_movies))
        P = np.random.normal(0, 0.1, (len(unique_users), 3))
        Q = np.random.normal(0, 0.1, (len(unique_movies), 3))
        users_arr = train.userId.map(user2idx).to_numpy()
        movies_arr = train.movieId.map(movie2idx).to_numpy()
        ratings_arr = train.rating.to_numpy()
        reference_mse = []
        lr = 0.005
        lambda_ = 0.02

        for _ in range(2):
            epoch_loss = 0
            for idx in np.random.permutation(len(ratings_arr)):
                u = users_arr[idx]
                i = movies_arr[idx]
                error = ratings_arr[idx] - (
                    mu + b_u[u] + b_i[i] + np.dot(P[u], Q[i])
                )
                P_u_old = P[u].copy()
                b_u[u] += lr * (error - lambda_ * b_u[u])
                b_i[i] += lr * (error - lambda_ * b_i[i])
                P[u] += lr * (error * Q[i] - lambda_ * P_u_old)
                Q[i] += lr * (error * P_u_old - lambda_ * Q[i])
                epoch_loss += error**2
            reference_mse.append(epoch_loss / len(ratings_arr))

        np.random.seed(seed)
        model = BiasedMatrixFactorization(k=3, epochs=2)
        with contextlib.redirect_stdout(io.StringIO()):
            actual_mse = model.fit(train)

        np.testing.assert_allclose(actual_mse, reference_mse, rtol=0, atol=1e-14)
        np.testing.assert_allclose(model.P, P, rtol=0, atol=1e-14)
        np.testing.assert_allclose(model.Q, Q, rtol=0, atol=1e-14)
        np.testing.assert_allclose(model.b_u, b_u, rtol=0, atol=1e-14)
        np.testing.assert_allclose(model.b_i, b_i, rtol=0, atol=1e-14)

    def test_prediction_cold_start_and_recommendation(self):
        model = BiasedMatrixFactorization(k=2, epochs=1)
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(self.ratings)
        self.assertIsNone(model.predict_rating(999, 10))
        self.assertIsNone(model.predict_rating(1, 999))
        self.assertTrue(0.5 <= model.predict_rating(1, 30) <= 5.0)
        recommendations = recommend_movies(
            model, 2, self.movies, self.ratings, n=10
        )
        self.assertEqual(recommendations.movieId.tolist(), [20])
        self.assertEqual(
            list(recommend_movies(model, 999, self.movies, self.ratings).columns),
            ["movieId", "title", "genres", "predicted_rating"],
        )
        self.assertTrue(
            recommend_movies(model, 1, self.movies, self.ratings).empty
        )
        self.assertTrue(
            get_user_history(1, self.ratings, self.movies).rating.is_monotonic_decreasing
        )

    def test_metrics_and_statistics(self):
        stats = dataset_statistics(self.ratings, self.movies)
        self.assertEqual(stats["n_users"], 2)
        self.assertEqual(stats["n_movies"], 3)
        self.assertEqual(stats["n_ratings"], 5)
        self.assertAlmostEqual(stats["density"], 5 / 6)
        self.assertEqual(ndcg_at_k([5.0, 1.0], [5.0, 1.0], 2), 1.0)

        model = BiasedMatrixFactorization(k=2, epochs=1)
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(self.ratings)
        metrics = evaluate_regression(model, self.ratings.iloc[3:])
        self.assertEqual(metrics["n_valid"], 2)
        self.assertGreaterEqual(metrics["rmse"], 0)
        self.assertGreaterEqual(metrics["mae"], 0)
        no_predictions = evaluate_regression(
            BiasedMatrixFactorization(), self.ratings
        )
        self.assertTrue(np.isnan(no_predictions["rmse"]))
        self.assertEqual(no_predictions["n_valid"], 0)


if __name__ == "__main__":
    unittest.main()
