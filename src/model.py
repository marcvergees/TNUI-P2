"""Biased matrix-factorization model trained with stochastic gradient descent."""

import numpy as np


class BiasedMatrixFactorization:
    """Matrix factorization with user and item biases."""

    def __init__(self, k=20, lr=0.005, lambda_=0.02, epochs=20):
        self.k = k
        self.lr = lr
        self.lambda_ = lambda_
        self.epochs = epochs
        self.user2idx = {}
        self.movie2idx = {}
        self.user_ids = None
        self.movie_ids = None
        self.mu = None
        self.b_u = None
        self.b_i = None
        self.P = None
        self.Q = None
        self.train_mse_list = []

    def fit(self, ratings_train_set):
        """Fit model parameters and return the training MSE for each epoch."""
        self.user_ids = ratings_train_set["userId"].unique()
        self.movie_ids = ratings_train_set["movieId"].unique()
        self.user2idx = {
            user_id: idx for idx, user_id in enumerate(self.user_ids)
        }
        self.movie2idx = {
            movie_id: idx for idx, movie_id in enumerate(self.movie_ids)
        }

        n_users = len(self.user_ids)
        n_movies = len(self.movie_ids)
        self.mu = ratings_train_set["rating"].mean()
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_movies)
        self.P = np.random.normal(0, 0.1, (n_users, self.k))
        self.Q = np.random.normal(0, 0.1, (n_movies, self.k))

        users_arr = ratings_train_set["userId"].map(self.user2idx).to_numpy()
        movies_arr = ratings_train_set["movieId"].map(self.movie2idx).to_numpy()
        ratings_arr = ratings_train_set["rating"].to_numpy()
        n_samples = len(ratings_arr)
        self.train_mse_list = []

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0

            for idx in indices:
                u = users_arr[idx]
                i = movies_arr[idx]
                r_ui = ratings_arr[idx]

                r_ui_barret = (
                    self.mu + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
                )
                e_ui = r_ui - r_ui_barret
                P_u_old = self.P[u].copy()

                self.b_u[u] += self.lr * (e_ui - self.lambda_ * self.b_u[u])
                self.b_i[i] += self.lr * (e_ui - self.lambda_ * self.b_i[i])
                self.P[u] += self.lr * (
                    e_ui * self.Q[i] - self.lambda_ * P_u_old
                )
                self.Q[i] += self.lr * (
                    e_ui * P_u_old - self.lambda_ * self.Q[i]
                )
                epoch_loss += e_ui**2

            train_mse = epoch_loss / n_samples
            self.train_mse_list.append(train_mse)
            print(f"Epoch {epoch + 1}/{self.epochs} - Train MSE: {train_mse:.4f}")

        return self.train_mse_list

    def predict_rating(self, user_id, movie_id):
        """Predict a rating, returning ``None`` for an unknown user or movie."""
        u = self.user2idx.get(user_id, -1)
        i = self.movie2idx.get(movie_id, -1)

        if u == -1:
            return None
        if i == -1:
            return None

        prediction = self.mu + self.b_u[u] + self.b_i[i] + self.P[u].dot(self.Q[i])

        if prediction < 0.5:
            prediction = 0.5
        elif prediction > 5:
            prediction = 5
        return prediction
