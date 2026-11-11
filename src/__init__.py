"""MovieLens matrix-factorization recommender."""

from .data import dataset_statistics, load_data, temporal_train_test_split
from .evaluation import evaluate_model, evaluate_regression, ndcg_at_k
from .model import BiasedMatrixFactorization
from .recommendation import get_user_history, recommend_movies

__all__ = [
    "BiasedMatrixFactorization",
    "dataset_statistics",
    "evaluate_model",
    "evaluate_regression",
    "get_user_history",
    "load_data",
    "ndcg_at_k",
    "recommend_movies",
    "temporal_train_test_split",
]
