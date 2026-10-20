import numpy as np
from src.core.dataset import Dataset
from src.core.entities import User
from src.models.base import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    """Mô hình gợi ý dựa trên nội dung thể loại phim[cite: 11, 12]."""

    def __init__(self):
        super().__init__()
        self._genre_list: list[str] = []
        self._genre_index: dict[str, int] = {}
        self._movie_vectors: dict[str, np.ndarray] = {}

    def fit(self, dataset: Dataset) -> None:
        movies = dataset.get_all_movies()
        genres_set = set()
        for m in movies:
            for g in m.genres:
                genres_set.add(g)

        self._genre_list = sorted(list(genres_set))
        self._genre_index = {g: idx for idx, g in enumerate(self._genre_list)}

        for m in movies:
            vec = np.zeros(len(self._genre_list))
            for g in m.genres:
                if g in self._genre_index:
                    vec[self._genre_index[g]] = 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self._movie_vectors[m.item_id] = vec

        self._is_fitted = True

    def recommend(self, user: User, k: int = 10) -> list[tuple[str, float]]:
        self._check_is_fitted()
        self._validate_k(k)

        user_vector = np.zeros(len(self._genre_list))
        if user.preferences:
            for genre in user.preferences:
                if genre in self._genre_index:
                    user_vector[self._genre_index[genre]] = 1.0

        for item_id, rating in user.ratings.items():
            if item_id in self._movie_vectors:
                user_vector += self._movie_vectors[item_id] * rating

        norm = np.linalg.norm(user_vector)
        if norm > 0:
            user_vector = user_vector / norm

        watched = set(user.ratings.keys())
        scores = {}
        for item_id, vec in self._movie_vectors.items():
            if item_id not in watched:
                sim = float(np.dot(user_vector, vec))
                scores[item_id] = sim

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]