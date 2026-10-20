from src.core.dataset import Dataset
from src.core.entities import User
from src.models.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Mô hình gợi ý dựa trên độ phổ biến của phim (Popularity base)[cite: 10, 11]."""

    def __init__(self):
        super().__init__()
        self._popular_items: list[tuple[str, float]] = []

    def fit(self, dataset: Dataset) -> None:
        """Tính toán điểm phổ biến cho từng phim dựa trên interactions[cite: 10, 11]."""
        interactions = dataset.get_interactions()
        movies = dataset.get_all_movies()

        item_stats: dict[str, list[float]] = {m.item_id: [] for m in movies}
        for inter in interactions:
            if inter.item_id in item_stats:
                item_stats[inter.item_id].append(inter.rating)

        pop_list = []
        for item_id, ratings in item_stats.items():
            if ratings:
                avg_score = sum(ratings) / len(ratings)
                score = avg_score * (len(ratings) ** 0.5)
            else:
                score = 0.0
            pop_list.append((item_id, float(score)))

        pop_list.sort(key=lambda x: x[1], reverse=True)
        self._popular_items = pop_list
        self._is_fitted = True

    def recommend(self, user: User, k: int = 10) -> list[tuple[str, float]]:
        """Gợi ý các phim phổ biến mà người dùng chưa xem[cite: 10, 11]."""
        self._check_is_fitted()
        self._validate_k(k)

        watched = set(user.ratings.keys())
        results = [(iid, score) for iid, score in self._popular_items if iid not in watched]
        return results[:k]