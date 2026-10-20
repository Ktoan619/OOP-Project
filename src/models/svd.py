import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from src.core.dataset import Dataset
from src.core.entities import User
from src.core.exceptions import InvalidParameterError, ModelNotTrainedError
from src.models.base import BaseRecommender


class SVDRecommender(BaseRecommender):
    """Mô hình SVD làm chính, kết hợp Popularity làm cơ sở dự phòng (base)[cite: 1]."""

    def __init__(self, n_factors: int = 20):
        super().__init__()
        self.n_factors = n_factors
        self._user_id_to_idx: dict[str, int] = {}
        self._idx_to_user_id: dict[int, str] = {}
        self._item_id_to_idx: dict[str, int] = {}
        self._idx_to_item_id: dict[int, str] = {}
        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None
        self._popular_items: list[tuple[str, float]] = []

    def fit(self, dataset: Dataset) -> None:
        all_movies = dataset.get_all_movies()
        all_users = dataset.get_all_users()
        interactions = dataset.get_interactions()

        if not interactions:
            raise InvalidParameterError("Dataset contains no interactions for training SVD.")

        for idx, movie in enumerate(all_movies):
            self._item_id_to_idx[movie.item_id] = idx
            self._idx_to_item_id[idx] = movie.item_id

        for idx, user in enumerate(all_users):
            self._user_id_to_idx[user.user_id] = idx
            self._idx_to_user_id[idx] = user.user_id

        num_users = len(self._user_id_to_idx)
        num_items = len(self._item_id_to_idx)

        # Tính toán Popularity Base Scores làm cơ sở dự phòng
        item_scores: dict[str, list[float]] = {m.item_id: [] for m in all_movies}
        for inter in interactions:
            if inter.item_id in item_scores:
                item_scores[inter.item_id].append(inter.rating)

        pop_list = []
        for iid, ratings in item_scores.items():
            avg = sum(ratings) / len(ratings) if ratings else 0.0
            score = avg * (len(ratings) ** 0.5)
            pop_list.append((iid, float(score)))
        pop_list.sort(key=lambda x: x[1], reverse=True)
        self._popular_items = pop_list

        # Dựng ma trận thưa User-Item
        rows, cols, data = [], [], []
        for inter in interactions:
            if inter.user_id in self._user_id_to_idx and inter.item_id in self._item_id_to_idx:
                rows.append(self._user_id_to_idx[inter.user_id])
                cols.append(self._item_id_to_idx[inter.item_id])
                data.append(inter.rating)

        R_sparse = csr_matrix((data, (rows, cols)), shape=(num_users, num_items), dtype=np.float32)
        k_factors = min(self.n_factors, min(num_users, num_items) - 1)
        if k_factors <= 0:
            k_factors = 1

        u, s, vt = svds(R_sparse, k=k_factors)
        idx_sort = np.argsort(s)[::-1]
        u = u[:, idx_sort]
        s = s[idx_sort]
        vt = vt[idx_sort, :]

        self._user_factors = u @ np.diag(s)
        self._item_factors = vt
        self._is_fitted = True

    def recommend(self, user: User, k: int = 10) -> list[tuple[str, float]]:
        self._check_is_fitted()
        self._validate_k(k)

        watched = set(user.ratings.keys())
        scores: dict[str, float] = {}

        if user.user_id in self._user_id_to_idx:
            u_idx = self._user_id_to_idx[user.user_id]
            pred = self._user_factors[u_idx] @ self._item_factors
            for i_idx, score in enumerate(pred):
                iid = self._idx_to_item_id[i_idx]
                if iid not in watched:
                    scores[iid] = float(score)
        else:
            # Fallback/Base sang Popularity nếu user chưa có trong ma trận SVD
            for iid, score in self._popular_items:
                if iid not in watched:
                    scores[iid] = score

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]