from abc import ABC, abstractmethod
from src.core.dataset import Dataset
from src.core.entities import User
from src.core.exceptions import InvalidParameterError, ModelNotTrainedError


class BaseRecommender(ABC):
    """Lớp cơ sở trừu tượng cho mọi mô hình gợi ý trong hệ thống[cite: 9, 10]."""

    def __init__(self):
        self._is_fitted: bool = False

    def _check_is_fitted(self):
        """Ném ModelNotTrainedError nếu model chưa được huấn luyện[cite: 9, 10]."""
        if not self._is_fitted:
            raise ModelNotTrainedError("Model has not been trained. Call fit() first[cite: 9, 10].")

    def _validate_k(self, k: int):
        """Ném InvalidParameterError nếu k <= 0[cite: 9, 10]."""
        if k <= 0:
            raise InvalidParameterError(f"k must be a positive integer, got {k}[cite: 9, 10]")

    @abstractmethod
    def fit(self, dataset: Dataset) -> None:
        """Nạp dữ liệu và huấn luyện mô hình[cite: 10]."""
        pass

    @abstractmethod
    def recommend(self, user: User, k: int = 10) -> list[tuple[str, float]]:
        """Trả về Top-K cặp (item_id, score) sắp xếp giảm dần theo score[cite: 10]."""
        pass