"""Streaming datasets: labels are cached once, images are fetched while iterating."""

from ._streaming import CACHE_DIR, TARGET_KEYS, Interleave, StreamingFoodDataset, collate_samples
from .local import LocalFoodDataset, local_available
from .mm_food_100k import MMFood100K
from .nutrion5k import Nutrition5k, Nutrition5kSideAngles
from .nutrition_verse_real import NutritionVerseReal

__all__ = ["CACHE_DIR", "TARGET_KEYS", "Interleave", "StreamingFoodDataset", "collate_samples",
           "LocalFoodDataset", "local_available", "MMFood100K", "Nutrition5k", "Nutrition5kSideAngles", "NutritionVerseReal"]
