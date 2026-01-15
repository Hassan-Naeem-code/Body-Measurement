"""
Model Manager - Singleton Pattern for ML Model Caching

This module provides efficient model loading and caching to avoid
reloading heavy ML models on each request. Models are loaded once
and reused across all requests.
"""

import logging
from typing import Optional, Dict, Any
from threading import Lock
import time

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton manager for all ML models used in the application.

    Benefits:
    - Models are loaded once and cached
    - Thread-safe access
    - Lazy loading (models loaded only when needed)
    - Memory-efficient (shared instances)
    """

    _instance = None
    _lock = Lock()

    # Model instances
    _person_detector = None
    _pose_detector = None
    _demographic_detector = None
    _circumference_extractor = None
    _size_recommender = None
    _processor = None

    # Loading timestamps for monitoring
    _load_times: Dict[str, float] = {}

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def person_detector(self):
        """Get or create cached PersonDetector"""
        if self._person_detector is None:
            with self._lock:
                if self._person_detector is None:
                    start = time.time()
                    from app.ml.person_detector import PersonDetector
                    self._person_detector = PersonDetector()
                    self._load_times['person_detector'] = time.time() - start
                    logger.info(f"PersonDetector loaded in {self._load_times['person_detector']:.2f}s")
        return self._person_detector

    @property
    def pose_detector(self):
        """Get or create cached PoseDetector"""
        if self._pose_detector is None:
            with self._lock:
                if self._pose_detector is None:
                    start = time.time()
                    from app.ml.pose_detector import PoseDetector
                    self._pose_detector = PoseDetector()
                    self._load_times['pose_detector'] = time.time() - start
                    logger.info(f"PoseDetector loaded in {self._load_times['pose_detector']:.2f}s")
        return self._pose_detector

    @property
    def demographic_detector(self):
        """Get or create cached DemographicDetector"""
        if self._demographic_detector is None:
            with self._lock:
                if self._demographic_detector is None:
                    start = time.time()
                    from app.ml.demographic_detector import DemographicDetector
                    self._demographic_detector = DemographicDetector()
                    self._load_times['demographic_detector'] = time.time() - start
                    logger.info(f"DemographicDetector loaded in {self._load_times['demographic_detector']:.2f}s")
        return self._demographic_detector

    @property
    def circumference_extractor(self):
        """Get or create cached CircumferenceExtractor with ML ratios"""
        if self._circumference_extractor is None:
            with self._lock:
                if self._circumference_extractor is None:
                    start = time.time()
                    from app.ml.circumference_extractor_ml import MLCircumferenceExtractor
                    self._circumference_extractor = MLCircumferenceExtractor(use_ml_ratios=True)
                    self._load_times['circumference_extractor'] = time.time() - start
                    logger.info(f"CircumferenceExtractor loaded in {self._load_times['circumference_extractor']:.2f}s")
        return self._circumference_extractor

    @property
    def size_recommender(self):
        """Get or create cached SizeRecommender"""
        if self._size_recommender is None:
            with self._lock:
                if self._size_recommender is None:
                    start = time.time()
                    from app.ml.size_recommender_v2 import EnhancedSizeRecommender
                    self._size_recommender = EnhancedSizeRecommender()
                    self._load_times['size_recommender'] = time.time() - start
                    logger.info(f"SizeRecommender loaded in {self._load_times['size_recommender']:.2f}s")
        return self._size_recommender

    @property
    def processor(self):
        """Get or create cached MultiPersonProcessor"""
        if self._processor is None:
            with self._lock:
                if self._processor is None:
                    start = time.time()
                    from app.ml.multi_person_processor_v3 import DepthBasedMultiPersonProcessor
                    self._processor = DepthBasedMultiPersonProcessor(use_ml_ratios=True)
                    self._load_times['processor'] = time.time() - start
                    logger.info(f"MultiPersonProcessor loaded in {self._load_times['processor']:.2f}s")
        return self._processor

    def preload_models(self):
        """
        Preload all models during application startup.
        Call this in app startup to avoid cold start latency.
        """
        logger.info("Preloading ML models...")
        start = time.time()

        # Access each property to trigger lazy loading
        _ = self.processor  # This loads most sub-models

        total_time = time.time() - start
        logger.info(f"All models preloaded in {total_time:.2f}s")
        return self._load_times

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded models"""
        return {
            "models_loaded": {
                "person_detector": self._person_detector is not None,
                "pose_detector": self._pose_detector is not None,
                "demographic_detector": self._demographic_detector is not None,
                "circumference_extractor": self._circumference_extractor is not None,
                "size_recommender": self._size_recommender is not None,
                "processor": self._processor is not None,
            },
            "load_times": self._load_times,
        }

    def clear_cache(self):
        """Clear all cached models (for testing or memory management)"""
        with self._lock:
            self._person_detector = None
            self._pose_detector = None
            self._demographic_detector = None
            self._circumference_extractor = None
            self._size_recommender = None
            self._processor = None
            self._load_times.clear()
            logger.info("Model cache cleared")


# Global instance
model_manager = ModelManager()


def get_processor():
    """Get the cached multi-person processor"""
    return model_manager.processor


def get_model_stats():
    """Get model loading statistics"""
    return model_manager.get_stats()


def preload_models():
    """Preload all models during startup"""
    return model_manager.preload_models()
