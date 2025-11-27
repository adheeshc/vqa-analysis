import time
from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


class BaseVQAModel(ABC):
    """Abstract base class for VQA models"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def load_model(self):
        """Load the model"""
        pass

    @abstractmethod
    def predict(self, image_path: str, question: str) -> str:
        """
        Generate answer for question about image

        Args:
            image_path: Path to image file
            question: Question

        Returns:
            answer: Predicted answer
        """
        pass

    def predict_with_metrics(self, image_path: str, question: str) -> Dict[str, Any]:
        """
        Predict with timing and memory metrics

        Returns:
            {
                'answer': str,
                'latency_ms': float,
                'memory_mb': float
            }
        """
        # Memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            memory_before = torch.cuda.memory_allocated() / 1024**2

        # Latency
        start_time = time.time()
        answer = self.predict(image_path, question)
        latency = (time.time() - start_time) * 1000

        # Memory
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            memory_used = peak_memory - memory_before
        else:
            memory_used = 0

        return {"answer": answer, "latency_ms": latency, "memory_mb": memory_used}

    @abstractmethod
    def cleanup(self):
        """Clean up model resources"""
        pass
