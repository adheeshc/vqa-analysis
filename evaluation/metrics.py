import re
from collections import defaultdict
from typing import Dict, List

import numpy as np


class VQAMetrics:
    """Compute VQA accuracy metrics"""

    @staticmethod
    def normalize_answer(answer: str) -> str:
        """Normalize answer for comparison"""
        answer = answer.lower().strip()
        answer = re.sub(r"\b(a|an|the)\b", "", answer)
        answer = re.sub(r"[^\w\s]", "", answer)
        answer = " ".join(answer.split())
        return answer

    @staticmethod
    def compute_accuracy(predicted: str, ground_truth: List[str]) -> float:
        """Compute VQA accuracy"""
        predicted_norm = VQAMetrics.normalize_answer(predicted)
        matches = sum(
            1
            for gt in ground_truth
            if VQAMetrics.normalize_answer(gt) == predicted_norm
        )
        accuracy = min(matches / 3.0, 1.0)  # VQA accuracy formula
        return accuracy

    @staticmethod
    def compute_metrics(predictions: List[Dict]) -> Dict:
        """Compute overall metrics from predictions"""
        accuracies = []
        by_question_type = defaultdict(list)
        by_answer_type = defaultdict(list)
        latencies = []
        memories = []

        for pred in predictions:
            if pred["predicted_answer"] == "ERROR":
                continue

            # accuracy metrics
            acc = VQAMetrics.compute_accuracy(
                pred["predicted_answer"], pred["ground_truth"]
            )
            accuracies.append(acc)

            q_type = pred.get("question_type", "other")
            by_question_type[q_type].append(acc)

            a_type = pred.get("answer_type", "other")
            by_answer_type[a_type].append(acc)

            # performance metrics
            latencies.append(pred["latency_ms"])
            memories.append(pred["memory_mb"])

        metrics = {
            "overall_accuracy": np.mean(accuracies) if accuracies else 0.0,
            "accuracy_std": np.std(accuracies) if accuracies else 0.0,
            "num_samples": len(accuracies),
            "by_question_type": {
                k: {"accuracy": np.mean(v), "count": len(v)}
                for k, v in by_question_type.items()
            },
            "by_answer_type": {
                k: {"accuracy": np.mean(v), "count": len(v)}
                for k, v in by_answer_type.items()
            },
            "performance": {
                "mean_latency_ms": np.mean(latencies) if latencies else 0.0,
                "median_latency_ms": np.median(latencies) if latencies else 0.0,
                "std_latency_ms": np.std(latencies) if latencies else 0.0,
                "mean_memory_mb": np.mean(memories) if memories else 0.0,
                "peak_memory_mb": np.max(memories) if memories else 0.0,
            },
        }

        return metrics
