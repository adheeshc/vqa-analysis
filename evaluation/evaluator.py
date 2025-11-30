import json
from pathlib import Path
from typing import Any, Dict

import torch
from tqdm import tqdm


class VQAEvaluator:
    """Evaluation Pipeline for VQA Comparison"""

    def __init__(
        self, models: Dict[str, Any], data_path: str = "data/vqa_samples_330.json"
    ):
        self.models = models
        self.data_path = data_path
        self.results = {model_name: [] for model_name in models}
        self.load_data()

    def load_data(self):
        """Load VQA Samples"""
        with open(self.data_path, "r") as f:
            self.samples = json.load(f)
        print(f"Loaded {len(self.samples)} VQA samples")
        return self.samples

    def evaluate_model(self, model_name: str):
        model = self.models[model_name]
        for sample in tqdm(self.samples, desc=f"{model.model_name} predictions"):
            result = self.evaluate_sample(model, sample)
            self.results[model_name].append(result)

        self.save_predictions(model_name)

        model.cleanup()
        torch.cuda.empty_cache()

    def evaluate_all(self):
        for model_name in self.models:
            print("\n" + "=" * 60)
            print(f"Evaluating {model_name}")
            print("=" * 60)
            self.models[model_name].load_model()
            self.evaluate_model(model_name)
        print("\nEvaluation complete for all models")

    def evaluate_sample(self, model, sample: Dict):
        """Evaluate single sample"""
        image_id = sample['image_id']
        image_path = f"data/images/COCO_val2014_{image_id:012d}.jpg"
        question = sample["question"]
        ground_truth_answers = [ans["answer"] for ans in sample["answers"]]

        try:
            prediction_data = model.predict_with_metrics(image_path, question)

            result = {
                "question_id": sample["question_id"],
                "image_id": sample["image_id"],
                "question": question,
                "predicted_answer": prediction_data["answer"],
                "ground_truth": ground_truth_answers,
                "latency_ms": prediction_data["latency_ms"],
                "memory_mb": prediction_data["memory_mb"],
                "question_type": sample.get("question_type", "other"),
                "answer_type": sample.get("answer_type", "other"),
            }
        except Exception as e:
            print(f"Error for sample {sample['question_id']}: {e}")
            result = {
                "question_id": sample["question_id"],
                "image_id": sample["image_id"],
                "question": question,
                "predicted_answer": "ERROR",
                "ground_truth": ground_truth_answers,
                "latency_ms": 0,
                "memory_mb": 0,
                "question_type": sample.get("question_type", "other"),
                "answer_type": sample.get("answer_type", "other"),
                "error": str(e),
            }

        return result

    def save_predictions(self, model_name: str):
        """Save predictions to JSON"""
        output_dir = Path("results/predictions")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = (
            output_dir / f"{model_name.lower().replace(' ', '_')}_predictions.json"
        )

        with open(output_path, "w") as f:
            json.dump(self.results[model_name], f, indent=4)

        print(f"Saved predictions to {output_path}")
