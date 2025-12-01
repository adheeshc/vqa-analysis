import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

from evaluation.metrics import VQAMetrics


class FailureAnalyzer:
    """Analyze failure patterns in VQA predictions"""

    def __init__(self, predictions_dir: str = "results/predictions"):
        self.predictions_dir = Path(predictions_dir)
        self.models = {}
        self.load_all_predictions()

    def load_all_predictions(self):
        """Load predictions from all models"""
        for pred_file in self.predictions_dir.glob("*_predictions.json"):
            model_name = pred_file.stem.replace("_predictions", "").upper()

            with open(pred_file, "r") as f:
                self.models[model_name] = json.load(f)

        print(f"Loaded predictions for: {', '.join(self.models.keys())}")

    def find_failure_cases(self, model_name: str, threshold: float = 0.5) -> List[Dict]:
        """Find cases where model accuracy is below threshold"""
        failures = []
        for pred in self.models[model_name]:
            if pred["predicted_answer"] == "ERROR":  # skip error
                failures.append({**pred, "accuracy": 0.0, "failure_type": "error"})
                continue

            acc = VQAMetrics.compute_accuracy(
                pred["predicted_answer"], pred["ground_truth"]
            )
            if acc < threshold:
                failures.append({**pred, "accuracy": acc, "failure_type": "incorrect"})

        return failures

    def analyze_failure_patterns(self, model_name: str) -> Dict:
        """Analyze common failure patterns"""
        failures = self.find_failure_cases(model_name, threshold=0.5)

        # question type
        by_q_type = defaultdict(int)
        for f in failures:
            by_q_type[f["question_type"]] += 1

        # answer type
        by_a_type = defaultdict(int)
        for f in failures:
            by_a_type[f["answer_type"]] += 1

        # mistake patterns
        mistake_patterns = []
        for f in failures:
            pattern = {
                "question": f["question"],
                "predicted": f["predicted_answer"],
                "expected": f["ground_truth"][0] if f["ground_truth"] else "N/A",
                "accuracy": f["accuracy"],
            }
            mistake_patterns.append(pattern)

        mistake_patterns.sort(key=lambda x: x["accuracy"])
        analysis = {
            "total_failures": len(failures),
            "failure_rate": len(failures) / len(self.models[model_name]),
            "by_question_type": dict(by_q_type),
            "by_answer_type": dict(by_a_type),
            "worst_cases": mistake_patterns[:20],  # Top 20 worst
        }
        return analysis

    def compare_models_on_failures(self) -> pd.DataFrame:
        """Compare how different models perform on same questions"""
        question_ids = set()
        for model_name in self.models:
            failures = self.find_failure_cases(model_name)
            question_ids.update(f["question_id"] for f in failures)

        comparisons = []
        for qid in question_ids:
            comparison = {"question_id": qid}
            for model_name, predictions in self.models.items():
                pred = next((p for p in predictions if p["question_id"] == qid), None)
                if pred:
                    acc = VQAMetrics.compute_accuracy(
                        pred["predicted_answer"], pred["ground_truth"]
                    )
                    comparison[f"{model_name}_answer"] = pred["predicted_answer"]
                    comparison[f"{model_name}_acc"] = acc
                    comparison["question"] = pred["question"]
                    comparison["ground_truth"] = pred["ground_truth"][0]
            comparisons.append(comparison)
        df = pd.DataFrame(comparisons)
        return df

    def generate_failure_report(self, output_path: str = "results/failure_analysis.md"):
        """Generate markdown report of failure analysis"""

        report = ["# VQA Failure Analysis\n\n"]
        for model_name in self.models.keys():
            report.append(f"## {model_name}\n\n")
            analysis = self.analyze_failure_patterns(model_name)
            
            report.append(f"**Failure Rate**: {analysis['failure_rate']:.1%}\n\n")
            report.append(f"**Total Failures**: {analysis['total_failures']}\n\n")
            report.append("### Failures by Question Type\n\n")
            
            for q_type, count in sorted(
                analysis["by_question_type"].items(), key=lambda x: x[1], reverse=True
            ):
                report.append(f"- {q_type}: {count}\n")
            
            report.append("\n")
            report.append("### Worst 10 Cases\n\n")
            report.append("| Question | Predicted | Expected | Accuracy |\n")
            report.append("|----------|-----------|----------|----------|\n")
            
            for case in analysis["worst_cases"][:10]:
                report.append(
                    f"| {case['question'][:50]} | {case['predicted']} | "
                    f"{case['expected']} | {case['accuracy']:.2f} |\n"
                )
            report.append("\n---\n\n")
        
        with open(output_path, "w") as f:
            f.writelines(report)
        print(f"Failure analysis report saved to {output_path}")
        return "".join(report)


if __name__ == "__main__":
    analyzer = FailureAnalyzer()
    analyzer.generate_failure_report()
