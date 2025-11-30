#!/usr/bin/env python3
"""
Main script to run VQA comparison evaluation

Usage:
    python main.py --models clip blip2 llava
    python main.py --models all
"""

import argparse
import json
from pathlib import Path

from evaluation.evaluator import VQAEvaluator
from evaluation.metrics import VQAMetrics
from models.blip2_wrapper import BLIP2VQAModel
from models.clip_wrapper import CLIPVQAModel
from models.llava_wrapper import LLaVAVQAModel
from models.myModel_wrapper import myModelVQAModel


def parse_args():
    parser = argparse.ArgumentParser(description="VQA Model Comparison")
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        choices=["clip", "blip2", "llava", "vlm_chatbot", "all"],
        default=["all"],
        help="Models to evaluate",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/vqa_samples_330.json",
        help="Path to VQA samples",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    models = {}

    if "all" in args.models:
        model_choices = ["clip", "blip2", "llava", "vlm_chatbot"]
    else:
        model_choices = args.models

    print(f"Evaluating models: {', '.join(model_choices)}")

    if "clip" in model_choices:
        models["CLIP"] = CLIPVQAModel()

    if "blip2" in model_choices:
        models["BLIP2"] = BLIP2VQAModel()

    if "llava" in model_choices:
        models["LLaVA"] = LLaVAVQAModel()

    if "vlm_chatbot" in model_choices:
        models["VLM_Chatbot"] = myModelVQAModel()

    # Run evaluation
    evaluator = VQAEvaluator(models, data_path=args.data)
    evaluator.evaluate_all()

    # Compute Metrics
    results_dir = Path("results/benchmarks")
    results_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    for model_name in evaluator.results.keys():
        print("\n" + "=" * 60)
        print(f"Computing metrics for {model_name}")
        print("=" * 60)

        predictions = evaluator.results[model_name]
        metrics = VQAMetrics.compute_metrics(predictions)

        all_metrics[model_name] = metrics

        print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.2%}")
        print(f"Mean Latency: {metrics['performance']['mean_latency_ms']:.1f}ms")
        print(f"Peak Memory: {metrics['performance']['peak_memory_mb']:.1f}MB")

        print("\nAccuracy by Question Type:")
        for q_type, q_metrics in metrics["by_question_type"].items():
            print(
                f"  {q_type}: {q_metrics['accuracy']:.2%} ({q_metrics['count']} samples)"
            )

    # Save all metrics
    metrics_path = results_dir / "all_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=float)

    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
