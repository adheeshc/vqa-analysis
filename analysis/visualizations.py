import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class VQAVisualizer:
    """Create comparison visualizations"""

    def __init__(self, metrics_path: str = "results/benchmarks/all_metrics.json"):
        with open(metrics_path, "r") as f:
            self.metrics = json.load(f)
        self.output_dir = Path("results/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 6)

    def plot_accuracy_comparison(self):
        """Bar chart comparing overall accuracy"""
        models = list(self.metrics.keys())
        accuracies = [self.metrics[m]["overall_accuracy"] * 100 for m in models]
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies, color=["#3498db", "#e74c3c", "#2ecc71"])

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.title("VQA Accuracy Comparison", fontsize=14, fontweight="bold")
        plt.ylim(0, max(accuracies) * 1.2)
        plt.tight_layout()
        plt.savefig(self.output_dir / "accuracy_comparison.png", dpi=300)
        plt.close()

        print("Saved accuracy comparison")

    def plot_latency_comparison(self):
        """Bar chart comparing mean latency"""
        models = list(self.metrics.keys())
        latencies = [self.metrics[m]["performance"]["mean_latency_ms"] for m in models]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, latencies, color=["#3498db", "#e74c3c", "#2ecc71"])

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.0f}ms",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        plt.ylabel("Mean Latency (ms)", fontsize=12)
        plt.title("Inference Latency Comparison", fontsize=14, fontweight="bold")
        plt.ylim(0, max(latencies) * 1.2)

        plt.tight_layout()
        plt.savefig(self.output_dir / "latency_comparison.png", dpi=300)
        plt.close()

        print("Saved latency comparison")

    def plot_memory_comparison(self):
        """Bar chart comparing peak memory"""
        models = list(self.metrics.keys())
        memories = [self.metrics[m]["performance"]["peak_memory_mb"] for m in models]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, memories, color=["#3498db", "#e74c3c", "#2ecc71"])

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.0f}MB",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        plt.ylabel("Peak Memory (MB)", fontsize=12)
        plt.title("Memory Usage Comparison", fontsize=14, fontweight="bold")
        plt.ylim(0, max(memories) * 1.2)

        plt.tight_layout()
        plt.savefig(self.output_dir / "memory_comparison.png", dpi=300)
        plt.close()

        print("Saved memory comparison")

    def plot_accuracy_by_question_type(self):
        """Grouped bar chart: accuracy by question type for each model"""
        question_types = set()
        for model_metrics in self.metrics.values():
            question_types.update(model_metrics["by_question_type"].keys())

        question_types = sorted(question_types)
        data = []
        for model_name, model_metrics in self.metrics.items():
            for q_type in question_types:
                if q_type in model_metrics["by_question_type"]:
                    acc = model_metrics["by_question_type"][q_type]["accuracy"] * 100
                    data.append(
                        {"Model": model_name, "Question Type": q_type, "Accuracy": acc}
                    )

        df = pd.DataFrame(data)

        plt.figure(figsize=(14, 6))
        ax = sns.barplot(data=df, x="Question Type", y="Accuracy", hue="Model")

        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.xlabel("Question Type", fontsize=12)
        plt.title("Accuracy by Question Type", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Model", loc="upper right")

        plt.tight_layout()
        plt.savefig(self.output_dir / "accuracy_by_question_type.png", dpi=300)
        plt.close()

        print("Saved accuracy by question type")

    def plot_performance_tradeoff(self):
        """Scatter plot: accuracy vs latency"""
        models = list(self.metrics.keys())
        accuracies = [self.metrics[m]["overall_accuracy"] * 100 for m in models]
        latencies = [self.metrics[m]["performance"]["mean_latency_ms"] for m in models]

        plt.figure(figsize=(10, 6))

        colors = ["#3498db", "#e74c3c", "#2ecc71"]
        for i, model in enumerate(models):
            plt.scatter(
                latencies[i],
                accuracies[i],
                s=300,
                c=colors[i],
                label=model,
                alpha=0.7,
                edgecolors="black",
                linewidth=2,
            )
            plt.text(
                latencies[i],
                accuracies[i],
                model,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        plt.xlabel("Mean Latency (ms)", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.title("Accuracy vs Latency Trade-off", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "accuracy_latency_tradeoff.png", dpi=300)
        plt.close()

        print("Saved accuracy vs latency plot")

    def generate_all_plots(self):
        """Generate all visualizations"""
        print("\nGenerating visualizations...")

        self.plot_accuracy_comparison()
        self.plot_latency_comparison()
        self.plot_memory_comparison()
        self.plot_accuracy_by_question_type()
        self.plot_performance_tradeoff()

        print(f"\nAll visualizations saved to {self.output_dir}")


if __name__ == "__main__":
    visualizer = VQAVisualizer()
    visualizer.generate_all_plots()
