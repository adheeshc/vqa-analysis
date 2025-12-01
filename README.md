# Vision-Language Model Evaluation Framework

A production-grade benchmarking system for evaluating state-of-the-art Vision-Language Models (VLMs) on Visual Question Answering tasks. Built with a focus on reproducibility, scalability, and memory efficiency.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Project Overview

This framework provides a robust, extensible pipeline for systematic evaluation of Vision-Language Models against the VQA v2.0 benchmark. Key features include:

- **Multi-Model Support**: Unified interface for CLIP, BLIP-2, LLaVA, and custom trained models
- **Production-Ready**: Memory-efficient incremental evaluation with automatic cleanup
- **Comprehensive Metrics**: Accuracy, latency, memory profiling, and granular performance breakdowns
- **Reproducible**: Standardized evaluation protocol with consistent preprocessing and scoring

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vqa-analysis.git
cd vqa-analysis

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Evaluate all models
python main.py --models all

# Evaluate specific models
python main.py -m clip blip2 llava

# Single model evaluation (memory-efficient)
python main.py -m llava

# Custom dataset
python main.py -m all --data path/to/vqa_samples.json
```

## Project Structure

```
vqa-analysis/
├── main.py                  # Main evaluation script
├── data/                    # VQA samples and images
│   ├── vqa_samples_330.json
│   └── images/
├── models/                  # Model wrappers
│   ├── base_model.py
│   ├── clip_wrapper.py
│   ├── blip2_wrapper.py
│   ├── llava_wrapper.py
│   └── myModel_wrapper.py
├── evaluation/              # Evaluation pipeline
│   ├── evaluator.py
│   └── metrics.py
├── analysis/                # Analysis scripts
│   └── failure_analysis.py
└── results/                 # Outputs
    ├── predictions/         # Model predictions (JSON)
    ├── benchmarks/          # Performance metrics (JSON)
    └── visualizations/      # Charts and plots
```

## Supported Models

| Model | Architecture | Parameters | Implementation |
|-------|-------------|------------|----------------|
| **CLIP** | Contrastive Vision-Language Pre-training | 428M | Zero-shot classification via image-text similarity (ViT-L/14) |
| **BLIP-2** | Bootstrapped Language-Image Pre-training | 3.9B | Q-Former architecture with frozen vision encoder and LLM |
| **LLaVA** | Large Language and Vision Assistant | 7B | Instruction-tuned multimodal LLM with visual projection |
| **VLM_Chatbot** | Projection Layer trained, quantized | Variable | For more info visit https://github.com/adheeshc/vlm-chatbot

### Technical Highlights

- **Memory Optimization**: Automatic quantization (4-bit/8-bit), gradient checkpointing, and CPU offloading
- **Distributed Inference**: Support for device mapping across GPU/CPU/disk
- **Batch Processing**: Efficient batched inference with dynamic padding
- **Error Handling**: Graceful degradation with detailed error logging

## Evaluation Metrics

### Core Metrics

| Metric | Description | Implementation |
|--------|-------------|----------------|
| **VQA Accuracy** | Soft accuracy score accounting for inter-annotator agreement | `min(#humans_said_answer/3, 1)` |
| **Latency** | End-to-end inference time per sample | GPU-synchronized timing with warmup |
| **Memory** | Peak VRAM/RAM consumption | Real-time profiling via `torch.cuda.max_memory_allocated()` |
| **Throughput** | Samples processed per second | Computed over full evaluation run |

### Granular Analysis

- **By Question Type**: Performance stratified across 20+ question categories (e.g., "what is", "how many", "why")
- **By Answer Type**: Breakdown by answer modality (yes/no, number, other)
- **Confidence Intervals**: Statistical significance testing with bootstrap resampling
- **Failure Analysis**: Automated categorization of error modes

## Results & Artifacts

### Output Structure

```
results/
├── predictions/                    # Per-sample predictions with metadata
│   ├── clip_predictions.json       
│   ├── blip2_predictions.json      
│   ├── llava_predictions.json      
│   └── vlm_chatbot_predictions.json
├── benchmarks/
│   └── all_metrics.json            # Aggregated performance metrics
└── visualizations/
    ├── accuracy_comparison.png     # Model comparison charts
    ├── latency_distribution.png
    └── confusion_matrix.png
```

### Incremental Evaluation Architecture

Designed for resource-constrained environments with intelligent checkpointing:

```bash
# Sequential evaluation with automatic state management
python main.py -m clip      # → Saves checkpoint
python main.py -m blip2     # → Appends to checkpoint
python main.py -m llava     # → Appends to checkpoint
```

**Key Features**:
- ✅ Atomic writes prevent data corruption
- ✅ Memory freed immediately after each model (GC + CUDA cache clear)
- ✅ Crash-resilient: Resume from last successful evaluation
- ✅ Parallel-safe: Multiple concurrent evaluations supported

## Technical Architecture

### System Design

```
┌─────────────────────────────────────────────────────────┐
│                     Main Orchestrator                    │
│                      (main.py)                          │
└───────────────┬─────────────────────────────────────────┘
                │
    ┌───────────┴───────────┐
    │                       │
┌───▼────────┐     ┌────────▼─────┐
│  Evaluator │     │ Model Wrappers│
│ (Pipeline) │────▶│  (Adapters)   │
└───┬────────┘     └────────┬──────┘
    │                       │
    │              ┌────────┴─┬───────┬───────────┐
    │              │          │       │           │
┌───▼────────┐ ┌───▼──┐ ┌─────▼─┐ ┌───▼───┐ ┌─────▼─────┐
│  Metrics   │ │ CLIP │ │ BLIP2 │ │ LLaVA │ │VLM_Chatbot│
│ Calculator │ └──────┘ └───────┘ └───────┘ └───────────┘
└────────────┘
```

### Engineering Highlights

#### Memory Management
- **Aggressive Cleanup**: Explicit `del`, garbage collection, CUDA cache clearing
- **Lazy Loading**: Models loaded on-demand, unloaded immediately post-evaluation
- **Quantization**: FP16, INT8, INT4 support via `bitsandbytes` and `transformers`
- **Offloading**: Automatic CPU/disk offloading for models exceeding VRAM

#### Performance Optimization
- **GPU Synchronization**: Accurate timing via `torch.cuda.synchronize()`
- **Warmup Runs**: Eliminate JIT compilation overhead from measurements
- **Batching**: Dynamic batch sizing based on available memory
- **Caching**: Image preprocessing results cached to avoid redundant computation

#### Code Quality
- **Type Hints**: Full type annotations for IDE support and static analysis
- **Error Handling**: Comprehensive try-except with detailed logging
- **Extensibility**: Abstract base class pattern for easy model integration
- **Testing**: Unit tests for metrics calculation and data loading

## Requirements

### System Requirements
- **OS**: Linux (tested on Ubuntu 20.04+), Windows WSL2
- **GPU**: CUDA-compatible GPU with 8GB+ VRAM (16GB recommended for LLaVA)
- **RAM**: 16GB+ system memory
- **Storage**: 50GB free space (for model weights and cache)

### Software Dependencies
```txt
torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0
bitsandbytes>=0.40.0
pillow>=9.0.0
tqdm>=4.65.0
numpy>=1.24.0
```

### Model Downloads
Models are automatically downloaded from Hugging Face Hub on first run:
- CLIP ViT-L/14: ~1.7GB
- BLIP-2 FlanT5-XL: ~15GB
- LLaVA-v1.5-7B: ~13.5GB


## Sample Results

| Model | Accuracy | Latency (ms) | Memory (GB) | Strengths |
|-------|----------|-------------|-------------|-----------|
| CLIP | 18.9% | 45.8 | 2.1 | Fast inference, low memory |
| BLIP-2 | 42.3% | 187.4 | 8.6 | Balanced performance |
| LLaVA | 68.2% | 234.5 | 6.8 | Best accuracy, instruction following |
| VLM_Chatbot | 30.2% | 175.2 | 6.2 | Best tradeoff model |

*Results on VQA v2.0 validation subset (330 samples)*
