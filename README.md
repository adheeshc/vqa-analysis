# VQA Model Comparison Project

## Quick Start

1. **Setup** (already done if you ran quick_start.py):
   ```bash
   python quick_start.py
   ```

2. **Download VQA samples**:
   ```bash
   python setup_data.py
   ```

3. **Run evaluation**:
   ```bash
   python run_evaluation.py --models all
   ```

4. **Generate visualizations**:
   ```bash
   python generate_visualizations.py
   ```

5. **Create report**:
   ```bash
   python generate_report.py
   ```

## Project Structure

```
vqa-comparison/
├── data/                    # VQA samples and images
├── models/                  # Model wrappers
├── evaluation/              # Evaluation pipeline
├── analysis/                # Analysis and visualization
├── results/                 # Outputs
└── notebooks/              # Jupyter notebooks
```

## Models

- **CLIP**: Zero-shot classification
- **BLIP-2**: Q-Former bridging
- **LLaVA**: Instruction-tuned

## Evaluation Metrics

- Accuracy (VQA metric)
- Inference latency
- Memory usage
- Performance by question type

## Results

Results will be saved in:
- `results/predictions/` - Model predictions
- `results/benchmarks/` - Performance metrics
- `results/visualizations/` - Charts and plots
- `results/VQA_Comparison_Report.md` - Final report
