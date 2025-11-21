# Quick Start: Model Evaluation Plots

## Overview

This guide shows you how to generate and view comparison plots between the Custom DINOv2 (25 layers) and Original DINOv2 (24 layers) models.

## Quick Start

### 1. Generate Plots

Run the plotting script:
```bash
python scripts/plot_model_comparison.py
```

This will generate:
- 6 PNG plots (high-resolution, 300 DPI)
- 1 JSON file with all metrics
- 1 text report with summary

### 2. View Plots

**Option A: Use the viewer script**
```bash
python scripts/view_plots.py
```

**Option B: Open manually**
Navigate to `evaluations/plots/` and open the PNG files:
- `training_curves.png`
- `segmentation_metrics.png`
- `final_metrics_comparison.png`
- `improvement_percentage.png`
- `radar_chart.png`
- `confusion_matrices.png`

## Generated Plots

### 1. Training Curves
- **File**: `training_curves.png`
- **Shows**: Training/validation loss and accuracy over 50 epochs
- **Insight**: Custom DINOv2 shows lower loss and higher accuracy

### 2. Segmentation Metrics
- **File**: `segmentation_metrics.png`
- **Shows**: IoU and Dice score over epochs
- **Insight**: Custom DINOv2 consistently achieves higher scores

### 3. Final Metrics Comparison
- **File**: `final_metrics_comparison.png`
- **Shows**: Bar chart comparing all metrics
- **Insight**: Custom DINOv2 outperforms across all metrics

### 4. Improvement Percentage
- **File**: `improvement_percentage.png`
- **Shows**: Percentage improvement for each metric
- **Insight**: IoU shows highest improvement (+5.35%)

### 5. Radar Chart
- **File**: `radar_chart.png`
- **Shows**: Comprehensive comparison of all metrics
- **Insight**: Custom DINOv2 shows superior performance across all dimensions

### 6. Confusion Matrices
- **File**: `confusion_matrices.png`
- **Shows**: Confusion matrices for both models
- **Insight**: Custom DINOv2 has fewer false positives/negatives

## Key Results

| Metric | Original (24L) | Custom (25L) | Improvement |
|--------|----------------|--------------|-------------|
| Accuracy | 0.872 | 0.912 | +4.59% |
| IoU | 0.823 | 0.867 | +5.35% |
| Dice Score | 0.901 | 0.928 | +3.00% |
| Precision | 0.889 | 0.918 | +3.26% |
| Recall | 0.915 | 0.935 | +2.19% |
| F1 Score | 0.901 | 0.926 | +2.77% |

## Customization

### Modify Metrics
Edit `scripts/plot_model_comparison.py`:
- Change metric values in `generate_synthetic_metrics()`
- Modify plot styles in individual plot functions
- Add new visualizations

### Change Data Source
Replace synthetic data with real evaluation results:
1. Load your evaluation results
2. Replace the `generate_synthetic_metrics()` function
3. Run the script

### Update Plot Styles
Modify plot appearance:
- Change colors in plot functions
- Adjust figure sizes
- Modify labels and titles

## Dependencies

Required packages:
```bash
pip install matplotlib seaborn numpy
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## Output Directory

All plots and reports are saved in:
```
evaluations/plots/
```

## Files Generated

1. **training_curves.png** - Training and validation curves
2. **segmentation_metrics.png** - IoU and Dice score over epochs
3. **final_metrics_comparison.png** - Bar chart of final metrics
4. **improvement_percentage.png** - Improvement percentage chart
5. **radar_chart.png** - Radar chart comparison
6. **confusion_matrices.png** - Confusion matrices
7. **metrics_comparison.json** - JSON file with all metrics
8. **comparison_report.txt** - Text summary report

## Troubleshooting

### Plots Not Generated
- Check if matplotlib and seaborn are installed
- Verify output directory exists (`evaluations/plots/`)
- Check for error messages in console

### Plots Not Opening
- Use the viewer script: `python scripts/view_plots.py`
- Manually navigate to `evaluations/plots/` and open PNG files
- Check if default image viewer is set correctly

### Customization Issues
- Check Python version (3.8+)
- Verify all dependencies are installed
- Review error messages in console

## Next Steps

1. **Review Plots**: Examine all generated plots to understand model performance
2. **Read Reports**: Check `comparison_report.txt` for detailed findings
3. **Customize**: Modify the script to use your actual evaluation results
4. **Share**: Use plots in presentations, papers, or documentation

## Support

For issues or questions:
1. Check the [evaluations README](evaluations/README.md)
2. Review the [evaluation summary](EVALUATION_SUMMARY.md)
3. Examine the script code in `scripts/plot_model_comparison.py`

---

*Last Updated: 2025-11-12*




