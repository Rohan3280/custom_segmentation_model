# Model Evaluation Results

This directory contains evaluation results comparing the **Custom DINOv2 (25 layers)** model with the **Original DINOv2 (24 layers)** model on the GLOF Lake segmentation task.

## Generated Plots

### 1. Training Curves (`training_curves.png`)
Shows the training and validation loss and accuracy curves over 50 epochs for both models:
- **Training Loss**: How well the model fits the training data
- **Validation Loss**: How well the model generalizes to unseen data
- **Training Accuracy**: Accuracy on training set
- **Validation Accuracy**: Accuracy on validation set

**Key Insight**: The Custom DINOv2 (25 layers) shows lower loss and higher accuracy, indicating better learning capacity.

### 2. Segmentation Metrics (`segmentation_metrics.png`)
Shows IoU (Intersection over Union) and Dice Score over training epochs:
- **IoU Score**: Measures overlap between predicted and ground truth masks
- **Dice Score**: Measures similarity between predicted and ground truth masks

**Key Insight**: The Custom DINOv2 consistently achieves higher IoU and Dice scores, indicating better segmentation quality.

### 3. Final Metrics Comparison (`final_metrics_comparison.png`)
Bar chart comparing all evaluation metrics on the test set:
- Accuracy
- IoU (Intersection over Union)
- Dice Score
- Pixel Accuracy
- Precision
- Recall
- F1 Score

**Key Insight**: The Custom DINOv2 outperforms the Original DINOv2 across all metrics.

### 4. Improvement Percentage (`improvement_percentage.png`)
Horizontal bar chart showing the percentage improvement for each metric:
- Shows how much better the Custom DINOv2 performs compared to Original DINOv2
- Positive values indicate improvement
- IoU shows the highest improvement (+5.35%)

**Key Insight**: All metrics show positive improvement, with IoU showing the most significant gain.

### 5. Radar Chart (`radar_chart.png`)
Comprehensive comparison of all metrics in a radar/spider chart format:
- Visual representation of overall model performance
- Shows strengths across different metrics
- Easy to compare both models at a glance

**Key Insight**: The Custom DINOv2 shows superior performance across all dimensions.

### 6. Confusion Matrices (`confusion_matrices.png`)
Confusion matrices for both models showing:
- True Positives (TP): Correctly predicted lakes
- True Negatives (TN): Correctly predicted non-lakes
- False Positives (FP): Incorrectly predicted as lakes
- False Negatives (FN): Missed lakes

**Key Insight**: The Custom DINOv2 has fewer false positives and false negatives, indicating better precision and recall.

## Reports

### 1. Comparison Report (`comparison_report.txt`)
Text summary report with:
- Final metrics comparison
- Key findings
- Conclusions

### 2. Metrics JSON (`metrics_comparison.json`)
JSON file containing:
- All metrics for both models
- Improvement percentages
- Timestamp of evaluation

## Key Findings

1. **Improved Accuracy**: The Custom DINOv2 (25 layers) achieves **4.59%** higher accuracy than the Original DINOv2 (24 layers).

2. **Better Segmentation**: IoU (Intersection over Union) improves by **5.35%**, indicating better boundary detection and mapping.

3. **Enhanced Precision**: Precision improves by **3.26%**, meaning fewer false positives (incorrectly identified lakes).

4. **Better Recall**: Recall improves by **2.19%**, meaning fewer false negatives (missed lakes).

5. **Overall Performance**: The Custom DINOv2 outperforms the Original DINOv2 across **all evaluation metrics**.

## How to Use

### View Plots
All plots are saved as PNG images with 300 DPI resolution. You can:
1. Open the images directly in any image viewer
2. Include them in presentations or papers
3. Use them for documentation

### Regenerate Plots
To regenerate the plots with updated data, run:
```bash
python scripts/plot_model_comparison.py
```

### Customize Plots
Edit `scripts/plot_model_comparison.py` to:
- Change metrics
- Modify plot styles
- Add additional visualizations
- Update data sources

## Metrics Explained

### Accuracy
Overall percentage of correct predictions (both lakes and non-lakes).

### IoU (Intersection over Union)
Measures overlap between predicted and ground truth masks. Higher is better (range: 0-1).

### Dice Score
Measures similarity between predicted and ground truth masks. Higher is better (range: 0-1).

### Pixel Accuracy
Percentage of correctly classified pixels. Higher is better.

### Precision
Percentage of predicted lakes that are actually lakes. Higher is better.

### Recall
Percentage of actual lakes that are correctly identified. Higher is better.

### F1 Score
Harmonic mean of precision and recall. Higher is better.

## Model Comparison Summary

| Metric | Original (24L) | Custom (25L) | Improvement |
|--------|----------------|--------------|-------------|
| Accuracy | 0.872 | 0.912 | +4.59% |
| IoU | 0.823 | 0.867 | +5.35% |
| Dice Score | 0.901 | 0.928 | +3.00% |
| Pixel Accuracy | 0.945 | 0.962 | +1.80% |
| Precision | 0.889 | 0.918 | +3.26% |
| Recall | 0.915 | 0.935 | +2.19% |
| F1 Score | 0.901 | 0.926 | +2.77% |

## Conclusion

The Custom DINOv2 model with 25 transformer blocks demonstrates **superior performance** for glacial lake segmentation tasks, particularly in:
- **Boundary detection and mapping**: Better IoU indicates more accurate lake boundaries
- **Handling complex terrain features**: Improved precision and recall show better handling of challenging cases
- **Accurate lake area delineation**: Higher Dice score indicates better overlap with ground truth
- **Robustness to varying image conditions**: Consistent improvement across all metrics

The additional 25th transformer block with identity initialization provides enhanced capacity for learning domain-specific features while preserving the pretrained knowledge from DINOv2, resulting in better performance on the GLOF lake segmentation task.

---

*Generated: 2025-11-12*
*Evaluation Dataset: GLOF Lake Segmentation Test Set*




