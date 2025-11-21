"""
Simple script to view all generated plots.
Opens all plots in the default image viewer.
"""

import os
import subprocess
import sys
from pathlib import Path

def open_plots():
    """Open all generated plots in the default image viewer."""
    plots_dir = Path("evaluations/plots")
    
    if not plots_dir.exists():
        print(f"Error: Directory {plots_dir} does not exist.")
        print("Please run 'python scripts/plot_model_comparison.py' first.")
        return
    
    # List of plot files
    plot_files = [
        "training_curves.png",
        "segmentation_metrics.png",
        "final_metrics_comparison.png",
        "improvement_percentage.png",
        "radar_chart.png",
        "confusion_matrices.png"
    ]
    
    print("Opening plots...")
    print("=" * 70)
    
    opened_count = 0
    for plot_file in plot_files:
        plot_path = plots_dir / plot_file
        if plot_path.exists():
            try:
                # Open with default system viewer
                if sys.platform == "win32":
                    os.startfile(plot_path)
                elif sys.platform == "darwin":
                    subprocess.run(["open", plot_path])
                else:
                    subprocess.run(["xdg-open", plot_path])
                print(f"[OK] Opened: {plot_file}")
                opened_count += 1
            except Exception as e:
                print(f"[ERROR] Failed to open {plot_file}: {e}")
        else:
            print(f"[WARNING] Plot not found: {plot_file}")
    
    print("=" * 70)
    print(f"Opened {opened_count}/{len(plot_files)} plots.")
    print("\nNote: On some systems, all plots may open at once.")
    print("If plots don't open automatically, navigate to: evaluations/plots/")

if __name__ == "__main__":
    open_plots()




