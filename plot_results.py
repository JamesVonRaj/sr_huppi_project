"""
plot_results.py

Visualizes the Scale-Rich vs HuPPI metrics comparison results.
Generates publication-quality plots for Total Effective Resistance (TER)
and Mean Ollivier-Ricci Curvature (ORC) against heterogeneity parameter alpha.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path

# Ensure figures directory exists
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_results(filepath: str = "sr_huppi_comparison_results.csv") -> pd.DataFrame:
    """Load the experiment results from CSV."""
    if not Path(filepath).exists():
        print(f"Error: Results file '{filepath}' not found.")
        print("Please run main.py first to generate the results.")
        sys.exit(1)
    
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} data points from {filepath}")
    return df


def create_comparison_plots(df: pd.DataFrame, output_path: str = None):
    """
    Create a figure with two subplots comparing SR metrics across heterogeneity values.
    
    Left plot: Total Effective Resistance vs. Alpha
    Right plot: Mean Ollivier-Ricci Curvature vs. Alpha
    """
    # Set publication-quality style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
    })
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Color palette - using a sophisticated scientific palette
    palette = sns.color_palette("deep")
    ter_color = palette[0]  # Blue
    orc_color = palette[3]  # Red-orange
    
    # --- Left Plot: Total Effective Resistance ---
    ax1 = axes[0]
    ax1.plot(df['alpha'], df['TER'], 
             marker='o', markersize=8, linewidth=2.5,
             color=ter_color, markeredgecolor='white', markeredgewidth=1.5)
    ax1.fill_between(df['alpha'], df['TER'], alpha=0.15, color=ter_color)
    
    ax1.set_xlabel(r'Heterogeneity Parameter ($\alpha$)', fontsize=12, fontweight='medium')
    ax1.set_ylabel('Total Effective Resistance (TER)', fontsize=12, fontweight='medium')
    ax1.set_title('Total Effective Resistance vs. Heterogeneity', 
                  fontsize=13, fontweight='bold', pad=10)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    # Add grid styling
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_axisbelow(True)
    
    # --- Right Plot: Mean Ollivier-Ricci Curvature ---
    ax2 = axes[1]
    ax2.plot(df['alpha'], df['Mean_ORC'], 
             marker='s', markersize=8, linewidth=2.5,
             color=orc_color, markeredgecolor='white', markeredgewidth=1.5)
    ax2.fill_between(df['alpha'], df['Mean_ORC'], alpha=0.15, color=orc_color)
    
    ax2.set_xlabel(r'Heterogeneity Parameter ($\alpha$)', fontsize=12, fontweight='medium')
    ax2.set_ylabel('Mean Ollivier-Ricci Curvature', fontsize=12, fontweight='medium')
    ax2.set_title('Mean Ollivier-Ricci Curvature vs. Heterogeneity', 
                  fontsize=13, fontweight='bold', pad=10)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    # Add horizontal line at y=0 for curvature reference
    ax2.axhline(y=0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Add grid styling
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout(pad=2.0)
    
    # Save figure
    if output_path is None:
        output_path = FIGURES_DIR / "sr_metrics_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")
    
    # Also display if running interactively
    plt.show()


def main():
    """Main entry point for plotting results."""
    print("=" * 60)
    print("Scale-Rich vs HuPPI Metrics Visualization")
    print("=" * 60)
    
    # Load results
    df = load_results()
    
    # Display summary statistics
    print("\nData Summary:")
    print("-" * 40)
    print(df.to_string(index=False))
    print("-" * 40)
    
    # Create plots
    create_comparison_plots(df)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

